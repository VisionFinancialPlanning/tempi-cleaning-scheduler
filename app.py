# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import re, unicodedata
import urllib.parse as _q
from datetime import datetime, time, timedelta

import pytz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype as _is_dt

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
TZ_NAME = "America/Panama"
TZ = pytz.timezone(TZ_NAME)

st.set_page_config(page_title="Tempi – Scheduler & Bookings", page_icon="🧹", layout="wide")
st.title("Tempi – Scheduler & Bookings 🧹📒")
st.caption(f"Zona horaria aplicada: {TZ_NAME}")

# ==============================
# HELPERS
# ==============================
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _find_col(df: pd.DataFrame, aliases):
    norm_map = {_norm(c): c for c in df.columns}
    for a in aliases:
        an = _norm(a)
        if an in norm_map:
            return norm_map[an]
    for a in aliases:
        an = _norm(a)
        for key, real in norm_map.items():
            if an in key:
                return real
    return None

def _coerce_time(series):
    if series is None:
        return pd.Series(dtype="object")
    def parse_one(v):
        if pd.isna(v): return None
        if isinstance(v, (pd.Timestamp, datetime)): return v.time()
        if isinstance(v, (int, float)):
            # Excel time serial
            try:
                frac = float(v) % 1.0
                secs = int(round(frac * 24 * 3600))
                hh, mm, ss = secs // 3600, (secs % 3600) // 60, secs % 60
                return time(hh % 24, mm, ss)
            except Exception:
                return None
        s = str(v).strip().replace(".", ":")
        try:
            t = pd.to_datetime(s, errors="coerce").time()
            if t is not None: return t
        except Exception:
            pass
        m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*$", s, flags=re.I)
        if m:
            hh = int(m.group(1)); mm = int(m.group(2)); ss = int(m.group(3) or 0)
            ampm = (m.group(4) or "").lower()
            if ampm.startswith("p") and hh < 12: hh += 12
            if ampm.startswith("a") and hh == 12: hh = 0
            return time(hh % 24, mm, ss)
        return None
    return series.apply(parse_one)

def _time_from_datecol(date_s):
    dates = pd.to_datetime(date_s, errors="coerce")
    out = []
    for d in dates:
        if pd.isna(d): out.append(None)
        else:
            hh, mm, ss = d.hour, d.minute, d.second
            out.append(None if (hh, mm, ss) == (0, 0, 0) else time(hh, mm, ss))
    return pd.Series(out, index=dates.index)

def _combine_smart(date_s, explicit_time_s, default_time: time):
    dates = pd.to_datetime(date_s, errors="coerce")
    t1 = _coerce_time(explicit_time_s) if explicit_time_s is not None else pd.Series([None]*len(dates))
    t2 = _time_from_datecol(date_s)
    out = []
    for d, a, b in zip(dates, t1, t2):
        if pd.isna(d): out.append(pd.NaT)
        else:
            tt = a if a is not None else (b if b is not None else default_time)
            out.append(pd.Timestamp.combine(pd.Timestamp(d).date(), tt))
    ser = pd.to_datetime(out, errors="coerce")
    if ser.empty or not _is_dt(ser): return ser
    try:
        ser = ser.dt.tz_localize(TZ_NAME, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        return ser
    return ser

def _fmt_tz(series, fmt):
    try:
        return series.dt.tz_convert(TZ_NAME).dt.strftime(fmt)
    except Exception:
        try: return series.dt.strftime(fmt)
        except Exception: return pd.Series([""]*len(series))

def _parse_money(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val)
    if not re.search(r"\d", s): return np.nan
    s = s.replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try: return float(s)
    except Exception: return np.nan

def _parse_parking_count(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)) and not np.isnan(x): return int(x)
    s = str(x).lower()
    if s in ["", "no", "none", "n/a", "na", "0"]: return 0
    m = re.search(r"(\d+)", s)
    if m: return int(m.group(1))
    return 1

def _is_si(x):
    """True solo si la celda dice 'SI' (acepta Sí/si/yes)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().lower().replace("í", "i")
    return s in {"si", "yes"}

def _day_bounds(day, start_t: time, end_t: time):
    d = pd.Timestamp(day).date()
    return TZ.localize(datetime.combine(d, start_t)), TZ.localize(datetime.combine(d, end_t))

def _canon_apt(x):
    """Limpia nombre de apartamento, elimina vacíos y unifica alias Jerónimo."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan
    norm = _norm(s)
    alias = {
        "jeronimocascoviejo": "Jeronimo Casco Viejo",
        "jeronimocentralandcozyapto": "Jeronimo Casco Viejo",
        "jeronimocentralandcozyapt": "Jeronimo Casco Viejo",
        "jeronimocentralcozyapto": "Jeronimo Casco Viejo",
        "jeronimocentralandcozy": "Jeronimo Casco Viejo",
    }
    return alias.get(norm, s)

def to_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try: s = s.dt.tz_convert(TZ_NAME)
    except Exception:
        try: s = s.dt.tz_localize(TZ_NAME)
        except Exception: pass
    return s

# ==============================
# NORMALIZACIÓN DE EXCEL
# ==============================
def parse_bookings_with_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if _find_col(df, ["Check-In"]) is None or _find_col(df, ["Check-Out"]) is None:
        raise ValueError("El archivo debe incluir al menos 'Check-In' y 'Check-Out'.")

    ci_date_col = _find_col(df, ["Check-In"])
    co_date_col = _find_col(df, ["Check-Out"])

    ci_time_col = _find_col(df, ["Check-In Hora","hora entrada","Hora Entrada","Hora Check In","Check-In Time","Hora Check-In"])
    co_time_col = _find_col(df, ["Check-Out Hora","hora salida","Hora Salida","Hora Check Out","Check-Out Time","Hora Check-Out"])

    ci_dt = _combine_smart(df[ci_date_col], df[ci_time_col] if ci_time_col else None, default_time=time(15,0))
    co_dt = _combine_smart(df[co_date_col], df[co_time_col] if co_time_col else None, default_time=time(12,0))

    ci_from_col  = _coerce_time(df[ci_time_col]).notna() if ci_time_col else pd.Series([False]*len(df))
    co_from_col  = _coerce_time(df[co_time_col]).notna() if co_time_col else pd.Series([False]*len(df))
    ci_from_date = _time_from_datecol(df[ci_date_col]).notna()
    co_from_date = _time_from_datecol(df[co_date_col]).notna()

    def _source(from_col, from_date, default_label):
        return np.where(from_col, "hora_col", np.where(from_date, "en_fecha", default_label))

    out = pd.DataFrame()

    apt_col   = _find_col(df, ["Property Internal Name"])
    guest_col = _find_col(df, ["Guest First Name"])
    if apt_col:
        out["apartment"]  = df[apt_col].apply(_canon_apt)
    if guest_col:
        out["guest_name"] = df[guest_col].astype(str)

    map_optional = {
        "number_guests": ["NUMBER GUESTS","Guests","# Guests","Numero Huespedes","Nº Huespedes"],
        "sofa_or_bed":   ["SOFA OR BED","Sofa Bed","Sofa cama","Sofacama","Sofa as Bed"],
        "reservation_id":["ID","Reservation ID","Booking ID","Código","Codigo"],
        "transporte":    ["TRANSPORTE","Transport","Pickup","Transfer"],
        "extra":         ["EXTRA","Extras","Requests","Solicitudes"],
        "parking":       ["PARKING","Estacionamiento","Parking Spot"],
        "nombre":        ["Nombre","Name","Guest Name"],
        "notes":         ["NOTES","Notas","Observaciones"],
        "channel":       ["Channel","Source","Booking Source","OTA","Origin","Website","Portal","Booked Via"],
        "payment_method":["Payment Method","Payment","Método de Pago","Forma de pago","Payment Type","Pay Method"],
        "amount_due":    ["Amount Due","Balance Due","Balance","Total Due","Pending Amount","Por cobrar","A Cobrar","Saldo","Saldo pendiente","Total Pendiente"],
        "total_price":   ["Total Price","Total","Precio Total","Importe","Booking Amount","Payment Amount","Grand Total","Monto","Monto Total"]
    }
    for new_col, aliases in map_optional.items():
        c = _find_col(df, aliases)
        if c: out[new_col] = df[c]

    # Transporte específico por evento (solo SI)
    trans_ci_col = _find_col(df, ["TRANSPORTE CHECK-IN", "Transporte Check-In", "Transporte Check In", "Pickup Check-In", "Transfer Check-In"])
    trans_co_col = _find_col(df, ["TRANSPORTE CHECK-OUT", "Transporte Check-Out", "Transporte Check Out", "Pickup Check-Out", "Transfer Check-Out"])
    if trans_ci_col: out["transport_ci"] = df[trans_ci_col]
    if trans_co_col: out["transport_co"] = df[trans_co_col]

    out["checkin"]  = ci_dt
    out["checkout"] = co_dt

    out["checkin_day"]   = _fmt_tz(out["checkin"],  "%Y-%m-%d")
    out["checkin_time"]  = _fmt_tz(out["checkin"],  "%H:%M")
    out["checkout_day"]  = _fmt_tz(out["checkout"], "%Y-%m-%d")
    out["checkout_time"] = _fmt_tz(out["checkout"], "%H:%M")

    try:
        nights = out["checkout"].dt.tz_convert(TZ_NAME).dt.date - out["checkin"].dt.tz_convert(TZ_NAME).dt.date
    except Exception:
        nights = pd.to_datetime(out["checkout"], errors="coerce").dt.date - pd.to_datetime(out["checkin"], errors="coerce").dt.date
    out["nights"] = [d.days if pd.notna(d) else None for d in nights]

    out["checkin_time_source"]  = _source(ci_from_col, ci_from_date, "default 15:00")
    out["checkout_time_source"] = _source(co_from_col, co_from_date, "default 12:00")

    # CASH detection
    def _needs_cash(row):
        reason = []
        ch = str(row.get("channel","")).lower()
        pm = str(row.get("payment_method","")).lower()
        nt = str(row.get("notes","")).lower()
        if "booking" in ch: reason.append("canal=Booking")
        if any(w in pm for w in ["cash","efectivo"]): reason.append("pago=cash")
        if any(w in nt for w in ["cash","efectivo"]): reason.append("nota menciona cash")
        needs = len(reason) > 0
        amt = _parse_money(row.get("amount_due", np.nan))
        if np.isnan(amt): amt = _parse_money(row.get("total_price", np.nan))
        return needs, ", ".join(reason), (amt if needs else np.nan)

    if len(out):
        cash, why, amt = zip(*[_needs_cash(r) for _, r in out.iterrows()])
        out["cash_pickup"] = list(cash)
        out["cash_reason"] = list(why)
        out["cash_amount"] = list(amt)
    else:
        out["cash_pickup"] = []; out["cash_reason"] = []; out["cash_amount"] = []

    preferred = [
        "apartment","guest_name","number_guests","sofa_or_bed","reservation_id",
        "transporte","extra","parking","nombre","notes",
        "channel","payment_method","amount_due","total_price",
        "cash_pickup","cash_amount","cash_reason",
        "checkin_day","checkin_time","checkin_time_source",
        "checkout_day","checkout_time","checkout_time_source",
        "nights","checkin","checkout",
        "transport_ci","transport_co"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

# ==============================
# EVENTOS / VENTANAS / RESUMEN
# ==============================
def _events_por_apartamento(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time, extra_apartments=None):
    """Devuelve dict por apto: cis, cos, ocupado, ventana=(ini, fin, tipo) con tipo∈{turnover, solo checkout, solo check-in, vacio}."""
    start_of_day, end_of_day = _day_bounds(day, day_start_t, day_end_t)
    df = normalized.copy()
    df["checkin"]  = to_tz(df["checkin"])
    df["checkout"] = to_tz(df["checkout"])

    apts = sorted(df["apartment"].dropna().astype(str).unique().tolist())
    if extra_apartments:
        for x in extra_apartments:
            if x not in apts: apts.append(x)

    out = {}
    for apt in apts:
        sub = df[df["apartment"].astype(str) == apt]
        cis = [ts for ts in sub["checkin"].dropna()  if ts.date()  == start_of_day.date()]
        cos = [ts for ts in sub["checkout"].dropna() if ts.date()  == start_of_day.date()]
        cis = sorted([ts.tz_convert(TZ_NAME) for ts in cis])
        cos = sorted([ts.tz_convert(TZ_NAME) for ts in cos])

        pisas = sub[(sub["checkin"] < end_of_day) & (sub["checkout"] > start_of_day)]
        ocupado = not pisas.empty

        ventana = None
        if cis and cos:
            co = max(cos); ci = min(cis)
            if ci > co: ventana = (co, ci, "turnover")
        elif cos:
            co = max(cos)
            if end_of_day > co: ventana = (co, end_of_day, "solo checkout")
        elif cis:
            ci = min(cis)
            if ci > start_of_day: ventana = (start_of_day, ci, "solo check-in")
        elif not ocupado:
            ventana = (start_of_day, end_of_day, "vacio")

        out[apt] = {"cis": cis, "cos": cos, "ocupado": ocupado, "ventana": ventana}
    return out

def day_summary_collapsed_v2(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time, extra_apts=None) -> pd.DataFrame:
    ev = _events_por_apartamento(normalized, day, day_start_t, day_end_t, extra_apts)
    rows = []
    for apt, d in ev.items():
        cis, cos, v = d["cis"], d["cos"], d["ventana"]
        def _fmt(ts_list): return ", ".join(sorted({t.strftime("%H:%M") for t in ts_list})) if ts_list else ""
        if v is None:  # ocupado pero sin ventana útil
            continue
        ini, fin, kind = v
        if kind == "turnover":
            gap = round((fin - ini).total_seconds()/3600.0, 2)
            rows.append({"apartment": apt, "tipo": "Turnover", "checkout_time": _fmt(cos), "checkin_time": _fmt(cis), "gap_hours": gap})
        elif kind == "solo checkout":
            rows.append({"apartment": apt, "tipo": "Solo Check-Out", "checkout_time": _fmt(cos), "checkin_time": "", "gap_hours": ""})
        elif kind == "solo check-in":
            rows.append({"apartment": apt, "tipo": "Solo Check-In", "checkout_time": "", "checkin_time": _fmt(cis), "gap_hours": ""})
        elif kind == "vacio":
            rows.append({"apartment": apt, "tipo": "Vacío (día completo)", "checkout_time": "", "checkin_time": "", "gap_hours": 24})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["apartment","tipo","checkout_time","checkin_time","gap_hours"])

    df = df[df["apartment"].notna()].copy()

    # BALI al final del resumen
    df["__bali__"] = df["apartment"].astype(str).eq("BALI")
    df = df.sort_values(["__bali__", "tipo", "apartment"], ascending=[True, True, True]).drop(columns="__bali__")
    return df.reset_index(drop=True)

def build_apartment_windows_v2(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time, extra_apts=None) -> pd.DataFrame:
    ev = _events_por_apartamento(normalized, day, day_start_t, day_end_t, extra_apts)
    rows = []
    for apt, d in ev.items():
        v = d["ventana"]
        if v is None: continue
        ini, fin, kind = v
        if fin > ini:
            rows.append({"apartment": apt, "start": ini, "end": fin, "kind": kind})
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["__bali__"] = out["apartment"].astype(str).eq("BALI")   # BALI al final también en el Gantt
    out = out.sort_values(["__bali__", "apartment"]).drop(columns="__bali__").reset_index(drop=True)
    return out

# ==============================
# PLANIFICADOR (opcional)
# ==============================
class EmployeeTimeline:
    def __init__(self, name, start_t: time, end_t: time, lunch_start: time, lunch_end: time, day):
        self.name = name; self.day = pd.Timestamp(day).date()
        self.start_dt = TZ.localize(datetime.combine(self.day, start_t))
        self.end_dt   = TZ.localize(datetime.combine(self.day, end_t))
        self.lunch_start = TZ.localize(datetime.combine(self.day, lunch_start))
        self.lunch_end   = TZ.localize(datetime.combine(self.day, lunch_end))
        self.cursor = self.start_dt; self.slots = []
    def _overlaps_lunch(self, start, end): return not (end <= self.lunch_start or start >= self.lunch_end)
    def trial(self, job, buffer_min=10, travel_min=0):
        est = max(self.cursor, job["start_window"]) + pd.Timedelta(minutes=travel_min)
        end = est + pd.Timedelta(minutes=job["duration_min"])
        if self._overlaps_lunch(est, end): return None
        deadline = job["deadline"] if pd.notna(job["deadline"]) else self.end_dt
        if end > deadline or end > self.end_dt: return None
        return {"employee": self.name, "apartment": job["apartment"], "unit_id": job["unit_id"],
                "guest_name": job.get("guest_name",""), "start": est, "end": end, "duration_min": job["duration_min"]}
    def schedule(self, job, buffer_min=10, travel_min=0):
        t = self.trial(job, buffer_min, travel_min)
        if t is None: return None
        self.slots.append(t); self.cursor = t["end"] + pd.Timedelta(minutes=buffer_min+travel_min); return t

def greedy_assign(jobs_df, employees, buffer_min=10, travel_min=0, early_priority=True):
    if jobs_df is None or jobs_df.empty:
        return pd.DataFrame(columns=["employee","apartment","unit_id","guest_name","start","end","duration_min"]), pd.DataFrame()
    sort_keys = jobs_df.copy()
    min_day = sort_keys["start_window"].dt.date.min()
    far = TZ.localize(datetime.combine(min_day, time(23, 59)))
    sort_keys["deadline_filled"] = sort_keys["deadline"].fillna(far)
    jobs_sorted = sort_keys.sort_values(
        ["deadline_filled","start_window","duration_min"] if early_priority
        else ["start_window","deadline_filled","duration_min"]
    ).drop(columns=["deadline_filled"])
    assignments, unassigned = [], []
    for _, job in jobs_sorted.iterrows():
        candidates = []
        for emp in employees:
            t = emp.trial(job, buffer_min=buffer_min, travel_min=travel_min)
            if t is not None: candidates.append((emp, t))
        if not candidates:
            unassigned.append(job.to_dict()); continue
        emp_sel, slot = sorted(candidates, key=lambda x: x[1]["end"])[0]
        emp_sel.schedule(job, buffer_min=buffer_min, travel_min=travel_min); assignments.append(slot)
    return pd.DataFrame(assignments), pd.DataFrame(unassigned)

def plot_gantt(df: pd.DataFrame, title, y_key):
    if df is None or df.empty:
        st.info("No hay datos para mostrar."); return
    fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(df)))
    labels = list(df[y_key].unique())
    y_map = {e:i for i,e in enumerate(labels)}
    day0 = df["start"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    for _, row in df.iterrows():
        y = y_map[row[y_key]]
        start = row["start"].to_pydatetime(); end = row["end"].to_pydatetime()
        left = (start - day0).total_seconds()/3600; width = (end-start).total_seconds()/3600
        ax.barh(y, width, left=left)
        label = row.get("apartment", row.get("employee",""))
        ax.text(left+width/2, y, f"{label}", va="center", ha="center", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Horas del día"); ax.set_title(title); st.pyplot(fig)

# ==============================
# SIDEBAR (config)
# ==============================
with st.sidebar:
    st.header("⚙️ Configuración del Día")
    work_date = st.date_input("Día a planificar (y preparar)", value=pd.Timestamp.now(TZ).date())

    st.subheader("Jornada operativa")
    day_start_t = st.time_input("Inicio del día", value=time(8,0))
    day_end_t   = st.time_input("Fin del día", value=time(18,0))

    st.subheader("Empleadas")
    emp1_name = st.text_input("Empleado 1", value="Mayerlin")
    emp2_name = st.text_input("Empleado 2", value="Evelyn")

    def shift_block(label, default_start=time(8,0), default_end=time(17,0)):
        c1, c2 = st.columns(2)
        with c1: s = st.time_input(f"{label} – inicio", value=default_start)
        with c2: e = st.time_input(f"{label} – fin", value=default_end)
        return s, e
    e1_start, e1_end = shift_block("Empleado 1", time(8,0), time(17,0))
    e2_start, e2_end = shift_block("Empleado 2", time(8,0), time(17,0))

    st.subheader("Almuerzo (movible)")
    e1_l1, e1_l2 = shift_block("Empleado 1 – Almuerzo", time(11,0), time(12,0))
    e2_l1, e2_l2 = shift_block("Empleado 2 – Almuerzo", time(11,0), time(12,0))

    st.subheader("Parámetros del Plan")
    buffer_minutes  = st.number_input("Buffer entre limpiezas (min)", value=10, step=5)
    travel_minutes  = st.number_input("Traslado entre apartamentos (min)", value=0, step=5)
    default_duration= st.number_input("Duración limpieza completa (min)", value=90, step=5)
    mop_minutes     = st.number_input("Duración mopa (solo check-in) (min)", value=20, step=5)
    early_priority  = st.checkbox("Priorizar salidas tempranas (deadline primero)", value=True)

    st.subheader("WhatsApp (opcional)")
    wa_emp1 = st.text_input("WhatsApp Empleado 1 (solo dígitos)", value="")
    wa_emp2 = st.text_input("WhatsApp Empleado 2 (solo dígitos)", value="")

# ==============================
# CARGA ARCHIVO
# ==============================
uploaded = st.file_uploader(
    "Sube tu Excel (.xlsx). Requeridas: 'Check-In', 'Check-Out' (horas con alias aceptados).",
    type=["xlsx"]
)
if uploaded is not None:
    df_raw = pd.read_excel(uploaded)
else:
    try:
        st.info("Usando datos de ejemplo: sample_bookings.xlsx")
        df_raw = pd.read_excel("sample_bookings.xlsx")
    except Exception:
        st.error("No subiste archivo y no se encontró sample_bookings.xlsx en el repo."); st.stop()

try:
    normalized = parse_bookings_with_fixed_columns(df_raw)
except Exception as e:
    st.error(f"Error al normalizar reservas: {e}"); st.stop()

# ==============================
# TABS
# ==============================
tab1, tab2 = st.tabs(["🧭 Operativa del día", "📈 Analítica mensual"])

# ====== TAB 1: Operativa del día ======
with tab1:
    today_str = pd.Timestamp(work_date).strftime("%Y-%m-%d")

    st.subheader("📦 Preparación del día (mismo día)")
    prep_df = normalized[(normalized["checkin_day"] == today_str) | (normalized["checkout_day"] == today_str)].copy()
    if prep_df.empty:
        st.info("No hay movimientos para la fecha seleccionada.")
    else:
        cols_show = [c for c in [
            "apartment","guest_name","number_guests","sofa_or_bed","reservation_id",
            "transporte","extra","parking","nombre","notes",
            "channel","payment_method","amount_due","total_price",
            "checkin_day","checkin_time","checkout_day","checkout_time"
        ] if c in prep_df.columns]
        st.dataframe(prep_df[cols_show], use_container_width=True)

    st.subheader("💵 Recolección de CASH (en checkout de hoy)")
    cash_pickups = normalized[(normalized.get("cash_pickup", False) == True) & (normalized["checkout_day"] == today_str)]  # noqa: E712
    if cash_pickups.empty:
        st.success("No hay recolecciones de CASH para hoy.")
    else:
        show_cols = [c for c in ["apartment","guest_name","checkout_time","cash_amount","cash_reason"] if c in cash_pickups.columns]
        st.warning("Coordinar recolección en estos apartamentos:")
        st.dataframe(cash_pickups[show_cols], use_container_width=True)

    # 🚐 Agendar Transporte (hoy) — SOLO si la columna correspondiente dice "SI"
    st.subheader("🚐 Agendar Transporte (hoy)")
    trans_df = prep_df.copy()
    has_ci = "transport_ci" in trans_df.columns
    has_co = "transport_co" in trans_df.columns

    def _hora_from_cols(row, kind):
        if kind == "ci":
            if "checkin_time" in row and pd.notna(row["checkin_time"]) and str(row["checkin_time"]).strip():
                return row["checkin_time"]
            if "checkin" in row and pd.notna(row["checkin"]):
                try: return row["checkin"].tz_convert(TZ_NAME).strftime("%H:%M")
                except Exception: return pd.to_datetime(row["checkin"]).strftime("%H:%M")
        else:
            if "checkout_time" in row and pd.notna(row["checkout_time"]) and str(row["checkout_time"]).strip():
                return row["checkout_time"]
            if "checkout" in row and pd.notna(row["checkout"]):
                try: return row["checkout"].tz_convert(TZ_NAME).strftime("%H:%M")
                except Exception: return pd.to_datetime(row["checkout"]).strftime("%H:%M")
        return ""

    if has_ci or has_co:
        out_rows = []
        if has_ci:
            df_ci = trans_df[(trans_df["checkin_day"] == today_str) & (trans_df["transport_ci"].apply(_is_si))].copy()
            for _, r in df_ci.iterrows():
                out_rows.append({"hora": _hora_from_cols(r, "ci"), "accion": "CHECK-IN",
                                 "apartment": r.get("apartment",""), "guest_name": r.get("guest_name","")})
        if has_co:
            df_co = trans_df[(trans_df["checkout_day"] == today_str) & (trans_df["transport_co"].apply(_is_si))].copy()
            for _, r in df_co.iterrows():
                out_rows.append({"hora": _hora_from_cols(r, "co"), "accion": "CHECK-OUT",
                                 "apartment": r.get("apartment",""), "guest_name": r.get("guest_name","")})
        trans_out = pd.DataFrame(out_rows, columns=["hora","accion","apartment","guest_name"])
        if trans_out.empty:
            st.info("No hay transportes para agendar hoy.")
        else:
            st.dataframe(trans_out.sort_values(["hora","accion","apartment"]), use_container_width=True)
    else:
        st.info("No se encontraron columnas ‘TRANSPORTE CHECK-IN’ / ‘TRANSPORTE CHECK-OUT’ en el archivo.")

    # 🅿️ Parking (cantidad de carros)
    st.subheader("🅿️ Parking – Check-ins de hoy")
    park_df = prep_df[prep_df["checkin_day"] == today_str].copy()
    if "parking" in park_df.columns and not park_df.empty:
        park_df["carros"] = park_df["parking"].apply(_parse_parking_count)
        park_out = park_df[["apartment","guest_name","checkin_time","carros","parking"]].rename(columns={"parking":"detalle"})
        park_out = park_out[park_out["carros"].notna()]
        if park_out.empty:
            st.info("No hay solicitudes de parking para hoy.")
        else:
            st.dataframe(park_out.sort_values(["checkin_time","apartment"]), use_container_width=True)
    else:
        st.info("No hay datos de parking para hoy.")

    # ========================= Resumen + ASIGNACIÓN MANUAL =========================
    st.subheader("🧾 Resumen por apartamento (día) + asignación manual")
    extra_apts = ["BALI"]  # apartamento extra fijo
    summary_df = day_summary_collapsed_v2(normalized, work_date, day_start_t, day_end_t, extra_apts)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("**Asigna quién y a qué hora limpiar (manual)**")
    # Ventanas por apto (para referencia)
    win_df = build_apartment_windows_v2(normalized, work_date, day_start_t, day_end_t, extra_apts)
    win_map = {r["apartment"]: r for _, r in win_df.iterrows()}
    def _fmt(t): return t.tz_convert(TZ_NAME).strftime("%H:%M") if pd.notna(t) else ""

    base_rows = []
    for _, r in summary_df.iterrows():
        apt = r["apartment"]
        w = win_map.get(apt)
        v_ini_str = _fmt(w["start"]) if w is not None else ""
        v_fin_str = _fmt(w["end"]) if w is not None else ""
        base_rows.append({
            "apartment": apt,
            "tipo": r["tipo"],
            "ventana_inicio": v_ini_str,
            "ventana_fin": v_fin_str,
            "empleado": "—",
            "inicio": "",
            "fin": ""
        })
    base_df = pd.DataFrame(base_rows)

    # Opciones de horas (cada 15 min)
    def _time_opts(t0: time, t1: time, step_min=15):
        cur = datetime.combine(datetime.today().date(), t0)
        end = datetime.combine(datetime.today().date(), t1)
        out = []
        while cur <= end:
            out.append(cur.strftime("%H:%M"))
            cur += timedelta(minutes=step_min)
        return out
    time_options = _time_opts(day_start_t, day_end_t, 15)
    employee_options = ["—", emp1_name, emp2_name]

    edited = st.data_editor(
        base_df,
        column_config={
            "apartment": st.column_config.Column("apartment", disabled=True),
            "tipo": st.column_config.Column("tipo", disabled=True),
            "ventana_inicio": st.column_config.Column("ventana_inicio", disabled=True),
            "ventana_fin": st.column_config.Column("ventana_fin", disabled=True),
            "empleado": st.column_config.SelectboxColumn("empleado", options=employee_options),
            "inicio": st.column_config.SelectboxColumn("inicio", options=time_options),
            "fin": st.column_config.SelectboxColumn("fin", options=time_options),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )

    # Convertimos selecciones a plan manual
    def _to_dt(day, hhmm):
        if not hhmm or str(hhmm).strip() == "": return pd.NaT
        t = datetime.strptime(hhmm, "%H:%M").time()
        return TZ.localize(datetime.combine(pd.Timestamp(day).date(), t))

    manual_rows = []
    for _, r in edited.iterrows():
        emp = r.get("empleado", "—")
        s = _to_dt(work_date, r.get("inicio","")); e = _to_dt(work_date, r.get("fin",""))
        if emp != "—" and pd.notna(s) and pd.notna(e) and e > s:
            manual_rows.append({
                "employee": emp,
                "apartment": r["apartment"],
                "start": s,
                "end": e,
                "duration_min": int((e - s).total_seconds()//60)
            })
    manual_plan = pd.DataFrame(manual_rows)

    st.subheader("🗓️ Gantt manual (según tus horas)")
    if manual_plan.empty:
        st.info("Asigna empleado e intervalos (inicio/fin) para ver el Gantt.")
    else:
        plot_gantt(manual_plan, title="Plan Manual de Limpieza (Gantt)", y_key="employee")

    # 💬 Mensajes WhatsApp por empleada
    st.subheader("💬 Mensajes de WhatsApp (según plan manual)")
    if manual_plan.empty:
        st.info("Asigna empleado e intervalos (inicio/fin) en la tabla para generar los mensajes.")
    else:
        plan_msg = manual_plan.merge(summary_df[["apartment","tipo"]], on="apartment", how="left")

        # CASH por apartamento (checkout de hoy)
        cash_map = {}
        if 'apartment' in cash_pickups.columns:
            tmp_cash = cash_pickups.copy()
            tmp_cash["co_hora"] = tmp_cash.get("checkout_time", "")
            for _, rr in tmp_cash.iterrows():
                key = rr.get("apartment", "")
                cash_map.setdefault(key, [])
                cash_map[key].append({
                    "amount": rr.get("cash_amount", np.nan),
                    "hora": rr.get("co_hora", "")
                })

        def _fmt_h(dt): 
            try: return dt.tz_convert(TZ_NAME).strftime("%H:%M")
            except Exception: return pd.to_datetime(dt).strftime("%H:%M")

        for emp, g in plan_msg.sort_values("start").groupby("employee"):
            lines = [f"Hola {emp}! Este es tu plan de limpieza para {pd.Timestamp(work_date).strftime('%d/%m/%Y')}:"]
            for _, row in g.iterrows():
                s = _fmt_h(row["start"]); e = _fmt_h(row["end"])
                apt = row["apartment"]; tipo = row.get("tipo","")
                extra = ""
                if apt in cash_map:
                    c = cash_map[apt][0]
                    try:
                        amt = float(c["amount"]) if not pd.isna(c["amount"]) else None
                    except Exception:
                        amt = None
                    if amt is not None:
                        extra = f" | Recoger CASH ${amt:,.2f}" + (f" (checkout {c['hora']})" if str(c['hora']).strip() else "")
                lines.append(f"• {s}-{e}  {apt}  ({tipo}){extra}")
            msg = "\n".join(lines) + "\n\n¡Gracias! 🙌"
            st.text_area(f"Mensaje para {emp}", msg, height=160)
            phone = wa_emp1 if emp == emp1_name else (wa_emp2 if emp == emp2_name else "")
            phone_digits = re.sub(r"\D", "", phone or "")
            if phone_digits:
                wa_link = f"https://wa.me/{phone_digits}?text={_q.quote(msg)}"
                st.markdown(f"[Abrir WhatsApp de **{emp}**]({wa_link})")

    # Ventanas disponibles (info)
    st.subheader("🏠 Ventanas disponibles por apartamento (info)")
    windows_df = build_apartment_windows_v2(normalized, work_date, day_start_t, day_end_t, extra_apts)
    if windows_df.empty:
        st.info("No hay ventanas disponibles para el día seleccionado.")
    else:
        fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(windows_df)))
        apartments = list(windows_df["apartment"].unique()); y_map = {a:i for i,a in enumerate(apartments)}
        day0 = windows_df["start"].min().replace(hour=0, minute=0, second=0, microsecond=0)
        for _, row in windows_df.iterrows():
            y = y_map[row["apartment"]]
            left = (row["start"] - day0).total_seconds()/3600
            width = (row["end"] - row["start"]).total_seconds()/3600
            ax.barh(y, width, left=left)
            ax.text(left+width/2, y, row["kind"], va="center", ha="center", fontsize=9)
        ax.set_yticks(range(len(apartments))); ax.set_yticklabels(apartments)
        ax.set_xlabel("Horas del día"); ax.set_title("Ventanas libres para limpieza")
        st.pyplot(fig)

    # (Opcional) Plan automático dentro de un expander
    with st.expander("⚙️ Plan automático (opcional)"):
        def build_cleaning_jobs_v2(normalized: pd.DataFrame, day, default_duration: int, mop_minutes: int,
                                   day_start_t: time, day_end_t: time, extra_apts=None) -> pd.DataFrame:
            ev = _events_por_apartamento(normalized, day, day_start_t, day_end_t, extra_apts)
            start_of_day, _ = _day_bounds(day, day_start_t, day_end_t)
            jobs = []
            for apt, d in ev.items():
                v = d["ventana"]
                if v is None: continue
                ini, fin, kind = v
                if kind == "turnover":
                    jobs.append({"apartment": apt, "unit_id": "", "guest_name": "",
                                 "start_window": ini, "deadline": fin, "duration_min": int(default_duration)})
                elif kind == "solo checkout":
                    jobs.append({"apartment": apt, "unit_id": "", "guest_name": "",
                                 "start_window": ini, "deadline": pd.NaT, "duration_min": int(default_duration)})
                elif kind == "solo check-in":
                    jobs.append({"apartment": apt, "unit_id": "", "guest_name": "",
                                 "start_window": start_of_day, "deadline": fin, "duration_min": int(mop_minutes)})
                # 'vacio' => sin tarea auto
            return pd.DataFrame(jobs)

        jobs_df = build_cleaning_jobs_v2(normalized, work_date, default_duration, mop_minutes, day_start_t, day_end_t, extra_apts)
        st.dataframe(jobs_df, use_container_width=True)
        e1 = EmployeeTimeline(emp1_name, e1_start, e1_end, e1_l1, e1_l2, work_date)
        e2 = EmployeeTimeline(emp2_name, e2_start, e2_end, e2_l1, e2_l2, work_date)
        plan_df, un_df = greedy_assign(jobs_df, [e1, e2], buffer_min=buffer_minutes, travel_min=travel_minutes, early_priority=early_priority)
        st.dataframe(plan_df, use_container_width=True)
        if not un_df.empty:
            st.warning("No se pudieron asignar algunos trabajos:")
            st.dataframe(un_df, use_container_width=True)
        if not plan_df.empty:
            plot_gantt(plan_df, title="Plan Automático (Gantt)", y_key="employee")

# ====== TAB 2: Analítica mensual ======
with tab2:
    st.subheader("Parámetros de análisis")
    month_anchor = st.date_input("Mes a analizar", value=work_date)
    month_start = pd.Timestamp(month_anchor).replace(day=1).tz_localize(TZ)
    next_month = (month_start + pd.offsets.MonthBegin(1))
    month_end = next_month

    def _booking_revenue(row):
        v = _parse_money(row.get("amount_due", np.nan))
        if np.isnan(v): v = _parse_money(row.get("total_price", np.nan))
        return v

    def _nights_overlap(ci, co, start, end):
        if pd.isna(ci) or pd.isna(co): return 0
        a = max(ci, start); b = min(co, end)
        if b <= a: return 0
        return int((b - a).days)

    normalized["checkin"]  = to_tz(normalized["checkin"])
    normalized["checkout"] = to_tz(normalized["checkout"])

    rows = []
    for _, r in normalized.iterrows():
        apt = r.get("apartment","Apto")
        ci, co = r.get("checkin", pd.NaT), r.get("checkout", pd.NaT)
        nights_total = r.get("nights", None)
        nights_total = int(nights_total) if pd.notna(nights_total) else None
        rev_total = _booking_revenue(r)
        overlap_nights = _nights_overlap(ci, co, month_start, month_end)
        nightly_rate = (rev_total / nights_total) if (rev_total and nights_total and nights_total > 0) else np.nan
        rev_in_month = overlap_nights * nightly_rate if not np.isnan(nightly_rate) else (rev_total if overlap_nights>0 and (nights_total in [None,0]) else np.nan)

        rows.append({
            "apartment": apt,
            "overlap_nights": overlap_nights,
            "rev_in_month": rev_in_month if pd.notna(rev_in_month) else 0.0,
            "checkin_in_month": 1 if (pd.notna(ci) and (month_start <= ci < month_end)) else 0,
            "checkout_in_month": 1 if (pd.notna(co) and (month_start <= co < month_end)) else 0,
            "channel": r.get("channel", None),
            "cash_pickup": bool(r.get("cash_pickup", False) and (pd.notna(co) and (month_start <= co < month_end))),
            "cash_amount": _parse_money(r.get("cash_amount", np.nan)) if (r.get("cash_pickup", False) and pd.notna(r.get("cash_amount", np.nan))) else np.nan,
            "checkin_time_dt": ci,
            "checkout_time_dt": co,
        })
    month_calc = pd.DataFrame(rows)

    if month_calc.empty:
        st.info("No hay reservas para el mes seleccionado.")
    else:
        days_in_month = (month_end - month_start).days
        apt_group = month_calc.groupby("apartment", dropna=False).agg(
            nights=("overlap_nights","sum"),
            revenue=("rev_in_month","sum"),
            checkins=("checkin_in_month","sum"),
            checkouts=("checkout_in_month","sum"),
            cash_pickups=("cash_pickup","sum"),
            cash_amount=("cash_amount", lambda s: np.nansum(s))
        ).reset_index()
        apt_group["ADR"] = (apt_group["revenue"] / apt_group["nights"]).replace([np.inf, -np.inf], np.nan)
        apt_group["occupancy_%"] = (apt_group["nights"] / days_in_month * 100).round(1)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Noches", f"{int(apt_group['nights'].sum())}")
        c2.metric("Ingresos", f"${float(apt_group['revenue'].sum()):,.2f}")
        adr_global = float(apt_group["revenue"].sum())/float(max(1, apt_group["nights"].sum()))
        c3.metric("ADR", f"${adr_global:,.2f}")
        c4.metric("Check-ins", f"{int(apt_group['checkins'].sum())}")
        c5.metric("Check-outs", f"{int(apt_group['checkouts'].sum())}")
        c6.metric("Cash a recoger", f"${float(np.nansum(apt_group['cash_amount'])):,.2f} ({int(apt_group['cash_pickups'].sum())})")

        st.markdown("**Tabla por apartamento**")
        show_cols = ["apartment","nights","revenue","ADR","occupancy_%","checkins","checkouts","cash_pickups","cash_amount"]
        st.dataframe(apt_group[show_cols].sort_values("revenue", ascending=False), use_container_width=True)

        # Ingresos por apartamento (robusto contra tipos/NaN)
        st.markdown("**Ingresos por apartamento**")
        y = apt_group["apartment"].fillna("—").astype(str)
        x = pd.to_numeric(apt_group["revenue"], errors="coerce").fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, max(3, 0.3*len(apt_group))))
        ax.barh(y, x)
        ax.set_xlabel("USD"); ax.set_ylabel("Apartamento"); ax.set_title("Ingresos del mes (prorrateados)")
        st.pyplot(fig)

        # Noches por apartamento (robusto contra tipos/NaN)
        st.markdown("**Noches por apartamento**")
        y2 = apt_group["apartment"].fillna("—").astype(str)
        x2 = pd.to_numeric(apt_group["nights"], errors="coerce").fillna(0)
        fig2, ax2 = plt.subplots(figsize=(11, max(3, 0.3*len(apt_group))))
        ax2.barh(y2, x2)
        ax2.set_xlabel("Noches"); ax2.set_ylabel("Apartamento"); ax2.set_title("Noches del mes")
        st.pyplot(fig2)

        # Mix de canal
        ch_counts = month_calc.copy()
        ch_counts["has_night"] = ch_counts["overlap_nights"] > 0
        ch_counts = ch_counts[ch_counts["has_night"]]
        if not ch_counts.empty and "channel" in ch_counts.columns:
            mix = ch_counts.groupby("channel").size().reset_index(name="reservas")
            st.markdown("**Mix de canal (reservas con noches en el mes)**")
            st.dataframe(mix.sort_values("reservas", ascending=False), use_container_width=True)

        # Horas promedio
        def _avg_time(series_dt):
            s = series_dt.dropna()
            if s.empty: return None
            mins = s.dt.hour*60 + s.dt.minute
            m = int(mins.mean())
            return f"{m//60:02d}:{m%60:02d}"
        avg_ci = _avg_time(month_calc.loc[month_calc["checkin_in_month"]==1, "checkin_time_dt"])
        avg_co = _avg_time(month_calc.loc[month_calc["checkout_in_month"]==1, "checkout_time_dt"])
        st.markdown(f"**Hora promedio** — Check-in: `{avg_ci or '—'}` • Check-out: `{avg_co or '—'}`")
