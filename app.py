# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import re, unicodedata, hashlib
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
# 🔒 PASSWORD GATE
# ==============================
def _get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

SECRET_PLAIN = os.environ.get("TEMPI_PASSWORD") or _get_secret("APP_PASSWORD", "")
SECRET_SHA   = os.environ.get("TEMPI_PASSWORD_SHA256") or _get_secret("APP_PASSWORD_SHA256", "")

def _check_plain(pwd: str) -> bool:
    return bool(SECRET_PLAIN) and (pwd == SECRET_PLAIN)

def _check_hash(pwd: str) -> bool:
    if not SECRET_SHA:
        return False
    try:
        return hashlib.sha256(pwd.encode("utf-8")).hexdigest() == SECRET_SHA.lower()
    except Exception:
        return False

def check_password():
    if not SECRET_PLAIN and not SECRET_SHA:
        with st.sidebar:
            st.info("🔓 Esta app no tiene contraseña configurada.\n\nConfigura `APP_PASSWORD` o `APP_PASSWORD_SHA256` en *Secrets* para activarla.")
        return True

    if st.session_state.get("pass_ok"):
        with st.sidebar:
            if st.button("Cerrar sesión"):
                st.session_state.pass_ok = False
                st.experimental_rerun()
        return True

    with st.sidebar:
        st.subheader("🔒 Acceso")
        pwd = st.text_input("Contraseña", type="password")
        c1, c2 = st.columns(2)
        with c1: login = st.button("Entrar")
        with c2: reset = st.button("Limpiar")
        if reset:
            st.experimental_rerun()
        if login:
            if _check_plain(pwd) or _check_hash(pwd):
                st.session_state.pass_ok = True
                st.experimental_rerun()
            else:
                st.error("Contraseña incorrecta.")
                st.stop()
        else:
            st.stop()

check_password()

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
    if x is None or (isinstance(x, float) and pd.isna(x)): return False
    s = str(x).strip().lower().replace("í", "i")
    return s in {"si", "yes"}

def _day_bounds(day, start_t: time, end_t: time):
    d = pd.Timestamp(day).date()
    return TZ.localize(datetime.combine(d, start_t)), TZ.localize(datetime.combine(d, end_t))

def _canon_apt(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}: return np.nan
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
        "total_price":   ["Total Price","Total","Precio Total","Importe","Booking Amount","Payment Amount","Grand Total","Monto","Monto Total"],
        "phone":         ["Phone","Guest Phone","Phone Number","Mobile","Celular","Teléfono","Telefono","Contact","Contact Phone","Guest Mobile"],
    }
    for new_col, aliases in map_optional.items():
        c = _find_col(df, aliases)
        if c: out[new_col] = df[c]

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
        "transporte","extra","parking","nombre","notes","phone",
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
        if v is None:
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
    out["__bali__"] = out["apartment"].astype(str).eq("BALI")
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
    wa_concierge = st.text_input("WhatsApp Conserje (solo dígitos)", value="")
    wa_driver = st.text_input("WhatsApp Motorista (solo dígitos)", value="")

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
tab1, tab2, tab3 = st.tabs(["🧭 Operativa del día", "📈 Analítica mensual", "🚐 Transportes"])

# ====== TAB 1: Operativa del día ======
with tab1:
    today_str = pd.Timestamp(work_date).strftime("%Y-%m-%d")

    st.subheader("📦 Preparación del día (mismo día)")
    prep_df = normalized[(normalized["checkin_day"] == today_str) | (normalized["checkout_day"] == today_str)].copy]()]()_
