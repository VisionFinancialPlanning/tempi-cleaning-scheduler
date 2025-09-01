# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import io, unicodedata, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import pytz
from pandas.api.types import is_datetime64_any_dtype as _is_dt

# ==============================
# CONFIG
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

def _find_col(df: pd.DataFrame, aliases) -> str|None:
    """Busca la primera columna coincidente (ignora acentos/espacios/guiones)."""
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
    """Convierte a time() valores tipo string, timestamp o fracciones de día (Excel)."""
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
            if t is not None:
                return t
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
    """Hora embebida en la fecha; si es 00:00:00, la considera 'sin hora'."""
    dates = pd.to_datetime(date_s, errors="coerce")
    out = []
    for d in dates:
        if pd.isna(d):
            out.append(None)
        else:
            hh, mm, ss = d.hour, d.minute, d.second
            out.append(None if (hh, mm, ss) == (0, 0, 0) else time(hh, mm, ss))
    return pd.Series(out, index=dates.index)

def _combine_smart(date_s, explicit_time_s, default_time: time):
    """Prioridad: 1) hora_en_columna 2) hora_embebida_en_fecha 3) default."""
    dates = pd.to_datetime(date_s, errors="coerce")
    t1 = _coerce_time(explicit_time_s) if explicit_time_s is not None else pd.Series([None]*len(dates))
    t2 = _time_from_datecol(date_s)
    out = []
    for d, a, b in zip(dates, t1, t2):
        if pd.isna(d):
            out.append(pd.NaT)
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
    try:    return series.dt.tz_convert(TZ_NAME).dt.strftime(fmt)
    except Exception:
        try: return series.dt.strftime(fmt)
        except Exception: return pd.Series([""]*len(series))

# ==============================
# NORMALIZACIÓN (columnas flexibles)
# ==============================
def parse_bookings_with_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas mínimas: 'Check-In', 'Check-Out'. Horas y extras con alias."""
    if _find_col(df, ["Check-In"]) is None or _find_col(df, ["Check-Out"]) is None:
        raise ValueError("El archivo debe incluir al menos 'Check-In' y 'Check-Out'.")

    # Fechas
    ci_date_col = _find_col(df, ["Check-In"])
    co_date_col = _find_col(df, ["Check-Out"])

    # Horas (alias)
    ci_time_col = _find_col(df, ["Check-In Hora","hora entrada","Hora Entrada","Hora Check In","Check-In Time","Hora Check-In"])
    co_time_col = _find_col(df, ["Check-Out Hora","hora salida","Hora Salida","Hora Check Out","Check-Out Time","Hora Check-Out"])

    ci_dt = _combine_smart(df[ci_date_col], df[ci_time_col] if ci_time_col else None, default_time=time(15,0))
    co_dt = _combine_smart(df[co_date_col], df[co_time_col] if co_time_col else None, default_time=time(12,0))

    # Origen de hora (diagnóstico)
    ci_from_col  = _coerce_time(df[ci_time_col]).notna() if ci_time_col else pd.Series([False]*len(df))
    co_from_col  = _coerce_time(df[co_time_col]).notna() if co_time_col else pd.Series([False]*len(df))
    ci_from_date = _time_from_datecol(df[ci_date_col]).notna()
    co_from_date = _time_from_datecol(df[co_date_col]).notna()

    def _source(from_col, from_date, default_label):
        return np.where(from_col, "hora_col", np.where(from_date, "en_fecha", default_label))

    out = pd.DataFrame()

    # Apartment, huésped
    apt_col   = _find_col(df, ["Property Internal Name"])
    guest_col = _find_col(df, ["Guest First Name"])
    if apt_col:   out["apartment"]  = df[apt_col].astype(str)
    if guest_col: out["guest_name"] = df[guest_col].astype(str)

    # Extras solicitadas (si existen)
    map_optional = {
        "number_guests": ["NUMBER GUESTS","Guests","# Guests","Numero Huespedes","Nº Huespedes"],
        "sofa_or_bed":   ["SOFA OR BED","Sofa Bed","Sofa cama","Sofacama","Sofa as Bed"],
        "reservation_id":["ID","Reservation ID","Booking ID","Código","Codigo"],
        "transporte":    ["TRANSPORTE","Transport","Pickup","Transfer"],
        "extra":         ["EXTRA","Extras","Requests","Solicitudes"],
        "parking":       ["PARKING","Estacionamiento","Parking Spot"],
        "nombre":        ["Nombre","Name","Guest Name"],
        "notes":         ["NOTES","Notas","Observaciones"],
        # Canal / Pago (para CASH)
        "channel":       ["Channel","Source","Booking Source","OTA","Origin","Website","Portal","Booked Via"],
        "payment_method":["Payment Method","Payment","Método de Pago","Forma de pago","Payment Type","Pay Method"]
    }
    for new_col, aliases in map_optional.items():
        c = _find_col(df, aliases)
        if c: out[new_col] = df[c]

    out["checkin"]  = ci_dt
    out["checkout"] = co_dt

    # Formato legible
    out["checkin_day"]   = _fmt_tz(out["checkin"],  "%Y-%m-%d")
    out["checkin_time"]  = _fmt_tz(out["checkin"],  "%H:%M")
    out["checkout_day"]  = _fmt_tz(out["checkout"], "%Y-%m-%d")
    out["checkout_time"] = _fmt_tz(out["checkout"], "%H:%M")

    # Noches
    try:
        nights = out["checkout"].dt.tz_convert(TZ_NAME).dt.date - out["checkin"].dt.tz_convert(TZ_NAME).dt.date
    except Exception:
        nights = pd.to_datetime(out["checkout"], errors="coerce").dt.date - pd.to_datetime(out["checkin"], errors="coerce").dt.date
    out["nights"] = [d.days if pd.notna(d) else None for d in nights]

    out["checkin_time_source"]  = _source(ci_from_col, ci_from_date, "default 15:00")
    out["checkout_time_source"] = _source(co_from_col, co_from_date, "default 12:00")

    # CASH pickup detection
    def _needs_cash(row):
        reason = []
        ch = str(row.get("channel","")).lower()
        pm = str(row.get("payment_method","")).lower()
        nt = str(row.get("notes","")).lower()
        if "booking" in ch: reason.append("canal=Booking")
        if any(w in pm for w in ["cash","efectivo"]): reason.append("pago=cash")
        if any(w in nt for w in ["cash","efectivo"]): reason.append("nota menciona cash")
        return (len(reason) > 0, ", ".join(reason))
    cash, why = zip(*[ _needs_cash(r) for _, r in out.iterrows() ]) if len(out) else ([],[])
    out["cash_pickup"]  = list(cash)
    out["cash_reason"]  = list(why)

    # Orden amigable
    preferred = [
        "apartment","guest_name","number_guests","sofa_or_bed","reservation_id",
        "transporte","extra","parking","nombre","notes",
        "channel","payment_method","cash_pickup","cash_reason",
        "checkin_day","checkin_time","checkin_time_source",
        "checkout_day","checkout_time","checkout_time_source",
        "nights","checkin","checkout",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

# ==============================
# SCHEDULER & DISPONIBILIDAD
# ==============================
def to_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try: s = s.dt.tz_convert(TZ_NAME)
    except Exception:
        try: s = s.dt.tz_localize(TZ_NAME)
        except Exception: pass
    return s

def filter_day(df: pd.DataFrame, day) -> pd.DataFrame:
    start = pd.Timestamp(day, tz=TZ); end = start + pd.Timedelta(days=1)
    if "checkout" in df.columns: df["checkout"] = to_tz(df["checkout"])
    if "checkin"  in df.columns: df["checkin"]  = to_tz(df["checkin"])
    mask = df["checkout"].between(start, end, inclusive="left") | df["checkin"].between(start, end, inclusive="left")
    return df[mask].copy()

def infer_duration(row, default_duration, cfg):
    if "clean_duration_minutes" in row and pd.notna(row["clean_duration_minutes"]):
        return int(row["clean_duration_minutes"])
    if cfg is not None and not cfg.empty and "apartment" in cfg.columns:
        apt_name = str(row.get("apartment","")).strip().lower()
        hit = cfg[cfg["apartment"].astype(str).str.lower() == apt_name]
        if not hit.empty: return int(hit.iloc[0].get("base_minutes", default_duration))
    return default_duration

def build_cleaning_jobs(day_df: pd.DataFrame, day, default_duration: int, cfg: pd.DataFrame, mop_minutes: int,
                        day_start_t: time, day_end_t: time) -> pd.DataFrame:
    jobs = []
    the_day = pd.Timestamp(day).date()
    start_of_day = TZ.localize(datetime.combine(the_day, day_start_t))
    for _, row in day_df.iterrows():
        apt = row.get("apartment", "Apto")
        unit = row.get("unit_id", "")
        guest = row.get("guest_name", "")
        checkout = row.get("checkout", pd.NaT)
        checkin  = row.get("checkin",  pd.NaT)
        is_co = pd.notna(checkout) and checkout.date() == the_day
        is_ci = pd.notna(checkin)  and checkin.date()  == the_day
        if is_co and is_ci:
            duration = infer_duration(row, default_duration, cfg); start_window = checkout; deadline = checkin
        elif is_co:
            duration = infer_duration(row, default_duration, cfg); start_window = checkout; deadline = pd.NaT
        elif is_ci:
            duration = int(mop_minutes); start_window = start_of_day; deadline = checkin
        else:
            continue
        jobs.append({"apartment": apt, "unit_id": unit, "guest_name": guest,
                     "start_window": start_window, "deadline": deadline, "duration_min": int(duration)})
    return pd.DataFrame(jobs)

def build_apartment_windows(day_df: pd.DataFrame, day, day_start_t: time, day_end_t: time) -> pd.DataFrame:
    the_day = pd.Timestamp(day).date()
    start_of_day = TZ.localize(datetime.combine(the_day, day_start_t))
    end_of_day   = TZ.localize(datetime.combine(the_day, day_end_t))
    rows = []
    for _, r in day_df.iterrows():
        apt = r.get("apartment", "Apto"); ci = r.get("checkin", pd.NaT); co = r.get("checkout", pd.NaT)
        is_ci = pd.notna(ci) and ci.date() == the_day; is_co = pd.notna(co) and co.date() == the_day
        if is_co and is_ci: rows.append({"apartment": apt, "start": co, "end": ci, "kind": "turnover"})
        elif is_co:        rows.append({"apartment": apt, "start": co, "end": end_of_day, "kind": "solo checkout"})
        elif is_ci:        rows.append({"apartment": apt, "start": start_of_day, "end": ci, "kind": "solo check-in (mopa)"})
    win = pd.DataFrame(rows)
    if not win.empty: win = win[win["end"] > win["start"]]
    return win

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

def plot_gantt(df: pd.DataFrame, title, label_key):
    if df is None or df.empty:
        st.info("No hay datos para mostrar."); return
    fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(df)))
    labels = list(df[label_key].unique())
    y_map = {e:i for i,e in enumerate(labels)}
    day0 = df["start"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for _, row in df.iterrows():
        y = y_map[row[label_key]]
        start = row["start"].to_pydatetime(); end = row["end"].to_pydatetime()
        left = (start - day0).total_seconds()/3600; width = (end-start).total_seconds()/3600
        ax.barh(y, width, left=left)
        label = row.get("apartment", row.get("employee",""))
        ax.text(left+width/2, y, f"{label}", va="center", ha="center", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Horas del día"); ax.set_title(title); st.pyplot(fig)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Configuración del Día")
    work_date = st.date_input("Día a planificar", value=pd.Timestamp.now(TZ).date())
    prep_date = st.date_input("Preparación (día a revisar)", value=(pd.Timestamp.now(TZ)+pd.Timedelta(days=1)).date(),
                              help="Por defecto mañana, para trabajar un día antes.")

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

# ==============================
# CARGA → NORMALIZA
# ==============================
uploaded = st.file_uploader(
    "Sube tu Excel (.xlsx). Requeridas: 'Check-In', 'Check-Out'. Alias de hora aceptados: 'Check-In Hora'/'hora entrada' y 'Check-Out Hora'/'hora salida'.",
    type=["xlsx"]
)
if uploaded is not None:
    df_raw = pd.read_excel(uploaded)
else:
    try:
        st.info("Usando datos de ejemplo: sample_bookings.xlsx")
        df_raw = pd.read_excel("sample_bookings.xlsx")
    except Exc
