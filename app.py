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
from datetime import datetime, time
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
    """Busca la primera columna que coincida con la lista de alias (ignorando acentos, espacios, guiones)."""
    norm_map = {_norm(c): c for c in df.columns}
    # match exact normalizado
    for a in aliases:
        an = _norm(a)
        if an in norm_map:
            return norm_map[an]
    # match por contener
    for a in aliases:
        an = _norm(a)
        for key, real in norm_map.items():
            if an in key:
                return real
    return None

def _coerce_time(series):
    """Convierte distintos formatos de hora a time() (o None). Soporta strings, Timestamp y fracciones de Excel."""
    if series is None:
        return pd.Series(dtype="object")
    def parse_one(v):
        if pd.isna(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.time()
        if isinstance(v, (int, float)):
            try:
                frac = float(v) % 1.0        # fracción de día
                secs = int(round(frac * 24 * 3600))
                hh = secs // 3600
                mm = (secs % 3600) // 60
                ss = secs % 60
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
    """Extrae la hora embebida en la columna de fecha (si la hay). Si viene 00:00:00 la considera 'sin hora'."""
    dates = pd.to_datetime(date_s, errors="coerce")
    out = []
    for d in dates:
        if pd.isna(d):
            out.append(None)
        else:
            hh, mm, ss = d.hour, d.minute, d.second
            if hh == 0 and mm == 0 and ss == 0:
                out.append(None)
            else:
                out.append(time(hh, mm, ss))
    return pd.Series(out, index=dates.index)

def _combine_smart(date_s, explicit_time_s, default_time: time):
    """Prioridad: 1) hora de columna de hora 2) hora embebida en la fecha 3) default."""
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
    if ser.empty or not _is_dt(ser):
        return ser
    try:
        ser = ser.dt.tz_localize(TZ_NAME, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        return ser
    return ser

def _fmt_tz(series, fmt):
    try:
        return series.dt.tz_convert(TZ_NAME).dt.strftime(fmt)
    except Exception:
        try:
            return series.dt.strftime(fmt)
        except Exception:
            return pd.Series([""] * len(series))

# ==============================
# NORMALIZACIÓN (columnas flexibles)
# ==============================
def parse_bookings_with_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de fecha obligatorias: 'Check-In', 'Check-Out'. Horas: múltiples alias."""
    if _find_col(df, ["Check-In"]) is None or _find_col(df, ["Check-Out"]) is None:
        raise ValueError("El archivo debe incluir al menos 'Check-In' y 'Check-Out'.")

    ci_date_col = _find_col(df, ["Check-In"])
    co_date_col = _find_col(df, ["Check-Out"])

    ci_time_col = _find_col(df, ["Check-In Hora","hora entrada","Hora Entrada","Hora Check In","Check-In Time","Hora Check-In"])
    co_time_col = _find_col(df, ["Check-Out Hora","hora salida","Hora Salida","Hora Check Out","Check-Out Time","Hora Check-Out"])

    # Combinar con prioridad hora_col -> hora_en_fecha -> default
    ci_dt = _combine_smart(df[ci_date_col], df[ci_time_col] if ci_time_col else None, default_time=time(15,0))
    co_dt = _combine_smart(df[co_date_col], df[co_time_col] if co_time_col else None, default_time=time(12,0))

    # Origen de la hora (diagnóstico)
    ci_from_col = _coerce_time(df[ci_time_col]).notna() if ci_time_col else pd.Series([False]*len(df))
    co_from_col = _coerce_time(df[co_time_col]).notna() if co_time_col else pd.Series([False]*len(df))
    ci_from_date = _time_from_datecol(df[ci_date_col]).notna()
    co_from_date = _time_from_datecol(df[co_date_col]).notna()

    def _source(from_col, from_date, default_label):
        src = np.where(from_col, "hora_col", np.where(from_date, "en_fecha", default_label))
        return src

    out = pd.DataFrame()

    # apartment & guest
    apt_col = _find_col(df, ["Property Internal Name"])
    guest_col = _find_col(df, ["Guest First Name"])
    if apt_col: out["apartment"] = df[apt_col].astype(str)
    if guest_col: out["guest_name"] = df[guest_col].astype(str)

    out["checkin"] = ci_dt
    out["checkout"] = co_dt

    out["checkin_day"] = _fmt_tz(out["checkin"], "%Y-%m-%d")
    out["checkin_time"] = _fmt_tz(out["checkin"], "%H:%M")
    out["checkout_day"] = _fmt_tz(out["checkout"], "%Y-%m-%d")
    out["checkout_time"] = _fmt_tz(out["checkout"], "%H:%M")

    # Noches
    try:
        nights = out["checkout"].dt.tz_convert(TZ_NAME).dt.date - out["checkin"].dt.tz_convert(TZ_NAME).dt.date
    except Exception:
        nights = pd.to_datetime(out["checkout"], errors="coerce").dt.date - pd.to_datetime(out["checkin"], errors="coerce").dt.date
    out["nights"] = [d.days if pd.notna(d) else None for d in nights]

    out["checkin_time_source"] = _source(ci_from_col, ci_from_date, "default 15:00")
    out["checkout_time_source"] = _source(co_from_col, co_from_date, "default 12:00")

    preferred = [
        "apartment","guest_name",
        "checkin_day","checkin_time","checkin_time_source",
        "checkout_day","checkout_time","checkout_time_source",
        "nights","checkin","checkout",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

# ==============================
# SCHEDULER & DISPONIBILIDAD
# ==============================
@st.cache_data
def load_apartment_config():
    try: return pd.read_csv("apartment_config.csv")
    except Exception: return pd.DataFrame()

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
    if "checkin" in df.columns: df["checkin"]  = to_tz(df["checkin"])
    mask = df["checkout"].between(start, end, inclusive="left") | df["checkin"].between(start, end, inclusive="left")
    return df[mask].copy()

def infer_duration(row, default_duration, cfg):
    if "clean_duration_minutes" in row and pd.notna(row["clean_duration_minutes"]):
        return int(row["clean_duration_minutes"])
    if cfg is not None and not cfg.empty and "apartment" in cfg.columns:
        apt_name = str(row.get("apartment","")).strip().lower()
        hit = cfg[cfg["apartment"].astype(str).str.lower() == apt_name]
        if not hit.empty:
            return int(hit.iloc[0].get("base_minutes", default_duration))
    return default_duration

def build_cleaning_jobs(day_df: pd.DataFrame, day, default_duration: int, cfg: pd.DataFrame, mop_minutes: int,
                        day_start_t: time, day_end_t: time) -> pd.DataFrame:
    jobs = []
    the_day = pd.Timestamp(day).date()
    start_of_day = TZ.localize(datetime.combine(the_day, day_start_t))
    for _, row in day_df.iterrows():
        apt = row.get("apartment", "Apto"); unit = row.get("unit_id", ""); guest = row.get("guest_name", "")
        checkout = row.get("checkout", pd.NaT); checkin = row.get("checkin", pd.NaT)
        is_co = pd.notna(checkout) and checkout.date() == the_day
        is_ci = pd.notna(checkin) and checkin.date() == the_day
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
    end_of_day = TZ.localize(datetime.combine(the_day, day_end_t))
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

def plot_gantt(plan_df: pd.DataFrame, title="Plan de Limpiezas (Gantt)", y_label="Empleadas"):
    if plan_df is None or plan_df.empty:
        st.info("No hay datos para mostrar."); return
    fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(plan_df)))
    labels = list(plan_df["employee"].unique()) if "employee" in plan_df.columns else list(plan_df["apartment"].unique())
    y_map = {e:i for i,e in enumerate(labels)}
    day0 = plan_df["start"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for _, row in plan_df.iterrows():
        key = "employee" if "employee" in row else "apartment"
        y = y_map[row[key]]
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
    buffer_minutes = st.number_input("Buffer entre limpiezas (min)", value=10, step=5)
    travel_minutes = st.number_input("Traslado entre apartamentos (min)", value=0, step=5)
    default_duration = st.number_input("Duración limpieza completa (min)", value=90, step=5)
    mop_minutes = st.number_input("Duración mopa (solo check-in) (min)", value=20, step=5)
    early_priority = st.checkbox("Priorizar salidas tempranas (deadline primero)", value=True)

# ==============================
# CARGA → NORMALIZA → DISPONIBILIDAD → PLAN
# ==============================
uploaded = st.file_uploader(
    "Sube tu Excel (.xlsx). Requeridas: 'Check-In', 'Check-Out'. Opcionales (alias aceptados): 'Check-In Hora'/'hora entrada' y 'Check-Out Hora'/'hora salida'.",
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

st.subheader("📄 Reservas – Original (preview)")
st.dataframe(df_raw.head(10), use_container_width=True)

# Normalizar
try:
    normalized = parse_bookings_with_fixed_columns(df_raw)
except Exception as e:
    st.error(f"Error al normalizar reservas: {e}"); st.stop()

# Aviso de fuentes de hora
if "checkin_time_source" in normalized.columns and "checkout_time_source" in normalized.columns:
    pct_ci = (normalized["checkin_time_source"] == "hora_col").mean() * 100
    pct_ci_date = (normalized["checkin_time_source"] == "en_fecha").mean() * 100
    pct_ci_def = (normalized["checkin_time_source"] == "default 15:00").mean() * 100
    pct_co = (normalized["checkout_time_source"] == "hora_col").mean() * 100
    pct_co_date = (normalized["checkout_time_source"] == "en_fecha").mean() * 100
    pct_co_def = (normalized["checkout_time_source"] == "default 12:00").mean() * 100
    st.caption(f"Origen horas → Check-In: col {pct_ci:.0f}%, en_fecha {pct_ci_date:.0f}%, default {pct_ci_def:.0f}% • "
               f"Check-Out: col {pct_co:.0f}%, en_fecha {pct_co_date:.0f}%, default {pct_co_def:.0f}%")

st.success("✅ Reservas normalizadas")
st.dataframe(normalized, use_container_width=True)

# Descargas normalizado
csv_bytes = normalized.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar Normalizado (CSV)", data=csv_bytes, file_name="checkins_checkouts.csv", mime="text/csv")
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    normalized.to_excel(writer, sheet_name="checkins_checkouts", index=False)
st.download_button("⬇️ Descargar Normalizado (Excel)", data=excel_buf.getvalue(),
                   file_name="checkins_checkouts.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Ventanas por apartamento (Gantt)
st.subheader("🏠 Ventanas disponibles por apartamento (Gantt)")
day_df = filter_day(normalized, work_date)
windows_df = build_apartment_windows(day_df, work_date, day_start_t, day_end_t)
if windows_df.empty:
    st.info("No hay ventanas disponibles para el día seleccionado.")
else:
    fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(windows_df)))
    apartments = list(windows_df["apartment"].unique()); y_map = {a:i for i,a in enumerate(apartments)}
    day0 = windows_df["start"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for _, row in windows_df.iterrows():
        y = y_map[row["apartment"]]
        start = row["start"].to_pydatetime(); end = row["end"].to_pydatetime()
        left = (start - day0).total_seconds()/3600; width = (end-start).total_seconds()/3600
        ax.barh(y, width, left=left)
        ax.text(left+width/2, y, row["kind"], va="center", ha="center", fontsize=9)
    ax.set_yticks(range(len(apartments))); ax.set_yticklabels(apartments)
    ax.set_xlabel("Horas del día"); ax.set_title("Ventanas libres para limpieza"); st.pyplot(fig)

# Planificación / Asignación
st.header("🧭 Horario de Limpieza (asignación)")
jobs_df = build_cleaning_jobs(day_df, work_date, default_duration, pd.DataFrame(), mop_minutes, day_start_t, day_end_t)
st.subheader("🧱 Trabajos a programar"); st.dataframe(jobs_df, use_container_width=True)

e1 = EmployeeTimeline(emp1_name, e1_start, e1_end, e1_l1, e1_l2, work_date)
e2 = EmployeeTimeline(emp2_name, e2_start, e2_end, e2_l1, e2_l2, work_date)
plan_df, un_df = greedy_assign(jobs_df, [e1, e2], buffer_min=buffer_minutes, travel_min=travel_minutes, early_priority=early_priority)

st.subheader("🗓️ Plan asignado"); st.dataframe(plan_df, use_container_width=True)
if not un_df.empty:
    st.warning("No se pudieron asignar algunos trabajos:"); st.dataframe(un_df, use_container_width=True)

st.subheader("📊 Visualización del plan (por empleada)")
plot_gantt(plan_df)

st.subheader("📲 Resumen para WhatsApp")
if plan_df.empty:
    st.code("(Sin asignaciones)")
else:
    lines = ["*Plan de Limpiezas* 🧼"]
    for emp in plan_df["employee"].unique():
        lines.append(f"\n*{emp}*")
        sub = plan_df[plan_df["employee"] == emp]
        for _, r in sub.sort_values("start").iterrows():
            lines.append(f"• {r['apartment']} — {r['start'].strftime('%H:%M')}–{r['end'].strftime('%H:%M')} ({int(r['duration_min'])}m)")
    st.code("\n".join(lines))

plan_csv = plan_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar plan (CSV)", data=plan_csv, file_name=f"plan_{work_date}.csv", mime="text/csv")
