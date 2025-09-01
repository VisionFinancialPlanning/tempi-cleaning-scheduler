import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import pytz

TZ = pytz.timezone("America/Panama")
st.set_page_config(page_title="Tempi Cleaning Scheduler", layout="wide")

def auth_gate():
    try:
        pwd = st.sidebar.text_input("Contraseña", type="password")
        if pwd != st.secrets.get("APP_PASSWORD", "tempi"):
            st.sidebar.info("Ingresa la contraseña para continuar")
            st.stop()
    except Exception:
        # Si no hay secrets configurados, usa valor por defecto
        pwd = st.sidebar.text_input("Contraseña", type="password")
        if pwd != "tempi":
            st.sidebar.info("Ingresa la contraseña para continuar")
            st.stop()
auth_gate()

st.title("🧼 Tempi – Planificador Diario de Limpiezas")

with st.sidebar:
    st.header("⚙️ Configuración")
    work_date = st.date_input("Día a planificar", value=pd.Timestamp.now(TZ).date())

    st.subheader("Empleadas (2)")
    emp1_name = st.text_input("Empleado 1", value="Mayerlin")
    emp2_name = st.text_input("Empleado 2", value="Evelyn")

    def shift_block(label, default_start=time(8, 0), default_end=time(17, 0)):
        c1, c2 = st.columns(2)
        with c1:
            s = st.time_input(f"{label} – inicio", value=default_start)
        with c2:
            e = st.time_input(f"{label} – fin", value=default_end)
        return s, e

    st.caption("Turnos y almuerzo – se aplican a cada empleada")
    e1_start, e1_end = shift_block("Empleado 1", time(8,0), time(17,0))
    e2_start, e2_end = shift_block("Empleado 2", time(8,0), time(17,0))

    st.subheader("Almuerzo")
    e1_l1, e1_l2 = shift_block("Empleado 1 – Almuerzo", time(12,0), time(13,0))
    e2_l1, e2_l2 = shift_block("Empleado 2 – Almuerzo", time(12,30), time(13,30))

    st.subheader("Parámetros del plan")
    buffer_minutes = st.number_input("Buffer entre limpiezas (min)", value=10, step=5)
    travel_minutes = st.number_input("Traslado entre apartamentos (min)", value=0, step=5)
    default_duration = st.number_input("Duración por defecto (min)", value=90, step=5)
    early_priority = st.checkbox("Priorizar salidas tempranas (deadline primero)", value=True)
    use_apt_cfg = st.checkbox("Usar matriz de tiempos por apartamento (apartment_config.csv)", value=True)

uploaded = st.file_uploader("Sube tu Excel de reservas • O usa `sample_bookings.xlsx`", type=["xlsx", "xls", "csv"])

# === NUEVO parse_bookings: admite columnas exactas 'Check-In', 'Check-In Hora', 'Check-Out', 'Check-Out Hora'.
# Si faltan horas, aplica Check-In 15:00 y Check-Out 12:00.

def parse_bookings(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names dict (lowercase for detection)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    # mapping for legacy columns
    ren = {}
    for k in ["apartment", "unit_id", "guest_name", "checkout", "checkin", "clean_duration_minutes"]:
        for orig_lower, orig in cols_lower.items():
            if orig_lower == k:
                ren[orig] = k
    df = df.rename(columns=ren)

    # If new fixed columns exist: "Check-In","Check-In Hora","Check-Out","Check-Out Hora"
    # (case-insensitive match)
    def find_col(name):
        for orig_lower, orig in cols_lower.items():
            if orig_lower == name.lower():
                return orig
        return None

    ci_date_col = find_col("Check-In")
    ci_time_col = find_col("Check-In Hora")
    co_date_col = find_col("Check-Out")
    co_time_col = find_col("Check-Out Hora")

    def _coerce_time(series):
        def parse_one(v):
            if pd.isna(v):
                return None
            if isinstance(v, (pd.Timestamp, datetime)):
                return v.time()
            try:
                return pd.to_datetime(str(v), errors="coerce").time()
            except Exception:
                return None
        return series.apply(parse_one)

    def _combine(date_s, time_s, default_time: time):
        dates = pd.to_datetime(date_s, errors="coerce")
        times = _coerce_time(time_s) if time_s is not None else None
        out = []
        for d, t in zip(dates, times if times is not None else [None]*len(dates)):
            if pd.isna(d):
                out.append(pd.NaT)
            else:
                if t is None:
                    t = default_time
                out.append(pd.Timestamp.combine(pd.Timestamp(d).date(), t))
        return pd.to_datetime(out, errors="coerce").dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT").dt.tz_convert(TZ)

    if ci_date_col and co_date_col:
        # Build checkin/checkout even if legacy columns are missing
        ci_dt = _combine(df[ci_date_col], df[ci_time_col] if (ci_time_col and ci_time_col in df.columns) else None, default_time=time(15,0))
        co_dt = _combine(df[co_date_col], df[co_time_col] if (co_time_col and co_time_col in df.columns) else None, default_time=time(12,0))
        df["checkin"] = ci_dt
        df["checkout"] = co_dt

    # Coerce legacy date columns if present
    for col in ["checkout", "checkin"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "clean_duration_minutes" not in df.columns:
        df["clean_duration_minutes"] = np.nan
    return df

@st.cache_data
def load_apartment_config():
    try:
        return pd.read_csv("apartment_config.csv")
    except Exception:
        return pd.DataFrame()

def filter_day(df: pd.DataFrame, day: datetime.date) -> pd.DataFrame:
    start = pd.Timestamp(day, tz=TZ)
    end = start + pd.Timedelta(days=1)
    def to_panama(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce")
        try:
            s = s.dt.tz_convert(TZ)
        except Exception:
            try:
                s = s.dt.tz_localize(TZ)
            except Exception:
                s = s
        return s
    if "checkout" in df.columns:
        df["checkout"] = to_panama(df["checkout"])
    if "checkin" in df.columns:
        df["checkin"] = to_panama(df["checkin"])
    mask = (
        (df["checkout"].between(start, end, inclusive="left")) |
        (df["checkin"].between(start, end, inclusive="left"))
    )
    return df[mask].copy()

def infer_duration(row, default_duration, cfg):
    if pd.notna(row.get("clean_duration_minutes")):
        return int(row["clean_duration_minutes"])
    if cfg is not None and not cfg.empty:
        hit = cfg[cfg["apartment"].str.lower() == str(row.get("apartment", "")).lower()]
        if not hit.empty:
            return int(hit.iloc[0].get("base_minutes", default_duration))
    return default_duration

def build_cleaning_jobs(day_df: pd.DataFrame, day: datetime.date, default_duration: int, cfg: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for _, row in day_df.iterrows():
        apt = row.get("apartment", "Apto")
        checkout = row.get("checkout", pd.NaT)
        checkin = row.get("checkin", pd.NaT)
        duration = infer_duration(row, default_duration, cfg)
        start_window = checkout if pd.notna(checkout) else pd.Timestamp.combine(day, time(8, 0), tzinfo=TZ)
        deadline = checkin if (pd.notna(checkin) and checkin.date() == day) else pd.NaT
        jobs.append({
            "apartment": apt,
            "unit_id": row.get("unit_id", ""),
            "guest_name": row.get("guest_name", ""),
            "start_window": start_window,
            "deadline": deadline,
            "duration_min": int(duration)
        })
    return pd.DataFrame(jobs)

class EmployeeTimeline:
    def __init__(self, name, start_t: time, end_t: time, lunch_start: time, lunch_end: time, day: datetime.date):
        self.name = name
        self.day = day
        self.start_dt = TZ.localize(datetime.combine(day, start_t))
        self.end_dt = TZ.localize(datetime.combine(day, end_t))
        self.lunch_start = TZ.localize(datetime.combine(day, lunch_start))
        self.lunch_end = TZ.localize(datetime.combine(day, lunch_end))
        self.cursor = self.start_dt
        self.slots = []
    def _overlaps_lunch(self, start, end):
        return not (end <= self.lunch_start or start >= self.lunch_end)
    def trial(self, job, buffer_min=10, travel_min=0):
        est = max(self.cursor, job["start_window"]) + pd.Timedelta(minutes=travel_min)
        end = est + pd.Timedelta(minutes=job["duration_min"])
        if self._overlaps_lunch(est, end):
            return None
        deadline = job["deadline"] if pd.notna(job["deadline"]) else self.end_dt
        if end > deadline or end > self.end_dt:
            return None
        return {
            "employee": self.name,
            "apartment": job["apartment"],
            "unit_id": job["unit_id"],
            "guest_name": job.get("guest_name", ""),
            "start": est,
            "end": end,
            "duration_min": job["duration_min"]
        }
    def schedule(self, job, buffer_min=10, travel_min=0):
        trial = self.trial(job, buffer_min=buffer_min, travel_min=travel_min)
        if trial is None:
            return None
        self.slots.append(trial)
        self.cursor = trial["end"] + pd.Timedelta(minutes=buffer_min + travel_min)
        return trial

def greedy_assign(jobs_df, employees, buffer_min=10, travel_min=0, early_priority=True):
    if jobs_df is None or jobs_df.empty:
        return pd.DataFrame(columns=["employee","apartment","unit_id","guest_name","start","end","duration_min"]), pd.DataFrame()
    sort_keys = jobs_df.copy()
    min_date = sort_keys["start_window"].dt.date.min()
    far = TZ.localize(datetime.combine(min_date, time(23, 59))) + pd.Timedelta(days=1)
    sort_keys["deadline_filled"] = sort_keys["deadline"].fillna(far)
    jobs_sorted = sort_keys.sort_values(
        ["deadline_filled", "start_window", "duration_min"] if early_priority
        else ["start_window", "deadline_filled", "duration_min"]
    ).drop(columns=["deadline_filled"])
    assignments, unassigned = [], []
    for _, job in jobs_sorted.iterrows():
        candidates = []
        for emp in employees:
            t = emp.trial(job, buffer_min=buffer_min, travel_min=travel_min)
            if t is not None:
                candidates.append((emp, t))
        if not candidates:
            unassigned.append(job.to_dict())
            continue
        # escoger el fin más temprano
        emp_sel, slot = sorted(candidates, key=lambda x: x[1]["end"])[0]
        emp_sel.schedule(job, buffer_min=buffer_min, travel_min=travel_min)
        assignments.append(slot)
    return pd.DataFrame(assignments), pd.DataFrame(unassigned)

def plot_gantt(plan_df: pd.DataFrame):
    if plan_df is None or plan_df.empty:
        st.info("No hay asignaciones para mostrar.")
        return
    fig, ax = plt.subplots(figsize=(10, 3+0.3*len(plan_df)))
    employees = list(plan_df["employee"].unique())
    y_map = {e:i for i,e in enumerate(employees)}
    for _, row in plan_df.iterrows():
        y = y_map[row["employee"]]
        start = row["start"].to_pydatetime()
        end = row["end"].to_pydatetime()
        ax.barh(y, (end-start).total_seconds()/3600, left=(start - start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()/3600)
        ax.text(((start.hour*60+start.minute)+(end.hour*60+end.minute))/120, y, f"{row['apartment']}", va='center')
    ax.set_yticks(range(len(employees)))
    ax.set_yticklabels(employees)
    ax.set_xlabel("Horas del día")
    ax.set_title("Plan de Limpiezas (Gantt)")
    st.pyplot(fig)

def whatsapp_summary(plan_df: pd.DataFrame) -> str:
    if plan_df is None or plan_df.empty:
        return "(Sin asignaciones)"
    lines = ["*Plan de Limpiezas* 🧼"]
    for emp in plan_df["employee"].unique():
        lines.append(f"\n*{emp}*")
        sub = plan_df[plan_df["employee"]==emp]
        for _, r in sub.sort_values("start").iterrows():
            lines.append(f"• {r['apartment']} — {r['start'].strftime('%H:%M')}–{r['end'].strftime('%H:%M')} ({int(r['duration_min'])}m)")
    return "\n".join(lines)

# === FLUJO PRINCIPAL ===
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
else:
    st.info("Usando datos de ejemplo `sample_bookings.xlsx` (en el repo)")
    df_raw = pd.read_excel("sample_bookings.xlsx")

st.subheader("📄 Reservas – Original (preview)")
st.dataframe(df_raw.head(10), use_container_width=True)

# Normaliza según tus columnas (Check-In / Check-In Hora / Check-Out / Check-Out Hora)
df = parse_bookings(df_raw)

# Filtrar por día
day_df = filter_day(df, work_date)
st.subheader("📄 Reservas del día (normalizadas)")
st.dataframe(day_df, use_container_width=True)

apt_cfg = load_apartment_config() if use_apt_cfg else pd.DataFrame()
jobs_df = build_cleaning_jobs(day_df, work_date, default_duration, apt_cfg)

st.subheader("🧱 Trabajos a programar")
st.dataframe(jobs_df, use_container_width=True)

# Timelines
e1 = EmployeeTimeline(emp1_name, e1_start, e1_end, e1_l1, e1_l2, work_date)
e2 = EmployeeTimeline(emp2_name, e2_start, e2_end, e2_l1, e2_l2, work_date)
plan_df, un_df = greedy_assign(
    jobs_df, [e1, e2],
    buffer_min=buffer_minutes,
    travel_min=travel_minutes,
    early_priority=early_priority
)

st.subheader("🗓️ Plan asignado")
st.dataframe(plan_df, use_container_width=True)

if not un_df.empty:
    st.warning("No se pudieron asignar algunos trabajos:")
    st.dataframe(un_df, use_container_width=True)

st.subheader("📊 Visualización")
plot_gantt(plan_df)

st.subheader("📲 Resumen para WhatsApp")
wa = whatsapp_summary(plan_df)
st.code(wa)

csv = plan_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar plan (CSV)",
    data=csv,
    file_name=f"plan_{work_date}.csv",
    mime="text/csv"
)
