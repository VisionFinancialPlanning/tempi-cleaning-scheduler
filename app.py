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
    pwd = st.secrets.get("APP_PASSWORD", None)
    if not pwd:
        return True
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 Acceso")
    token = st.sidebar.text_input("Contraseña", type="password")
    if token == pwd:
        st.sidebar.success("Acceso concedido")
        return True
    else:
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
        start = c1.time_input(f"Inicio {label}", value=default_start)
        end = c2.time_input(f"Fin {label}", value=default_end)
        l1, l2 = st.columns(2)
        lunch_start = l1.time_input(f"Almuerzo inicio {label}", value=time(12, 30))
        lunch_end = l2.time_input(f"Almuerzo fin {label}", value=time(13, 30))
        return start, end, lunch_start, lunch_end

    e1_start, e1_end, e1_l1, e1_l2 = shift_block(emp1_name)
    e2_start, e2_end, e2_l1, e2_l2 = shift_block(emp2_name)

    default_duration = st.number_input("Duración por defecto (min)", min_value=30, max_value=300, value=120, step=15)
    buffer_minutes = st.number_input("Buffer entre limpiezas (min)", min_value=0, max_value=60, value=10, step=5)
    travel_minutes = st.number_input("Tiempo de traslado fijo (min)", min_value=0, max_value=60, value=0, step=5)
    early_priority = st.checkbox("Priorizar check-ins más tempranos", value=True)

    st.markdown("---")
    st.caption("Excel: apartment, unit_id(opc), guest_name(opc), checkout, checkin, clean_duration_minutes(opc)")
    use_apt_cfg = st.checkbox("Usar matriz de tiempos por apartamento (apartment_config.csv)", value=True)

uploaded = st.file_uploader("Sube tu Excel de reservas • O usa `sample_bookings.xlsx`", type=["xlsx", "xls", "csv"])

def parse_bookings(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for k in ["apartment", "unit_id", "guest_name", "checkout", "checkin", "clean_duration_minutes"]:
        for orig_lower, orig in cols.items():
            if orig_lower == k:
                ren[orig] = k
    df = df.rename(columns=ren)
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
        if s.dt.tz is None:
            s = s.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
        else:
            s = s.dt.tz_convert(TZ)
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
            "checkout": checkout,
            "checkin": checkin,
            "start_window": start_window,
            "deadline": deadline,
            "duration_min": duration
        })
    return pd.DataFrame(jobs)

class EmployeeTimeline:
    def __init__(self, name, start_time: time, end_time: time, lunch_start: time, lunch_end: time, day: datetime.date):
        self.name = name
        self.start_dt = TZ.localize(datetime.combine(day, start_time))
        self.end_dt = TZ.localize(datetime.combine(day, end_time))
        self.lunch_start = TZ.localize(datetime.combine(day, lunch_start))
        self.lunch_end = TZ.localize(datetime.combine(day, lunch_end))
        self.cursor = self.start_dt
        self.slots = []
        self.slots.append({
            "employee": self.name,
            "apartment": "ALMUERZO",
            "unit_id": "",
            "guest_name": "",
            "start": self.lunch_start,
            "end": self.lunch_end,
            "duration_min": int((self.lunch_end - self.lunch_start).total_seconds() // 60)
        })
    def _overlaps_lunch(self, start, end):
        return (start < self.lunch_end) and (end > self.lunch_start)
    def trial(self, job, buffer_min=10, travel_min=0):
        est = max(self.cursor, job["start_window"] if pd.notna(job["start_window"]) else self.start_dt, self.start_dt)
        end = est + pd.Timedelta(minutes=int(job["duration_min"]))
        if self._overlaps_lunch(est, end):
            est = max(self.lunch_end, job["start_window"] if pd.notna(job["start_window"]) else self.start_dt, self.start_dt)
            end = est + pd.Timedelta(minutes=int(job["duration_min"]))
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
        best_emp, best_trial = min(candidates, key=lambda x: x[1]["end"])
        best_emp.schedule(job, buffer_min=buffer_min, travel_min=travel_min)
        assignments.append(best_trial)
    out = []
    for emp in employees:
        out.extend(emp.slots)
    plan_df = pd.DataFrame(out).sort_values(["employee", "start"])
    un_df = pd.DataFrame(unassigned)
    return plan_df, un_df

def plot_gantt(plan_df):
    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * max(len(plan_df), 1)))
    if plan_df.empty:
        ax.set_title("No hay tareas asignadas")
        st.pyplot(fig)
        return
    emps = list(plan_df["employee"].unique())
    y_map = {e: i for i, e in enumerate(emps)}
    for _, row in plan_df.iterrows():
        y = y_map[row["employee"]]
        ax.barh(
            y,
            (row["end"] - row["start"]).total_seconds() / 60.0,
            left=row["start"].timestamp() / 60.0,
            height=0.4
        )
        label = "🥪 Almuerzo" if row["apartment"] == "ALMUERZO" else f"{row['apartment']} ({int(row['duration_min'])}m)"
        ax.text(row["start"].timestamp() / 60.0, y, label, va="center", ha="left", fontsize=8)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(emps)
    ax.set_xlabel("Minutos desde epoch (escala relativa)")
    ax.set_title("Gantt de Limpiezas")
    st.pyplot(fig)

def whatsapp_summary(plan_df):
    if plan_df.empty:
        return "No hay tareas asignadas."
    lines = []
    for emp in plan_df["employee"].unique():
        sub = plan_df[plan_df["employee"] == emp].sort_values("start")
        lines.append(f"*Turno {emp}*")
        for _, r in sub.iterrows():
            if r["apartment"] == "ALMUERZO":
                lines.append(f"• {r['start'].strftime('%H:%M')}–{r['end'].strftime('%H:%M')}  🥪 Almuerzo")
            else:
                lines.append(f"• {r['start'].strftime('%H:%M')}–{r['end'].strftime('%H:%M')}  {r['apartment']} ({int(r['duration_min'])}m)")
        lines.append("")
    return "\n".join(lines)

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
else:
    st.info("Usando datos de ejemplo `sample_bookings.xlsx` (en el repo)")
    df_raw = pd.read_excel("sample_bookings.xlsx")

df = parse_bookings(df_raw)
day_df = filter_day(df, work_date)
st.subheader("📄 Reservas del día")
st.dataframe(day_df)

apt_cfg = load_apartment_config() if use_apt_cfg else pd.DataFrame()
jobs_df = build_cleaning_jobs(day_df, work_date, default_duration, apt_cfg)

e1 = EmployeeTimeline(emp1_name, e1_start, e1_end, e1_l1, e1_l2, work_date)
e2 = EmployeeTimeline(emp2_name, e2_start, e2_end, e2_l1, e2_l2, work_date)
plan_df, un_df = greedy_assign(
    jobs_df, [e1, e2],
    buffer_min=buffer_minutes,
    travel_min=travel_minutes,
    early_priority=early_priority
)

st.subheader("🗓️ Plan del día (incluye 🥪 Almuerzo)")
st.dataframe(plan_df)

st.subheader("⚠️ No asignadas (revisar)")
st.dataframe(un_df)

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
