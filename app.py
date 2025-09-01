# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import io
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
# CONFIGURACIÓN GENERAL
# ==============================
TZ_NAME = "America/Panama"
TZ = pytz.timezone(TZ_NAME)

st.set_page_config(page_title="Tempi – Scheduler & Bookings", page_icon="🧹", layout="wide")
st.title("Tempi – Scheduler & Bookings 🧹📒")
st.caption(f"Zona horaria aplicada: {TZ_NAME}")

# ==============================
# HELPERS DE FECHA/HORA
# ==============================
def _coerce_time(series):
    """Convierte distintos formatos de hora a time() (o None)."""
    if series is None:
        return pd.Series(dtype="object")
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
    """
    Combina fecha (obligatoria) + hora (opcional). Si la hora falta, usa default_time.
    Devuelve Serie datetime; intenta localizar a America/Panama sin romper si falla.
    """
    dates = pd.to_datetime(date_s, errors="coerce")
    times = _coerce_time(time_s) if time_s is not None else pd.Series([None] * len(dates))

    out = []
    for d, t in zip(dates, times):
        if pd.isna(d):
            out.append(pd.NaT)
        else:
            if t is None:
                t = default_time
            out.append(pd.Timestamp.combine(pd.Timestamp(d).date(), t))
    ser = pd.to_datetime(out, errors="coerce")

    # Si no es datetime o está vacío, regresamos tal cual
    if ser.empty or not _is_dt(ser):
        return ser

    # Intentar localizar a TZ; si falla, devolver naive
    try:
        ser = ser.dt.tz_localize(TZ_NAME, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        return ser
    return ser

def parse_bookings_with_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa nombres EXACTOS:
      - Check-In (fecha)
      - Check-In Hora (hora) [opcional]
      - Check-Out (fecha)
      - Check-Out Hora (hora) [opcional]
    Si faltan horas, aplica CI=15:00 / CO=12:00.
    """
    missing_dates = [c for c in ["Check-In", "Check-Out"] if c not in df.columns]
    if missing_dates:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(missing_dates)
            + ". Debe incluir al menos 'Check-In' y 'Check-Out'."
        )

    ci_dt = _combine(
        df["Check-In"],
        df["Check-In Hora"] if "Check-In Hora" in df.columns else None,
        default_time=time(15, 0),
    )
    co_dt = _combine(
        df["Check-Out"],
        df["Check-Out Hora"] if "Check-Out Hora" in df.columns else None,
        default_time=time(12, 0),
    )

    out = pd.DataFrame()

    # Mantener columnas útiles si existen
    for optional in ["apartment", "unit_id", "guest_name", "Apartamento", "Unidad", "Huésped"]:
        if optional in df.columns:
            out[optional] = df[optional]

    out["checkin"] = ci_dt
    out["checkout"] = co_dt

    # Derivadas para mostrar/descargar (robustas si falta tz)
    def _tz_fmt(series, fmt):
        try:
            return series.dt.tz_convert(TZ_NAME).dt.strftime(fmt)
        except Exception:
            try:
                # Si viene naive
                return series.dt.strftime(fmt)
            except Exception:
                return pd.Series([""] * len(series))

    out["checkin_day"] = _tz_fmt(out["checkin"], "%Y-%m-%d")
    out["checkin_time"] = _tz_fmt(out["checkin"], "%H:%M")
    out["checkout_day"] = _tz_fmt(out["checkout"], "%Y-%m-%d")
    out["checkout_time"] = _tz_fmt(out["checkout"], "%H:%M")

    # Noches (intenta con tz y fallback sin tz)
    try:
        nights = (
            out["checkout"].dt.tz_convert(TZ_NAME).dt.date
            - out["checkin"].dt.tz_convert(TZ_NAME).dt.date
        )
    except Exception:
        nights = (
            pd.to_datetime(out["checkout"], errors="coerce").dt.date
            - pd.to_datetime(out["checkin"], errors="coerce").dt.date
        )
    out["nights"] = [d.days if pd.notna(d) else None for d in nights]

    preferred = [
        "apartment", "unit_id", "guest_name",
        "checkin_day", "checkin_time",
        "checkout_day", "checkout_time",
        "nights", "checkin", "checkout",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

# ==============================
# SCHEDULER (lógica integrada)
# ==============================
@st.cache_data
def load_apartment_config():
    """Carga apartment_config.csv si existe; si no, DataFrame vacío."""
    try:
        return pd.read_csv("apartment_config.csv")
    except Exception:
        return pd.DataFrame()

def to_tz(series: pd.Series) -> pd.Series:
    """Convierte serie a datetime TZ; si ya trae tz, intenta convertir; si no, localiza."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_convert(TZ_NAME)
    except Exception:
        try:
            s = s.dt.tz_localize(TZ_NAME)
        except Exception:
            pass
    return s

def filter_day(df: pd.DataFrame, day) -> pd.DataFrame:
    """Filtra reservas cuyo check-out o check-in caen en el día indicado."""
    start = pd.Timestamp(day, tz=TZ)
    end = start + pd.Timedelta(days=1)
    if "checkout" in df.columns:
        df["checkout"] = to_tz(df["checkout"])
    if "checkin" in df.columns:
        df["checkin"] = to_tz(df["checkin"])
    mask = (
        (df["checkout"].between(start, end, inclusive="left")) |
        (df["checkin"].between(start, end, inclusive="left"))
    )
    return df[mask].copy()

def infer_duration(row, default_duration, cfg):
    """Duración: por fila (clean_duration_minutes), por apto (apartment_config), o default."""
    if "clean_duration_minutes" in row and pd.notna(row["clean_duration_minutes"]):
        return int(row["clean_duration_minutes"])
    if cfg is not None and not cfg.empty and "apartment" in cfg.columns:
        apt_name = str(row.get("apartment", row.get("Apartamento", ""))).strip().lower()
        hit = cfg[cfg["apartment"].astype(str).str.lower() == apt_name]
        if not hit.empty:
            return int(hit.iloc[0].get("base_minutes", default_duration))
    return default_duration

def build_cleaning_jobs(day_df: pd.DataFrame, day, default_duration: int, cfg: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for _, row in day_df.iterrows():
        apt = row.get("apartment", row.get("Apartamento", "Apto"))
        unit = row.get("unit_id", row.get("Unidad", ""))
        guest = row.get("guest_name", row.get("Huésped", ""))

        checkout = row.get("checkout", pd.NaT)
        checkin = row.get("checkin", pd.NaT)

        duration = infer_duration(row, default_duration, cfg)

        # ventana mínima: desde checkout (si existe) o 08:00
        start_window = checkout if pd.notna(checkout) else TZ.localize(datetime.combine(pd.Timestamp(day).date(), time(8, 0)))
        # deadline: checkin del mismo día si existe
        deadline = checkin if (pd.notna(checkin) and checkin.date() == pd.Timestamp(day).date()) else pd.NaT

        jobs.append({
            "apartment": apt,
            "unit_id": unit,
            "guest_name": guest,
            "start_window": start_window,
            "deadline": deadline,
            "duration_min": int(duration),
        })
    return pd.DataFrame(jobs)

class EmployeeTimeline:
    def __init__(self, name, start_t: time, end_t: time, lunch_start: time, lunch_end: time, day):
        self.name = name
        self.day = pd.Timestamp(day).date()
        self.start_dt = TZ.localize(datetime.combine(self.day, start_t))
        self.end_dt = TZ.localize(datetime.combine(self.day, end_t))
        self.lunch_start = TZ.localize(datetime.combine(self.day, lunch_start))
        self.lunch_end = TZ.localize(datetime.combine(self.day, lunch_end))
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
            "duration_min": job["duration_min"],
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
    # Rellenar deadlines vacíos con muy tarde
    min_day = sort_keys["start_window"].dt.date.min()
    far = TZ.localize(datetime.combine(min_day, time(23, 59)))
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
        # elegir quien termina más temprano
        emp_sel, slot = sorted(candidates, key=lambda x: x[1]["end"])[0]
        emp_sel.schedule(job, buffer_min=buffer_min, travel_min=travel_min)
        assignments.append(slot)

    return pd.DataFrame(assignments), pd.DataFrame(unassigned)

def plot_gantt(plan_df: pd.DataFrame):
    if plan_df is None or plan_df.empty:
        st.info("No hay asignaciones para mostrar.")
        return
    fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(plan_df)))
    employees = list(plan_df["employee"].unique())
    y_map = {e: i for i, e in enumerate(employees)}
    day0 = plan_df["start"].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for _, row in plan_df.iterrows():
        y = y_map[row["employee"]]
        start = row["start"].to_pydatetime()
        end = row["end"].to_pydatetime()
        left = (start - day0).total_seconds() / 3600
        width = (end - start).total_seconds() / 3600
        ax.barh(y, width, left=left)
        ax.text(left + width/2, y, f"{row['apartment']}", va="center", ha="center", fontsize=9)
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
        sub = plan_df[plan_df["employee"] == emp]
        for _, r in sub.sort_values("start").iterrows():
            lines.append(f"• {r['apartment']} — {r['start'].strftime('%H:%M')}–{r['end'].strftime('%H:%M')} ({int(r['duration_min'])}m)")
    return "\n".join(lines)

# ==============================
# SIDEBAR – PARÁMETROS
# ==============================
with st.sidebar:
    st.header("⚙️ Configuración del Día")
    work_date = st.date_input("Día a planificar", value=pd.Timestamp.now(TZ).date())

    st.subheader("Empleadas")
    emp1_name = st.text_input("Empleado 1", value="Mayerlin")
    emp2_name = st.text_input("Empleado 2", value="Evelyn")

    def shift_block(label, default_start=time(8,0), default_end=time(17,0)):
        c1, c2 = st.columns(2)
        with c1:
            s = st.time_input(f"{label} – inicio", value=default_start)
        with c2:
            e = st.time_input(f"{label} – fin", value=default_end)
        return s, e

    e1_start, e1_end = shift_block("Empleado 1", time(8,0), time(17,0))
    e2_start, e2_end = shift_block("Empleado 2", time(8,0), time(17,0))

    st.subheader("Almuerzo")
    e1_l1, e1_l2 = shift_block("Empleado 1 – Almuerzo", time(12,0), time(13,0))
    e2_l1, e2_l2 = shift_block("Empleado 2 – Almuerzo", time(12,30), time(13,30))

    st.subheader("Parámetros del Plan")
    buffer_minutes = st.number_input("Buffer entre limpiezas (min)", value=10, step=5)
    travel_minutes = st.number_input("Traslado entre apartamentos (min)", value=0, step=5)
    default_duration = st.number_input("Duración por defecto (min)", value=90, step=5)
    early_priority = st.checkbox("Priorizar salidas tempranas (deadline primero)", value=True)
    use_apt_cfg = st.checkbox("Usar apartment_config.csv (tiempos por apto)", value=True)

# ==============================
# CARGA → NORMALIZA → SCHEDULER (UN SOLO FLUJO)
# ==============================
uploaded = st.file_uploader(
    "Sube tu Excel de reservas (.xlsx). Requeridas: 'Check-In', 'Check-Out'. Opcionales: 'Check-In Hora', 'Check-Out Hora'",
    type=["xlsx"]
)

if uploaded is not None:
    df_raw = pd.read_excel(uploaded)
else:
    # Fallback a archivo de ejemplo
    try:
        st.info("Usando datos de ejemplo: sample_bookings.xlsx")
        df_raw = pd.read_excel("sample_bookings.xlsx")
    except Exception:
        st.error("No subiste archivo y no se encontró sample_bookings.xlsx en el repo.")
        st.stop()

st.subheader("📄 Reservas – Original (preview)")
st.dataframe(df_raw.head(10), use_container_width=True)

# Normalizar (con tus columnas exactas)
try:
    normalized = parse_bookings_with_fixed_columns(df_raw)
except Exception as e:
    st.error(f"Error al normalizar reservas: {e}")
    st.stop()

st.success("✅ Reservas normalizadas")
st.dataframe(normalized, use_container_width=True)

# Descargas del normalizado
csv_bytes = normalized.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar Normalizado (CSV)", data=csv_bytes, file_name="checkins_checkouts.csv", mime="text/csv")

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    normalized.to_excel(writer, sheet_name="checkins_checkouts", index=False)
st.download_button(
    "⬇️ Descargar Normalizado (Excel)",
    data=excel_buf.getvalue(),
    file_name="checkins_checkouts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ==============================
# Construcción de trabajos y asignación
# ==============================
st.header("🧭 Horario de Limpieza")

apt_cfg = load_apartment_config() if use_apt_cfg else pd.DataFrame()

# Filtrar por día seleccionado
day_df = filter_day(normalized, work_date)
st.subheader("📄 Reservas del día (normalizadas)")
st.dataframe(day_df, use_container_width=True)

# Generar trabajos
jobs_df = build_cleaning_jobs(day_df, work_date, default_duration, apt_cfg)
st.subheader("🧱 Trabajos a programar")
st.dataframe(jobs_df, use_container_width=True)

# Timelines / Empleadas
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

# Descargar plan asignado
plan_csv = plan_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar plan (CSV)",
    data=plan_csv,
    file_name=f"plan_{work_date}.csv",
    mime="text/csv"
)
