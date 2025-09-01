import io
import re
from datetime import datetime, time

import pandas as pd
import pytz
import streamlit as st

# ==============================
# CONFIGURACIÓN GENERAL APP
# ==============================
TZ = pytz.timezone("America/Panama")
st.set_page_config(page_title="Tempi – Scheduler & Bookings", page_icon="🧹", layout="wide")
st.title("Tempi – Scheduler & Bookings 🧹📒")

st.caption("Todo en un solo lugar: dashboard de limpieza + carga/normalización de reservas por Excel. Zona horaria: America/Panama.")

# ==============================
# HELPERS COMUNES (Fechas/Horas)
# ==============================
DATE_COL_PATTERNS = [
    r"check[-_\s]*in[\s_-]*date",
    r"check[-_\s]*out[\s_-]*date",
    r"arrival|arribo|llegada",
    r"departure|salida",
    r"fecha\s*check\s*in",
    r"fecha\s*check\s*out",
    r"checkin\s*date|checkout\s*date",
]
TIME_COL_PATTERNS = [
    r"check[-_\s]*in[\s_-]*time",
    r"check[-_\s]*out[\s_-]*time",
    r"hora\s*check\s*in|hora\s*entrada|hora\s*ingreso",
    r"hora\s*check\s*out|hora\s*salida",
    r"arrival\s*time|departure\s*time",
]
APT_PATTERNS = [r"apto|apartment|unidad|listing|propiedad|unit|room|departamento|apt|apartment\s*name"]
GUEST_PATTERNS = [r"guest|hu(e|é)sped|name|cliente|reserv(a|ation)\s*name|contact"]


def _best_match(colnames, patterns):
    lower_map = {c: c.lower() for c in colnames}
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in colnames:
            if rx.search(lower_map[c]):
                return c
    return None


def _coerce_date(series):
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def _coerce_time(series):
    def parse_one(v):
        if pd.isna(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.time()
        if isinstance(v, (int, float)):
            try:
                seconds = float(v) * 24 * 3600
                seconds = int(round(seconds))
                return (datetime(1900, 1, 1) + pd.to_timedelta(seconds, unit="s")).time()
            except Exception:
                return None
        s = str(v).strip()
        s = s.replace(".", ":")
        try:
            t = pd.to_datetime(s, errors="coerce").time()
            return t
        except Exception:
            pass
        m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
        if m:
            h = int(m.group(1)); mi = int(m.group(2)); se = int(m.group(3) or 0)
            h = h % 24
            return datetime(1900,1,1,h,mi,se).time()
        return None

    return series.apply(parse_one)


def _combine_date_time(date_series, time_series, default_time=None):
    dt = pd.to_datetime(date_series, errors="coerce")
    times = _coerce_time(time_series) if time_series is not None else None
    out = []
    for d, t in zip(dt, times if times is not None else [None]*len(dt)):
        if pd.isna(d):
            out.append(pd.NaT)
        else:
            if t is None and default_time is not None:
                t = default_time
            elif t is None:
                out.append(pd.NaT)
                continue
            out.append(pd.Timestamp.combine(d.date(), t))
    return pd.to_datetime(out, errors="coerce").dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT").dt.tz_convert(TZ)


def normalize_bookings(df_raw: pd.DataFrame,
                       col_checkin_date: str,
                       col_checkout_date: str,
                       col_checkin_time: str | None,
                       col_checkout_time: str | None,
                       col_apartment: str | None,
                       col_guest: str | None) -> pd.DataFrame:
    """Normaliza un DataFrame de reservas y devuelve columnas estandarizadas."""
    out = pd.DataFrame()
    if col_apartment:
        out["apartment"] = df_raw[col_apartment].astype(str).str.strip()
    if col_guest:
        out["guest_name"] = df_raw[col_guest].astype(str).str.strip()

    out["checkin_date_raw"] = _coerce_date(df_raw[col_checkin_date])
    out["checkout_date_raw"] = _coerce_date(df_raw[col_checkout_date])

    checkin_dt = _combine_date_time(
        df_raw[col_checkin_date],
        df_raw[col_checkin_time] if col_checkin_time else None,
        default_time=time(15, 0),
    )
    checkout_dt = _combine_date_time(
        df_raw[col_checkout_date],
        df_raw[col_checkout_time] if col_checkout_time else None,
        default_time=time(12, 0),
    )

    out["checkin_at"] = checkin_dt
    out["checkout_at"] = checkout_dt

    out["nights"] = (out["checkout_at"].dt.date - out["checkin_at"].dt.date).apply(lambda d: d.days if pd.notna(d) else None)
    out["checkin_day"] = out["checkin_at"].dt.strftime("%Y-%m-%d")
    out["checkin_time"] = out["checkin_at"].dt.strftime("%H:%M").fillna("")
    out["checkout_day"] = out["checkout_at"].dt.strftime("%Y-%m-%d")
    out["checkout_time"] = out["checkout_at"].dt.strftime("%H:%M").fillna("")

    preferred_cols = [
        "apartment", "guest_name", "checkin_day", "checkin_time", "checkout_day", "checkout_time", "nights",
    ]
    ordered = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    return out[ordered]


# ==============================
# LAYOUT PRINCIPAL (TABS)
# ==============================
TAB1, TAB2 = st.tabs(["🧭 Horario de Limpieza (existente)", "📒 Cargar/Normalizar Reservas (Excel)"])

with TAB1:
    st.subheader("Horario de limpieza")
    st.info("Aquí va tu lógica actual del scheduler. (Integramos sin tocar tu flujo).")
    # TODO: Pega aquí tu código existente del scheduler (lo que ya tenías en app.py)

with TAB2:
    st.subheader("Cargar/Normalizar Reservas (Excel)")
    st.write(
        "Puedes subir tu Excel. Si no subes nada, cargaremos automáticamente **sample_bookings.xlsx** del repo como archivo de ejemplo. Si tampoco existe, te lo indicaremos."
    )

    uploaded = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

    # Carga fuente: 1) lo subido 2) sample_bookings.xlsx
    df_raw = None
    source_label = None

    if uploaded is not None:
        try:
            xl = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Selecciona la hoja a procesar", xl.sheet_names)
            df_raw = xl.parse(sheet)
            source_label = f"Archivo subido: {uploaded.name} – Hoja: {sheet}"
        except Exception as e:
            st.error(f"No se pudo leer el archivo subido: {e}")
    else:
        # Fallback: sample_bookings.xlsx en el repo
        try:
            xl = pd.ExcelFile("sample_bookings.xlsx")
            sheet = st.selectbox("Selecciona la hoja a procesar", xl.sheet_names)
            df_raw = xl.parse(sheet)
            source_label = f"Archivo de ejemplo: sample_bookings.xlsx – Hoja: {sheet}"
        except Exception as e:
            st.warning("No subiste archivo y no se encontró sample_bookings.xlsx en el repo.")

    if df_raw is None:
        st.stop()

    st.write(f"**Fuente:** {source_label}")
    st.write("### Vista previa (primeras 10 filas)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    cols = list(df_raw.columns)
    col_checkin_date = _best_match(cols, [DATE_COL_PATTERNS[0], DATE_COL_PATTERNS[2], DATE_COL_PATTERNS[4], DATE_COL_PATTERNS[6]])
    col_checkout_date = _best_match(cols, [DATE_COL_PATTERNS[1], DATE_COL_PATTERNS[3], DATE_COL_PATTERNS[5]])
    col_checkin_time = _best_match(cols, [TIME_COL_PATTERNS[0], TIME_COL_PATTERNS[2]])
    col_checkout_time = _best_match(cols, [TIME_COL_PATTERNS[1], TIME_COL_PATTERNS[3]])
    col_apartment = _best_match(cols, APT_PATTERNS)
    col_guest = _best_match(cols, GUEST_PATTERNS)

    st.write("### Mapeo de columnas")
    col1, col2, col3 = st.columns(3)
    with col1:
        col_checkin_date = st.selectbox("Fecha Check‑in", [None] + cols, index=(cols.index(col_checkin_date)+1) if col_checkin_date in cols else 0)
        col_checkin_time = st.selectbox("Hora Check‑in (opcional)", [None] + cols, index=(cols.index(col_checkin_time)+1) if col_checkin_time in cols else 0)
    with col2:
        col_checkout_date = st.selectbox("Fecha Check‑out", [None] + cols, index=(cols.index(col_checkout_date)+1) if col_checkout_date in cols else 0)
        col_checkout_time = st.selectbox("Hora Check‑out (opcional)", [None] + cols, index=(cols.index(col_checkout_time)+1) if col_checkout_time in cols else 0)
    with col3:
        col_apartment = st.selectbox("Columna Apartamento/Unidad (opcional)", [None] + cols, index=(cols.index(col_apartment)+1) if col_apartment in cols else 0)
        col_guest = st.selectbox("Columna Huésped/Nombre (opcional)", [None] + cols, index=(cols.index(col_guest)+1) if col_guest in cols else 0)

    if not col_checkin_date or not col_checkout_date:
        st.error("Debes seleccionar al menos **Fecha Check‑in** y **Fecha Check‑out**.")
        st.stop()

    out = normalize_bookings(
        df_raw,
        col_checkin_date=col_checkin_date,
        col_checkout_date=col_checkout_date,
        col_checkin_time=col_checkin_time if col_checkin_time else None,
        col_checkout_time=col_checkout_time if col_checkout_time else None,
        col_apartment=col_apartment if col_apartment else None,
        col_guest=col_guest if col_guest else None,
    )

    st.success("✅ Listo. Abajo puedes revisar los resultados y descargarlos.")
    st.dataframe(out, use_container_width=True)

    # Botones de descarga
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="checkins_checkouts.csv", mime="text/csv")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="checkins_checkouts", index=False)

    st.download_button("⬇️ Descargar Excel", data=excel_buf.getvalue(), file_name="checkins_checkouts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption("Si el archivo no trae horas, se asigna automáticamente: Check‑in 3:00 PM, Check‑out 12:00 PM.")
