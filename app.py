# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

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
    Devuelve Serie datetime (localizada en America/Panama si es posible).
    Parcheado para evitar AttributeError en tz_localize.
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

    # Si la serie no es datetime o está vacía, regresamos tal cual
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
    Usa nombres de columnas EXACTOS:
      - Check-In (fecha)
      - Check-In Hora (hora)
      - Check-Out (fecha)
      - Check-Out Hora (hora)
    Si faltan horas, aplica CI=15:00 / CO=12:00.
    """
    required = ["Check-In", "Check-In Hora", "Check-Out", "Check-Out Hora"]
    missing = [c for c in ["Check-In", "Check-Out"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}. Debe incluir al menos 'Check-In' y 'Check-Out'.")

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
    # Mantener columnas opcionales comunes si existen
    for optional in ["apartment", "unit_id", "guest_name", "Apartamento", "Unidad", "Huésped"]:
        if optional in df.columns:
            out[optional] = df[optional]

    out["checkin"] = ci_dt
    out["checkout"] = co_dt
    out["nights"] = (out["checkout"].dt.tz_convert(TZ_NAME).dt.date - out["checkin"].dt.tz_convert(TZ_NAME).dt.date).apply(
        lambda d: d.days if pd.notna(d) else None
    )

    # Derivadas para mostrar/descargar
    try:
        out["checkin_day"] = out["checkin"].dt.tz_convert(TZ_NAME).dt.strftime("%Y-%m-%d")
    except Exception:
        out["checkin_day"] = pd.NaT
    try:
        out["checkin_time"] = out["checkin"].dt.tz_convert(TZ_NAME).dt.strftime("%H:%M")
    except Exception:
        out["checkin_time"] = ""
    try:
        out["checkout_day"] = out["checkout"].dt.tz_convert(TZ_NAME).dt.strftime("%Y-%m-%d")
    except Exception:
        out["checkout_day"] = pd.NaT
    try:
        out["checkout_time"] = out["checkout"].dt.tz_convert(TZ_NAME).dt.strftime("%H:%M")
