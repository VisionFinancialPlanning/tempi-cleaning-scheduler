# app.py
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import re, unicodedata, hashlib, io
import urllib.parse as _q
from datetime import datetime, time, timedelta

import pytz
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype as _is_dt

# ---- Matplotlib (opcional) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

# ==============================
# CONFIG
# ==============================
TZ_NAME = "America/Panama"
TZ = pytz.timezone(TZ_NAME)

st.set_page_config(page_title="Tempi – Scheduler & Bookings", page_icon="🧹", layout="wide")
st.title("Tempi – Scheduler & Bookings 🧹📒")
st.caption(f"Zona horaria aplicada: {TZ_NAME}")

# ==============================
# 🔒 PASSWORD
# ==============================
def _get_secret(key, default=""):
    try: return st.secrets.get(key, default)
    except Exception: return default

SECRET_PLAIN = os.environ.get("TEMPI_PASSWORD") or _get_secret("APP_PASSWORD", "")
SECRET_SHA   = os.environ.get("TEMPI_PASSWORD_SHA256") or _get_secret("APP_PASSWORD_SHA256", "")

def _check_plain(pwd: str) -> bool:
    return bool(SECRET_PLAIN) and (pwd == SECRET_PLAIN)

def _check_hash(pwd: str) -> bool:
    if not SECRET_SHA: return False
    try: return hashlib.sha256(pwd.encode("utf-8")).hexdigest() == SECRET_SHA.lower()
    except Exception: return False

def check_password():
    if not SECRET_PLAIN and not SECRET_SHA:
        with st.sidebar:
            st.info("🔓 Sin contraseña. Configura `APP_PASSWORD` o `APP_PASSWORD_SHA256` en *Secrets* si deseas protegerla.")
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
        with c1: ok = st.button("Entrar")
        with c2: rst = st.button("Limpiar")
        if rst: st.experimental_rerun()
        if ok:
            if _check_plain(pwd) or _check_hash(pwd):
                st.session_state.pass_ok = True
                st.experimental_rerun()
            else:
                st.error("Contraseña incorrecta."); st.stop()
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
        if an in norm_map: return norm_map[an]
    for a in aliases:
        an = _norm(a)
        for key, real in norm_map.items():
            if an in key: return real
    return None

def _parse_date_flex(series):
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s2 if s2.notna().sum() > s1.notna().sum() else s1

def _coerce_time(series):
    if series is None: return pd.Series(dtype="object")
    def parse_one(v):
        if pd.isna(v): return None
        if isinstance(v, (pd.Timestamp, datetime)): return v.time()
        if isinstance(v, (int, float)):
            try:
                frac = float(v) % 1.0
                secs = int(round(frac * 24 * 3600))
                hh, mm, ss = secs // 3600, (secs % 3600) // 60, secs % 60
                return time(hh % 24, mm, ss)
            except Exception: return None
        s = str(v).strip().replace(".", ":")
        try:
            t = pd.to_datetime(s, errors="coerce").time()
            if t is not None: return t
        except Exception: pass
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
    dates = _parse_date_flex(date_s)
    out = []
    for d in dates:
        if pd.isna(d):
            out.append(None)
        else:
            hh, mm, ss = d.hour, d.minute, d.second
            out.append(None if (hh, mm, ss) == (0, 0, 0) else time(hh, mm, ss))
    return pd.Series(out, index=dates.index)

def _combine_smart(date_s, explicit_time_s, default_time: time):
    dates = _parse_date_flex(date_s)
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
    try: ser = ser.dt.tz_localize(TZ_NAME, nonexistent="NaT", ambiguous="NaT")
    except Exception: return ser
    return ser

def _fmt_tz(series, fmt):
    try: return series.dt.tz_convert(TZ_NAME).dt.strftime(fmt)
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
    if s.count(",") > 0 and s.count(".") == 0: s = s.replace(",", ".")
    else: s = s.replace(",", "")
    try: return float(s)
    except Exception: return np.nan

# --- PARKING estricto ---
def _parse_parking_count(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float)) and not np.isnan(x):
        return int(max(0, x))
    s = str(x).strip().lower()
    if s in ["", "no", "0", "none", "n/a", "na", "-", "sin", "ninguno"]: return 0
    m = re.search(r"(\d+)", s)
    if m:
        try: return int(m.group(1))
        except: return 1
    positivos = {"si","sí","yes","carro","auto","vehiculo","vehículo","parqueo","parking","estacionamiento","space"}
    if any(w in s for w in positivos): return 1
    return 0

def _is_si(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return False
    s = str(x).strip().lower().replace("í", "i")
    return s in {"si", "sí", "yes"}

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

def _pick_name(row):
    g = str(row.get("guest_name","")).strip()
    return g if g else str(row.get("nombre","")).strip()

def _clean_phone(val: str) -> str:
    s = re.sub(r"\D", "", str(val or ""))
    if len(s) == 8: s = "507" + s
    if s.startswith("00"): s = s[2:]
    return s

def _hora_from_cols(row, kind):
    if kind == "ci":
        if "checkin_time" in row and str(row["checkin_time"]).strip(): return row["checkin_time"]
        if "checkin" in row and pd.notna(row["checkin"]):
            try: return row["checkin"].tz_convert(TZ_NAME).strftime("%H:%M")
            except Exception: return pd.to_datetime(row["checkin"]).strftime("%H:%M")
    else:
        if "checkout_time" in row and str(row["checkout_time"]).strip(): return row["checkout_time"]
        if "checkout" in row and pd.notna(row["checkout"]):
            try: return row["checkout"].tz_convert(TZ_NAME).strftime("%H:%M")
            except Exception: return pd.to_datetime(row["checkout"]).strftime("%H:%M")
    return ""

def _fecha_es_larga(iso_yyyy_mm_dd: str) -> str:
    try: d = datetime.strptime(iso_yyyy_mm_dd, "%Y-%m-%d")
    except Exception: return iso_yyyy_mm_dd
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    return f"{d.day} de {meses[d.month-1]}"

def _hora_12h(hhmm: str) -> str:
    try:
        h, m = map(int, str(hhmm).split(":")[:2]); suf = "am"
        if h == 0: h = 12; suf = "am"
        elif h == 12: suf = "pm"
        elif h > 12: h -= 12; suf = "pm"
        return f"{h}:{m:02d} {suf}"
    except Exception: return str(hhmm)

def _extract_flight(notes: str) -> str:
    s = str(notes or "")
    m = re.search(r"(vuelo\s+[A-Za-z]{1,3}\s?\d{2,5})", s, flags=re.I)
    if m:
        t = m.group(1)
        return " ".join([w.upper() if re.fullmatch(r"[A-Za-z]{1,3}\s?\d{2,5}", w) else w for w in t.split()])
    m2 = re.search(r"\b([A-Za-z]{1,3}\s?\d{2,5})\b", s)
    return f"Vuelo {m2.group(1).upper()}" if m2 else ""

# ==============================
# PARSEO DEL EXCEL
# ==============================
def parse_bookings_with_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    ci_date_col = _find_col(df, [
        "Check-In","fecha check in","fecha checkin","fecha de check in",
        "fecha de check-in","check in","check-in","fecha ingreso","entrada","fecha entrada"
    ])
    co_date_col = _find_col(df, [
        "Check-Out","fecha check out","fecha checkout","fecha de check out",
        "fecha de check-out","check out","check-out","fecha salida","salida","fecha de salida"
    ])
    if ci_date_col is None or co_date_col is None:
        raise ValueError("El archivo debe incluir columnas de fecha de entrada y salida (Check-In / Check-Out o equivalentes).")

    ci_time_col = _find_col(df, [
        "Check-In Hora","hora entrada","Hora Entrada","Hora Check In",
        "Check-In Time","Hora Check-In","CHECK-IN HORA","hora de entrada"
    ])
    co_time_col = _find_col(df, [
        "Check-Out Hora","hora salida","Hora Salida","Hora Check Out",
        "Check-Out Time","Hora Check-Out","CHECK-OUT HORA","hora de salida"
    ])

    ci_dt = _combine_smart(df[ci_date_col], df[ci_time_col] if ci_time_col else None, default_time=time(15,0))
    co_dt = _combine_smart(df[co_date_col], df[co_time_col] if co_time_col else None, default_time=time(12,0))

    ci_from_col  = _coerce_time(df[ci_time_col]).notna() if ci_time_col else pd.Series([False]*len(df))
    co_from_col  = _coerce_time(df[co_time_col]).notna() if co_time_col else pd.Series([False]*len(df))
    ci_from_date = _time_from_datecol(df[ci_date_col]).notna()
    co_from_date = _time_from_datecol(df[co_date_col]).notna()

    def _source(from_col, from_date, default_label):
        return np.where(from_col, "hora_col", np.where(from_date, "en_fecha", default_label))

    out = pd.DataFrame()

    apt_col   = _find_col(df, ["Property Internal Name","Apto","Apartamento","Unidad"])
    guest_col = _find_col(df, ["Guest First Name","Nombre","Name","Guest Name"])
    if apt_col:   out["apartment"]  = df[apt_col].apply(_canon_apt)
    if guest_col: out["guest_name"] = df[guest_col].astype(str)

    map_optional = {
        "number_guests": ["NUMBER GUESTS","Guests","# Guests","Numero Huespedes","Nº Huespedes","Huéspedes"],
        "sofa_or_bed":   ["SOFA OR BED","Sofa Bed","Sofa cama","Sofacama","Sofa as Bed"],
        "reservation_id":["ID","Reservation ID","Booking ID","Código","Codigo"],
        "transporte":    ["TRANSPORTE","Transport","Pickup","Transfer"],
        "extra":         ["EXTRA","Extras","Requests","Solicitudes"],
        "parking":       ["PARKING","Estacionamiento","Parking Spot","Parqueo"],
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

    if "parking" in out.columns: out["parking_count"] = out["parking"].apply(_parse_parking_count)
    else: out["parking_count"] = 0

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
        out["cash_pickup"] = list(cash); out["cash_reason"] = list(why); out["cash_amount"] = list(amt)
    else:
        out["cash_pickup"] = []; out["cash_reason"] = []; out["cash_amount"] = []

    preferred = [
        "apartment","guest_name","number_guests","sofa_or_bed","reservation_id",
        "transporte","extra","parking","parking_count","nombre","notes","phone",
        "channel","payment_method","amount_due","total_price",
        "cash_pickup","cash_amount","cash_reason",
        "checkin_day","checkin_time","checkin_time_source",
        "checkout_day","checkout_time","checkout_time_source",
        "nights","checkin","checkout","transport_ci","transport_co"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

# ==============================
# EVENTOS / DÍA  (ACTUALIZADO)
# ==============================
def _events_por_apartamento(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time, extra_apartments=None):
    start_of_day, end_of_day = _day_bounds(day, day_start_t, day_end_t)

    df = normalized.copy()
    df["checkin"]  = to_tz(df["checkin"])
    df["checkout"] = to_tz(df["checkout"])

    apts = sorted(df["apartment"].dropna().astype(str).unique().tolist())
    if extra_apartments:
        for x in extra_apartments:
            if x not in apts:
                apts.append(x)

    out = {}
    work_date = pd.Timestamp(day).date()

    for apt in apts:
        sub = df[df["apartment"].astype(str) == apt].copy()

        cis_day = [ts for ts in list(sub["checkin"].dropna())  if ts.date() == work_date]
        cos_day = [ts for ts in list(sub["checkout"].dropna()) if ts.date() == work_date]

        pisas = sub[(sub["checkin"] < end_of_day) & (sub["checkout"] > start_of_day)]
        ocupado = not pisas.empty

        ventana = None
        if cis_day and cos_day:
            co = max(cos_day); ci = min(cis_day)
            win_start = max(co, start_of_day)
            win_end   = min(ci, end_of_day)
            if win_end > win_start:
                ventana = (win_start, win_end, "turnover")
        elif cos_day:
            co = max(cos_day)
            win_start = max(co, start_of_day)
            win_end   = end_of_day
            if win_end > win_start:
                ventana = (win_start, win_end, "solo checkout")
        elif cis_day:
            ci = min(cis_day)
            win_start = start_of_day
            win_end   = min(ci, end_of_day)
            if win_end > win_start:
                ventana = (win_start, win_end, "solo check-in")
        elif not ocupado:
            ventana = (start_of_day, end_of_day, "vacio")

        out[apt] = {"cis": cis_day, "cos": cos_day, "ocupado": ocupado, "ventana": ventana}

    return out

def day_summary_collapsed_v2(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time, extra_apts=None) -> pd.DataFrame:
    ev = _events_por_apartamento(normalized, day, day_start_t, day_end_t, extra_apts)
    rows = []

    def _fmt(ts_list):
        if not ts_list:
            return ""
        hs = sorted({ts.strftime("%H:%M") for ts in ts_list})
        return ", ".join(hs)

    for apt, d in ev.items():
        cis, cos, v = d["cis"], d["cos"], d["ventana"]
        if v is None:
            continue
        ini, fin, kind = v
        if kind == "turnover":
            gap = round((fin - ini).total_seconds() / 3600.0, 2)
            rows.append({
                "apartment": apt, "tipo": "Turnover",
                "checkout_time": _fmt(cos), "checkin_time": _fmt(cis),
                "gap_hours": gap
            })
        elif kind == "solo checkout":
            rows.append({
                "apartment": apt, "tipo": "Solo Check-Out",
                "checkout_time": _fmt(cos), "checkin_time": "",
                "gap_hours": ""
            })
        elif kind == "solo check-in":
            rows.append({
                "apartment": apt, "tipo": "Solo Check-In",
                "checkout_time": "", "checkin_time": _fmt(cis),
                "gap_hours": ""
            })
        elif kind == "vacio":
            rows.append({
                "apartment": apt, "tipo": "Vacío (día completo)",
                "checkout_time": "", "checkin_time": "",
                "gap_hours": 24
            })

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
        if fin > ini: rows.append({"apartment": apt, "start": ini, "end": fin, "kind": kind})
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["__bali__"] = out["apartment"].astype(str).eq("BALI")
    out = out.sort_values(["__bali__", "apartment"]).drop(columns="__bali__").reset_index(drop=True)
    return out

def active_parkings_today(normalized: pd.DataFrame, day, day_start_t: time, day_end_t: time) -> pd.DataFrame:
    start_of_day, end_of_day = _day_bounds(day, day_start_t, day_end_t)
    df = normalized.copy()
    df["checkin"]  = to_tz(df["checkin"])
    df["checkout"] = to_tz(df["checkout"])
    df["parking_count"] = pd.to_numeric(df.get("parking_count", 0), errors="coerce").fillna(0).astype(int)
    occ = df[(df["checkin"] < end_of_day) & (df["checkout"] > start_of_day) & (df["parking_count"] > 0)].copy()
    if occ.empty:
        return pd.DataFrame(columns=["apartment","guest_name","parking_count","checkin","checkout",
                                     "checkin_day","checkin_time","checkout_day","checkout_time"])
    occ["checkin_day"]   = occ["checkin"].dt.strftime("%Y-%m-%d")
    occ["checkin_time"]  = occ["checkin"].dt.strftime("%H:%M")
    occ["checkout_day"]  = occ["checkout"].dt.strftime("%Y-%m-%d")
    occ["checkout_time"] = occ["checkout"].dt.strftime("%H:%M")
    return occ[["apartment","guest_name","parking_count","checkin","checkout",
                "checkin_day","checkin_time","checkout_day","checkout_time"]].sort_values(["apartment","checkin"])

# ==============================
# PLANIFICADOR
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
    if plt is None:
        st.error("Para ver el gráfico instala matplotlib (añade `matplotlib` a requirements.txt).")
        return
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
# SIDEBAR
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
# CARGA
# ==============================
uploaded = st.file_uploader(
    "Sube tu Excel (.xlsx). Requeridas: 'Check-In'/'Check-Out' o equivalentes (horas con alias aceptados).",
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

# ====== TAB 1 ======
with tab1:
    start_day, end_day = _day_bounds(work_date, day_start_t, day_end_t)

    st.subheader("📦 Preparación del día (mismo día)")
    df_tmp = normalized.copy()
    df_tmp["checkin"]  = to_tz(df_tmp["checkin"])
    df_tmp["checkout"] = to_tz(df_tmp["checkout"])
    mask_ci = (df_tmp["checkin"]  >= start_day) & (df_tmp["checkin"]  < end_day)
    mask_co = (df_tmp["checkout"] >= start_day) & (df_tmp["checkout"] < end_day)
    prep_df = df_tmp[mask_ci | mask_co].copy()

    if prep_df.empty:
        st.info("No hay movimientos (check-ins/outs) para la fecha seleccionada.")
    else:
        cols_show = [c for c in [
            "apartment","guest_name","number_guests","sofa_or_bed","reservation_id",
            "transporte","extra","parking","parking_count","nombre","notes","phone",
            "channel","payment_method","amount_due","total_price",
            "checkin_day","checkin_time","checkout_day","checkout_time"
        ] if c in prep_df.columns]
        st.dataframe(prep_df[cols_show], use_container_width=True)

    # CASH hoy (por checkout)
    mask_cash = (normalized.get("cash_pickup", False) == True)
    df_cash = normalized.copy()
    df_cash["checkout"] = to_tz(df_cash["checkout"])
    mask_cash &= (df_cash["checkout"] >= start_day) & (df_cash["checkout"] < end_day)
    cash_pickups = df_cash[mask_cash] if isinstance(mask_cash, pd.Series) else pd.DataFrame(columns=normalized.columns)

    st.subheader("💵 Recolección de CASH (en checkout de hoy)")
    if cash_pickups.empty:
        st.success("No hay recolecciones de CASH para hoy.")
    else:
        show_cols = [c for c in ["apartment","guest_name","checkout_time","cash_amount","cash_reason"] if c in cash_pickups.columns]
        st.warning("Coordinar recolección en estos apartamentos:")
        st.dataframe(cash_pickups[show_cols], use_container_width=True)

    # 🚗 Estacionamientos activos HOY
    st.subheader("🚗 Estacionamientos activos hoy")
    active_pk = active_parkings_today(normalized, work_date, day_start_t, day_end_t)
    if active_pk.empty:
        st.info("No hay estacionamientos activos hoy.")
        apt_parking_count_map = {}
    else:
        view_pk = active_pk.rename(columns={
            "apartment":"Apartamento","guest_name":"Huésped",
            "parking_count":"Parking (xN)",
            "checkin_day":"Check-In (fecha)","checkin_time":"Check-In (hora)",
            "checkout_day":"Check-Out (fecha)","checkout_time":"Check-Out (hora)"
        })[["Apartamento","Huésped","Parking (xN)","Check-In (fecha)","Check-In (hora)","Check-Out (fecha)","Check-Out (hora)"]]
        st.dataframe(view_pk, use_container_width=True)
        apt_parking_count_map = active_pk.groupby("apartment")["parking_count"].sum().to_dict()

    # 🚐 Transportes de HOY (rango horario)
    st.subheader("🚐 Transportes de HOY (alerta rápida)")
    nm = normalized.copy()
    nm["checkin"]  = to_tz(nm["checkin"])
    nm["checkout"] = to_tz(nm["checkout"])

    def _prep_view_day(df_in, kind):
        if df_in.empty:
            return pd.DataFrame(columns=["Hora","Acción","Apartamento","Nombre","Teléfono","Notas","Fecha"])
        if kind == "CHECK-IN":
            df_in = df_in[(df_in["checkin"]>=start_day) & (df_in["checkin"]<end_day)].copy()
            df_in["Hora"] = df_in.apply(lambda r: _hora_from_cols(r,"ci"), axis=1)
            df_in["Fecha"] = df_in["checkin"].dt.strftime("%Y-%m-%d")
        else:
            df_in = df_in[(df_in["checkout"]>=start_day) & (df_in["checkout"]<end_day)].copy()
            df_in["Hora"] = df_in.apply(lambda r: _hora_from_cols(r,"co"), axis=1)
            df_in["Fecha"] = df_in["checkout"].dt.strftime("%Y-%m-%d")
        df_in["Nombre"]   = df_in.apply(_pick_name, axis=1)
        df_in["Teléfono"] = df_in.get("phone", pd.Series([""]*len(df_in))).apply(_clean_phone)
        nota_tag = "Check-In " if kind=="CHECK-IN" else "Check-Out "
        df_in["Notas"] = df_in.get("notes","").astype(str).fillna("") \
                         + np.where(df_in["Hora"].astype(str).str.strip()!="", " | "+nota_tag+df_in["Hora"].astype(str), "")
        out = df_in[["Hora","apartment","Nombre","Teléfono","Notas","Fecha"]].rename(columns={"apartment":"Apartamento"})
        out["Acción"] = kind
        return out

    t_ci_src = nm[nm.get("transport_ci", pd.Series([False]*len(nm))).apply(_is_si)] if "transport_ci" in nm.columns else nm.iloc[0:0]
    t_co_src = nm[nm.get("transport_co", pd.Series([False]*len(nm))).apply(_is_si)] if "transport_co" in nm.columns else nm.iloc[0:0]
    ci_day_view = _prep_view_day(t_ci_src, "CHECK-IN")
    co_day_view = _prep_view_day(t_co_src, "CHECK-OUT")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CI de hoy (con transporte)**")
        st.dataframe(ci_day_view.sort_values(["Hora","Apartamento"]) if not ci_day_view.empty else ci_day_view, use_container_width=True)
    with c2:
        st.markdown("**CO de hoy (con transporte)**")
        st.dataframe(co_day_view.sort_values(["Hora","Apartamento"]) if not co_day_view.empty else co_day_view, use_container_width=True)

    if not ci_day_view.empty or not co_day_view.empty:
        lines = [f"Traslados de hoy — {pd.Timestamp(work_date).strftime('%d/%m/%Y')}"]
        def _block_lines_day(df_in, tag):
            out = []
            for _, r in df_in.sort_values(["Hora","Apartamento"]).iterrows():
                out.append(f"• {r['Hora']} {tag} – {r['Apartamento']} – {r['Nombre']} – {r['Teléfono']} | {r['Notas']}")
            return out
        if not ci_day_view.empty: lines += ["", "CHECK-IN:"] + _block_lines_day(ci_day_view, "CHECK-IN")
        if not co_day_view.empty: lines += ["", "CHECK-OUT:"] + _block_lines_day(co_day_view, "CHECK-OUT")
        msg_hoy = "\n".join(lines)
        st.text_area("Mensaje (hoy)", msg_hoy, height=160)
        driver_phone = re.sub(r"\D", "", (wa_driver or ""))
        if driver_phone:
            st.markdown(f"[Enviar por WhatsApp al **Motorista**](https://wa.me/{driver_phone}?text={_q.quote(msg_hoy)})")
    else:
        st.info("No hay transportes requeridos para hoy.")

    # --- Resumen por apartamento
    st.subheader("🧾 Resumen por apartamento (día)")
    extra_apts = ["BALI"]
    summary_df = day_summary_collapsed_v2(normalized, work_date, day_start_t, day_end_t, extra_apts)
    st.dataframe(summary_df, use_container_width=True)

    # --- Conserje
    st.subheader("🧑‍🏫 Resumen para Conserje — Check-In de hoy (incluye Turnover)")
    conc_df = df_tmp[(df_tmp["checkin"]>=start_day) & (df_tmp["checkin"]<end_day)].copy()
    if not conc_df.empty and not summary_df.empty and "apartment" in conc_df.columns:
        conc_df = conc_df.merge(summary_df[["apartment","tipo"]], on="apartment", how="left")
        conc_df["tipo"] = conc_df["tipo"].fillna("Solo Check-In")
    else:
        conc_df["tipo"] = "Solo Check-In"

    def _est_text(apt):
        n = int(apt_parking_count_map.get(apt, 0))
        return "No" if n <= 0 else f"Sí (x{n})"
    if not conc_df.empty:
        conc_df["Estacionamiento"] = conc_df["apartment"].apply(_est_text)

    keep_cols = [c for c in ["apartment", "guest_name", "checkin_time", "tipo", "Estacionamiento"] if c in conc_df.columns]
    conc_view = conc_df[keep_cols].sort_values(["checkin_time","apartment"]) if not conc_df.empty else conc_df
    if conc_view.empty:
        st.info("No hay check-ins para hoy.")
    else:
        st.dataframe(conc_view.rename(columns={
            "apartment":"Apartamento", "guest_name":"Huésped", "checkin_time":"Hora Check-In"
        }), use_container_width=True)
        fecha_str = pd.Timestamp(work_date).strftime("%d/%m/%Y")
        lines = [f"Entradas de hoy – {fecha_str}:"]
        for _, r in conc_view.iterrows():
            hora = str(r.get("checkin_time","")).strip()
            apt  = str(r.get("apartment","")).strip()
            g    = str(r.get("guest_name","")).strip()
            tipo = str(r.get("tipo","")).strip()
            estc = str(r.get("Estacionamiento","")).strip()
            suf  = " (Turnover)" if tipo.lower().startswith("turnover") else ""
            extra_est = f" | Estacionamiento: {estc}"
            lines.append(f"• {hora} – {apt} – {g}{suf}{extra_est}")
        concierge_msg = "\n".join(lines) + "\n\nGracias."
        st.text_area("Mensaje para Conserje", concierge_msg, height=160)
        phone_digits = re.sub(r"\D", "", (wa_concierge or ""))
        if phone_digits:
            st.markdown(f"[Enviar por WhatsApp al **Conserje**](https://wa.me/{phone_digits}?text={_q.quote(concierge_msg)})")

    # --- Asignación manual (dos empleadas)
    st.markdown("**Asigna quién y a qué hora limpiar (manual)**")
    win_df = build_apartment_windows_v2(normalized, work_date, day_start_t, day_end_t, extra_apts=["BALI"])
    win_map = {r["apartment"]: r for _, r in win_df.iterrows()}

    def _fmt(ts):
        try: return ts.tz_convert(TZ_NAME).strftime("%H:%M")
        except Exception: return ""

    base_rows = []
    base_summary = day_summary_collapsed_v2(normalized, work_date, day_start_t, day_end_t, extra_apts=["BALI"])
    for _, r in base_summary.iterrows():
        apt = r["apartment"]; w = win_map.get(apt)
        v_ini_str = _fmt(w["start"]) if w is not None else ""
        v_fin_str = _fmt(w["end"]) if w is not None else ""
        pcount = int(apt_parking_count_map.get(apt, 0))
        base_rows.append({
            "apartment": apt, "tipo": r["tipo"],
            "ventana_inicio": v_ini_str, "ventana_fin": v_fin_str,
            "parking_activo": pcount,
            "empleado1": "—", "empleado2": "—",
            "inicio": "", "fin": ""
        })
    base_df = pd.DataFrame(base_rows)

    def _time_opts(t0: time, t1: time, step_min=15):
        cur = datetime.combine(datetime.today().date(), t0)
        end = datetime.combine(datetime.today().date(), t1)
        out = []
        while cur <= end:
            out.append(cur.strftime("%H:%M"))
            cur += timedelta(minutes=step_min)
        return out
    time_options = _time_opts(day_start_t, day_end_t, 15)

    edited = st.data_editor(
        base_df,
        column_config={
            "apartment": st.column_config.Column("apartment", disabled=True),
            "tipo": st.column_config.Column("tipo", disabled=True),
            "ventana_inicio": st.column_config.Column("ventana_inicio", disabled=True),
            "ventana_fin": st.column_config.Column("ventana_fin", disabled=True),
            "parking_activo": st.column_config.Column("parking_activo", disabled=True, help="Suma de parkings activos hoy"),
            "empleado1": st.column_config.SelectboxColumn("empleado1", options=[ "—", emp1_name, emp2_name ]),
            "empleado2": st.column_config.SelectboxColumn("empleado2", options=[ "—", emp1_name, emp2_name ]),
            "inicio": st.column_config.SelectboxColumn("inicio", options=time_options),
            "fin": st.column_config.SelectboxColumn("fin", options=time_options),
        },
        use_container_width=True, hide_index=True, num_rows="fixed"
    )

    def _to_dt(day, hhmm):
        if not hhmm or str(hhmm).strip() == "": return pd.NaT
        t = datetime.strptime(hhmm, "%H:%M").time()
        return TZ.localize(datetime.combine(pd.Timestamp(day).date(), t))

    manual_rows = []
    for _, r in edited.iterrows():
        s = _to_dt(work_date, r.get("inicio","")); e = _to_dt(work_date, r.get("fin",""))
        if pd.notna(s) and pd.notna(e) and e > s:
            seleccion = [r.get("empleado1","—"), r.get("empleado2","—")]
            seleccion = [x for x in seleccion if x != "—"]
            seleccion = list(dict.fromkeys(seleccion))
            for emp in seleccion:
                manual_rows.append({"employee": emp, "apartment": r["apartment"], "start": s, "end": e,
                                    "duration_min": int((e - s).total_seconds()//60)})
    manual_plan = pd.DataFrame(manual_rows)

    st.subheader("🗓️ Gantt manual (según tus horas)")
    if manual_plan.empty: st.info("Asigna empleado(s) e intervalos para ver el Gantt.")
    else: plot_gantt(manual_plan, title="Plan Manual de Limpieza (Gantt)", y_key="employee")

    # --- 📲 WhatsApp para Empleadas (basado en la asignación manual)
    st.subheader("📲 Mensajes de WhatsApp para empleadas")
    if manual_plan.empty:
        st.info("Asigna al menos una limpieza para generar los mensajes.")
    else:
        tipo_map = {r["apartment"]: r["tipo"] for _, r in base_summary.iterrows()}
        cash_map = {}
        if not cash_pickups.empty and 'apartment' in cash_pickups.columns:
            tmp_cash = cash_pickups.copy()
            tmp_cash["co_hora"] = tmp_cash.get("checkout_time", "")
            for _, rr in tmp_cash.iterrows():
                key = rr.get("apartment", "")
                cash_map.setdefault(key, [])
                cash_map[key].append({"amount": rr.get("cash_amount", np.nan), "hora": rr.get("co_hora", "")})

        def cash_str(a):
            lst = cash_map.get(a, [])
            if not lst: return ""
            c = lst[0]
            try:
                amt = float(c["amount"]) if not pd.isna(c["amount"]) else None
            except Exception:
                amt = None
            return f" — CASH: ${amt:,.2f} (checkout {c['hora']})" if amt is not None else ""

        def fmt(dt):
            try: return dt.tz_convert(TZ_NAME).strftime("%H:%M")
            except Exception:
                try: return pd.to_datetime(dt).strftime("%H:%M")
                except: return ""

        lunch = {
            emp1_name: (e1_l1.strftime("%H:%M"), e1_l2.strftime("%H:%M")),
            emp2_name: (e2_l1.strftime("%H:%M"), e2_l2.strftime("%H:%M")),
        }

        for emp in sorted(manual_plan["employee"].dropna().unique()):
            emp_tasks = manual_plan[manual_plan["employee"] == emp].sort_values("start")
            if emp_tasks.empty: continue
            lines = [f"Hola {emp} 👋, este es tu plan de hoy {pd.Timestamp(work_date).strftime('%d/%m/%Y')}:"]
            for _, t in emp_tasks.iterrows():
                apt  = t["apartment"]
                tipo = tipo_map.get(apt, "")
                pk   = int(apt_parking_count_map.get(apt, 0))
                park = "" if pk <= 0 else f" — Parking: x{pk}"
                lines.append(f"• {fmt(t['start'])}–{fmt(t['end'])} — {apt} ({tipo}){park}{cash_str(apt)}")
            l1, l2 = lunch.get(emp, ("",""))
            if l1 and l2:
                lines += ["", f"⏱️ Almuerzo sugerido: {l1}–{l2}"]
            msg = "\n".join(lines)
            st.text_area(f"Mensaje para {emp}", msg, height=180, key=f"wa_emp_{_norm(emp)}")
            phone = re.sub(r"\D", "", wa_emp1 if emp == emp1_name else (wa_emp2 if emp == emp2_name else ""))
            if phone:
                st.markdown(f"[Enviar a {emp} por WhatsApp](https://wa.me/{phone}?text={_q.quote(msg)})")

    # ==== Descargas (Excel/PDF) ====
    st.subheader("📥 Descargar planificación del día (Excel / PDF)")
    def _build_plan_export():
        if manual_plan.empty: return pd.DataFrame()
        df = manual_plan.copy()
        df = df.merge(base_summary[["apartment","tipo"]], on="apartment", how="left")
        df["Parking"] = df["apartment"].map(lambda a: ("Sí (x"+str(int(apt_parking_count_map.get(a,0)))+")") if apt_parking_count_map.get(a,0)>0 else "No")
        cash_map_local = {}
        if not cash_pickups.empty and 'apartment' in cash_pickups.columns:
            tmp_cash = cash_pickups.copy()
            tmp_cash["co_hora"] = tmp_cash.get("checkout_time", "")
            for _, rr in tmp_cash.iterrows():
                key = rr.get("apartment", "")
                cash_map_local.setdefault(key, [])
                cash_map_local[key].append({"amount": rr.get("cash_amount", np.nan), "hora": rr.get("co_hora", "")})
        def _cash_str(a):
            lst = cash_map_local.get(a, [])
            if not lst: return ""
            c = lst[0]
            try: amt = float(c["amount"]) if not pd.isna(c["amount"]) else None
            except Exception: amt = None
            if amt is None: return ""
            return f"${amt:,.2f}" + (f" (checkout {c['hora']})" if str(c['hora']).strip() else "")
        df["CASH"] = df["apartment"].map(_cash_str)
        def _f(dt):
            try: return dt.tz_convert(TZ_NAME).strftime("%H:%M")
            except Exception:
                try: return pd.to_datetime(dt).strftime("%H:%M")
                except: return ""
        df["Inicio"] = df["start"].apply(_f); df["Fin"] = df["end"].apply(_f)
        out_cols = ["employee","apartment","tipo","Inicio","Fin","duration_min","Parking","CASH"]
        df = df[out_cols].rename(columns={"employee":"Empleado","apartment":"Apartamento","tipo":"Tipo","duration_min":"Duración (min)"})
        return df.sort_values(["Inicio","Empleado","Apartamento"])
    plan_export = _build_plan_export()

    if plan_export.empty:
        st.info("Completa la tabla de asignación manual para habilitar descargas.")
    else:
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer) as writer:
                plan_export.to_excel(writer, index=False, sheet_name="Plan")
                if not win_df.empty: win_df.to_excel(writer, index=False, sheet_name="Ventanas")
                if not prep_df.empty: prep_df.to_excel(writer, index=False, sheet_name="Preparación")
                if not cash_pickups.empty: cash_pickups.to_excel(writer, index=False, sheet_name="CASH")
                if not active_pk.empty: active_pk.to_excel(writer, index=False, sheet_name="ParkingActivo")
            st.download_button("⬇️ Descargar Excel (plan del día)",
                               data=excel_buffer.getvalue(),
                               file_name=f"plan_dia_{pd.Timestamp(work_date).strftime('%Y-%m-%d')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as ex:
            st.error(f"No se pudo generar el Excel: {ex}")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            W, H = A4; margin = 1.5*cm; y = H - margin
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, y, f"Plan de Limpieza – {pd.Timestamp(work_date).strftime('%d/%m/%Y')}"); y -= 0.8*cm
            c.setFont("Helvetica", 10)
            c.drawString(margin, y, "Hora     Empleado       Apartamento                 Tipo                 Parking   CASH"); y -= 0.4*cm
            c.line(margin, y, W - margin, y); y -= 0.3*cm
            for _, r in plan_export.iterrows():
                line = f"{r['Inicio']}-{r['Fin']:<5}  {str(r['Empleado'])[:12]:<12}  {str(r['Apartamento'])[:24]:<24}  {str(r['Tipo'])[:18]:<18}  {str(r['Parking'])[:9]:<9}  {str(r['CASH'])[:18]}"
                if y < margin + 2*cm:
                    c.showPage(); y = H - margin; c.setFont("Helvetica", 10)
                c.drawString(margin, y, line); y -= 0.35*cm
            y -= 0.5*cm; c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Estacionamientos activos hoy"); y -= 0.5*cm
            c.setFont("Helvetica", 10)
            if active_pk.empty:
                c.drawString(margin, y, "— Sin registros —"); y -= 0.4*cm
            else:
                for _, r in active_pk.iterrows():
                    t = f"{r['apartment']}: {r['guest_name']}  (x{int(r['parking_count'])})  CI {r['checkin_day']} {r['checkin_time']}  /  CO {r['checkout_day']} {r['checkout_time']}"
                    if y < margin + 2*cm:
                        c.showPage(); y = H - margin; c.setFont("Helvetica", 10)
                    c.drawString(margin, y, t); y -= 0.35*cm
            c.showPage(); c.save()
            st.download_button("⬇️ Descargar PDF (plan del día)",
                               data=pdf_buffer.getvalue(),
                               file_name=f"plan_dia_{pd.Timestamp(work_date).strftime('%Y-%m-%d')}.pdf",
                               mime="application/pdf")
        except Exception:
            st.warning("Para generar el PDF, agrega **reportlab** a tu `requirements.txt`.")

    # --- Ventanas por apartamento (gráfico)
    st.subheader("🏠 Ventanas disponibles por apartamento (info)")
    windows_df = build_apartment_windows_v2(normalized, work_date, day_start_t, day_end_t, extra_apts=["BALI"])
    if windows_df.empty: st.info("No hay ventanas disponibles para el día seleccionado.")
    else:
        if plt is None: st.error("Instala `matplotlib` para ver el gráfico.")
        else:
            fig, ax = plt.subplots(figsize=(11, 3 + 0.35*len(windows_df)))
            apartments = list(windows_df["apartment"].unique()); y_map = {a:i for i,a in enumerate(apartments)}
            day0 = windows_df["start"].min().replace(hour=0, minute=0, second=0, microsecond=0)
            for _, row in windows_df.iterrows():
                yv = y_map[row["apartment"]]
                left = (row["start"] - day0).total_seconds()/3600
                width = (row["end"] - row["start"]).total_seconds()/3600
                ax.barh(yv, width, left=left); ax.text(left+width/2, yv, row["kind"], va="center", ha="center", fontsize=9)
            ax.set_yticks(range(len(apartments))); ax.set_yticklabels(apartments)
            ax.set_xlabel("Horas del día"); ax.set_title("Ventanas libres para limpieza")
            st.pyplot(fig)

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
        adr_global = float(apt_group["revenue"].sum())/float(max(1, apt_group['nights'].sum()))
        c3.metric("ADR", f"${adr_global:,.2f}")
        c4.metric("Check-ins", f"{int(apt_group['checkins'].sum())}")
        c5.metric("Check-outs", f"{int(apt_group['checkouts'].sum())}")
        c6.metric("Cash a recoger", f"${float(np.nansum(apt_group['cash_amount'])):,.2f} ({int(apt_group['cash_pickups'].sum())})")

        st.markdown("**Tabla por apartamento**")
        show_cols = ["apartment","nights","revenue","ADR","occupancy_%","checkins","checkouts","cash_pickups","cash_amount"]
        st.dataframe(apt_group[show_cols].sort_values("revenue", ascending=False), use_container_width=True)

        st.markdown("**Ingresos por apartamento**")
        if plt is None:
            st.error("Para ver el gráfico instala matplotlib (añade `matplotlib` a requirements.txt).")
        else:
            y = apt_group["apartment"].fillna("—").astype(str)
            x = pd.to_numeric(apt_group["revenue"], errors="coerce").fillna(0.0)
            fig, ax = plt.subplots(figsize=(11, max(3, 0.3*len(apt_group))))
            ax.barh(y, x)
            ax.set_xlabel("USD"); ax.set_ylabel("Apartamento"); ax.set_title("Ingresos del mes (prorrateados)")
            st.pyplot(fig)

        st.markdown("**Noches por apartamento**")
        if plt is None:
            st.error("Para ver el gráfico instala matplotlib (añade `matplotlib` a requirements.txt).")
        else:
            y2 = apt_group["apartment"].fillna("—").astype(str)
            x2 = pd.to_numeric(apt_group["nights"], errors="coerce").fillna(0)
            fig2, ax2 = plt.subplots(figsize=(11, max(3, 0.3*len(apt_group))))
            ax2.barh(y2, x2)
            ax2.set_xlabel("Noches"); ax2.set_ylabel("Apartamento"); ax2.set_title("Noches del mes")
            st.pyplot(fig2)

        ch_counts = month_calc.copy()
        ch_counts["has_night"] = ch_counts["overlap_nights"] > 0
        ch_counts = ch_counts[ch_counts["has_night"]]
        if not ch_counts.empty and "channel" in ch_counts.columns:
            mix = ch_counts.groupby("channel").size().reset_index(name="reservas")
            st.markdown("**Mix de canal (reservas con noches en el mes)**")
            st.dataframe(mix.sort_values("reservas", ascending=False), use_container_width=True)

        def _avg_time(series_dt):
            s = series_dt.dropna()
            if s.empty: return None
            mins = s.dt.hour*60 + s.dt.minute
            m = int(mins.mean())
            return f"{m//60:02d}:{m%60:02d}"
        avg_ci = _avg_time(month_calc.loc[month_calc["checkin_in_month"]==1, "checkin_time_dt"])
        avg_co = _avg_time(month_calc.loc[month_calc["checkout_in_month"]==1, "checkout_time_dt"])
        st.markdown(f"**Hora promedio** — Check-in: `{avg_ci or '—'}` • Check-out: `{avg_co or '—'}`")

# ====== TAB 3: Transportes (MES COMPLETO + HOY)
with tab3:
    st.header("🚐 Resumen de transportes (MES)")

    mes_transportes = st.date_input("Mes a analizar (transportes)", value=work_date, key="t_month_all")
    month_start_t = pd.Timestamp(mes_transportes).replace(day=1).tz_localize(TZ)
    month_end_t   = month_start_t + pd.offsets.MonthBegin(1)

    norm_m = normalized.copy()
    norm_m["checkin"]  = to_tz(norm_m["checkin"])
    norm_m["checkout"] = to_tz(norm_m["checkout"])

    has_ci = "transport_ci" in norm_m.columns
    has_co = "transport_co" in norm_m.columns

    if has_ci:
        t_ci_m = norm_m[
            (norm_m["checkin"]>=month_start_t) & (norm_m["checkin"]<month_end_t) & (norm_m["transport_ci"].apply(_is_si))
        ].copy()
    else:
        t_ci_m = pd.DataFrame(columns=norm_m.columns)

    if has_co:
        t_co_m = norm_m[
            (norm_m["checkout"]>=month_start_t) & (norm_m["checkout"]<month_end_t) & (norm_m["transport_co"].apply(_is_si))
        ].copy()
    else:
        t_co_m = pd.DataFrame(columns=norm_m.columns)

    def _prep_view_month(df_in, kind):
        if df_in.empty:
            return pd.DataFrame(columns=["Fecha","Hora","Acción","Apartamento","Nombre","Teléfono","Notas"])
        if kind == "CHECK-IN":
            df_in["Fecha"] = df_in["checkin"].dt.strftime("%Y-%m-%d")
            df_in["Hora"]  = df_in.apply(lambda r: _hora_from_cols(r,"ci"), axis=1)
        else:
            df_in["Fecha"] = df_in["checkout"].dt.strftime("%Y-%m-%d")
            df_in["Hora"]  = df_in.apply(lambda r: _hora_from_cols(r,"co"), axis=1)
        df_in["Nombre"]   = df_in.apply(_pick_name, axis=1)
        df_in["Teléfono"] = df_in.get("phone", pd.Series([""]*len(df_in))).apply(_clean_phone)
        nota_tag = "Check-In " if kind=="CHECK-IN" else "Check-Out "
        df_in["Notas"] = df_in.get("notes","").astype(str).fillna("") \
                         + np.where(df_in["Hora"].astype(str).str.strip()!="", " | "+nota_tag+df_in["Hora"].astype(str), "")
        out = df_in[["Fecha","Hora","apartment","Nombre","Teléfono","Notas"]].rename(columns={"apartment":"Apartamento"})
        out["Acción"] = kind
        return out

    ci_month_view = _prep_view_month(t_ci_m, "CHECK-IN")
    co_month_view = _prep_view_month(t_co_m, "CHECK-OUT")
    month_view = pd.concat([ci_month_view, co_month_view], ignore_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total CHECK-IN (mes)", len(ci_month_view))
    c2.metric("Total CHECK-OUT (mes)", len(co_month_view))
    c3.metric("Total traslados (mes)", len(month_view))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Transportes CHECK-IN (mes)**")
        st.dataframe(ci_month_view.sort_values(["Fecha","Hora","Apartamento"]) if not ci_month_view.empty else ci_month_view,
                     use_container_width=True)
    with colB:
        st.markdown("**Transportes CHECK-OUT (mes)**")
        st.dataframe(co_month_view.sort_values(["Fecha","Hora","Apartamento"]) if not co_month_view.empty else co_month_view,
                     use_container_width=True)

    st.subheader("📝 Mensaje general para Motorista (mes)")
    meses_es = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    titulo_mes = f"{meses_es[month_start_t.month-1].capitalize()} {month_start_t.year}"
    lines = [f"Traslados del mes — {titulo_mes}"]
    def _block_lines(df_in, tag):
        out = []
        for _, r in df_in.sort_values(["Fecha","Hora","Apartamento"]).iterrows():
            out.append(f"• {r['Fecha']} {r['Hora']} {tag} – {r['Apartamento']} – {r['Nombre']} – {r['Teléfono']} | {r['Notas']}")
        return out
    ci_lines = _block_lines(ci_month_view, "CHECK-IN")
    co_lines = _block_lines(co_month_view, "CHECK-OUT")
    if ci_lines: lines += ["", "CHECK-IN:"] + ci_lines
    if co_lines: lines += ["", "CHECK-OUT:"] + co_lines
    msg_mes = "\n".join(lines) if (ci_lines or co_lines) else "Sin traslados para este mes."
    st.text_area("Mensaje (mes)", msg_mes, height=250)
    driver_phone = re.sub(r"\D", "", (wa_driver or ""))
    if driver_phone and (ci_lines or co_lines):
        st.markdown(f"[Enviar por WhatsApp al **Motorista**](https://wa.me/{driver_phone}?text={_q.quote(msg_mes)})")

    st.subheader("✉️ Mensajes individuales (WhatsApp) — Todo el mes")
    driver_display = st.text_input("Nombre del motorista para el saludo", value="Balentina", key="t_driver_name")

    def _row_msg_month(r: pd.Series) -> str:
        nombre = r.get("Nombre","").strip()
        fecha  = _fecha_es_larga(r.get("Fecha",""))
        hora   = _hora_12h(r.get("Hora",""))
        vuelo  = _extract_flight(r.get("Notas",""))
        if r.get("Acción") == "CHECK-IN":
            ruta = f"Desde Tocumen hacia {r.get('Apartamento','')}"
        else:
            ruta = f"Desde {r.get('Apartamento','')} hacia Tocumen"
        saludo = f"Hola {driver_display} te comparto los datos" if driver_display else "Te comparto los datos"
        msg = f"{saludo}\n\n{nombre}\n{fecha}\n{hora}"
        if vuelo:
            msg += f"\n{vuelo}"
        msg += f"\n{ruta}"
        return msg

    cci, cco = st.columns(2)
    with cci:
        st.markdown("**Mensajes – CHECK-IN (mes)**")
        if ci_month_view.empty:
            st.info("Sin transportes de check-in este mes.")
        else:
            for i, r in ci_month_view.iterrows():
                txt = _row_msg_month(r)
                st.text_area(f"CI • {r['Fecha']} {r['Hora']} – {r['Apartamento']} – {r['Nombre']}", txt,
                             height=120, key=f"ci_msg_m_{i}")
                if driver_phone:
                    st.markdown(f"[Enviar a Motorista](https://wa.me/{driver_phone}?text={_q.quote(txt)})")
    with cco:
        st.markdown("**Mensajes – CHECK-OUT (mes)**")
        if co_month_view.empty:
            st.info("Sin transportes de check-out este mes.")
        else:
            for j, r in co_month_view.iterrows():
                txt = _row_msg_month(r)
                st.text_area(f"CO • {r['Fecha']} {r['Hora']} – {r['Apartamento']} – {r['Nombre']}", txt,
                             height=120, key=f"co_msg_m_{j}")
                if driver_phone:
                    st.markdown(f"[Enviar a Motorista](https://wa.me/{driver_phone}?text={_q.quote(txt)})")

    st.download_button("⬇️ Descargar CSV mensual (transportes)",
                       data=month_view.sort_values(["Fecha","Hora","Acción","Apartamento"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"transportes_mes_{month_start_t.strftime('%Y-%m')}.csv",
                       mime="text/csv")

    st.markdown("---")
    with st.expander("📅 Transportes de HOY (detalle)"):
        if has_ci:
            t_ci_day = norm_m[(norm_m["checkin"]>=_day_bounds(work_date, day_start_t, day_end_t)[0]) &
                              (norm_m["checkin"]<_day_bounds(work_date, day_start_t, day_end_t)[1]) &
                              (norm_m["transport_ci"].apply(_is_si))].copy()
        else:
            t_ci_day = pd.DataFrame(columns=norm_m.columns)
        if has_co:
            t_co_day = norm_m[(norm_m["checkout"]>=_day_bounds(work_date, day_start_t, day_end_t)[0]) &
                              (norm_m["checkout"]<_day_bounds(work_date, day_start_t, day_end_t)[1]) &
                              (norm_m["transport_co"].apply(_is_si))].copy()
        else:
            t_co_day = pd.DataFrame(columns=norm_m.columns)

        def _prep_view_day(df_in, kind):
            if df_in.empty:
                return pd.DataFrame(columns=["Hora","Acción","Apartamento","Nombre","Teléfono","Notas"])
            if kind == "CHECK-IN":
                df_in["Hora"]  = df_in.apply(lambda r: _hora_from_cols(r,"ci"), axis=1)
            else:
                df_in["Hora"]  = df_in.apply(lambda r: _hora_from_cols(r,"co"), axis=1)
            df_in["Nombre"]   = df_in.apply(_pick_name, axis=1)
            df_in["Teléfono"] = df_in.get("phone", pd.Series([""]*len(df_in))).apply(_clean_phone)
            nota_tag = "Check-In " if kind=="CHECK-IN" else "Check-Out "
            df_in["Notas"] = df_in.get("notes","").astype(str).fillna("") \
                             + np.where(df_in["Hora"].astype(str).str.strip()!="", " | "+nota_tag+df_in["Hora"].astype(str), "")
            out = df_in[["Hora","apartment","Nombre","Teléfono","Notas"]].rename(columns={"apartment":"Apartamento"})
            out["Acción"] = kind
            return out

        ci_day_view = _prep_view_day(t_ci_day, "CHECK-IN")
        co_day_view = _prep_view_day(t_co_day, "CHECK-OUT")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CI de hoy (con transporte)**")
            st.dataframe(ci_day_view.sort_values(["Hora","Apartamento"]) if not ci_day_view.empty else ci_day_view, use_container_width=True)
        with col2:
            st.markdown("**CO de hoy (con transporte)**")
            st.dataframe(co_day_view.sort_values(["Hora","Apartamento"]) if not co_day_view.empty else co_day_view, use_container_width=True)

        if not ci_day_view.empty or not co_day_view.empty:
            lines = [f"Traslados de hoy — {pd.Timestamp(work_date).strftime('%d/%m/%Y')}"]
            def _block_lines_day(df_in, tag):
                out = []
                for _, r in df_in.sort_values(["Hora","Apartamento"]).iterrows():
                    out.append(f"• {r['Hora']} {tag} – {r['Apartamento']} – {r['Nombre']} – {r['Teléfono']} | {r['Notas']}")
                return out
            lines += (["", "CHECK-IN:"] + _block_lines_day(ci_day_view, "CHECK-IN")) if not ci_day_view.empty else []
            lines += (["", "CHECK-OUT:"] + _block_lines_day(co_day_view, "CHECK-OUT")) if not co_day_view.empty else []
            msg_hoy = "\n".join(lines)
            st.text_area("Mensaje (hoy)", msg_hoy, height=160, key="msg_driver_hoy_t3")
            driver_phone = re.sub(r"\D", "", (wa_driver or ""))
            if driver_phone:
                st.markdown(f"[Enviar por WhatsApp al **Motorista**](https://wa.me/{driver_phone}?text={_q.quote(msg_hoy)})")
        else:
            st.info("No hay transportes requeridos para hoy.")
