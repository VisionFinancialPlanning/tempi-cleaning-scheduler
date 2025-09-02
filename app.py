# ====== TAB 3: Transportes ======
with tab3:
    st.header("🚐 Resumen de transportes (MES)")

    # Mes a analizar (mismo selector que ya usas en Analítica)
    mes_transportes = st.date_input("Mes a analizar (transportes)", value=work_date, key="t_month_all")
    month_start_t = pd.Timestamp(mes_transportes).replace(day=1).tz_localize(TZ)
    month_end_t   = month_start_t + pd.offsets.MonthBegin(1)

    # -------- Helpers específicos de transportes --------
    def _pick_name(row):
        g = str(row.get("guest_name","")).strip()
        return g if g else str(row.get("nombre","")).strip()

    def _clean_phone(val: str) -> str:
        s = re.sub(r"\D", "", str(val or ""))
        if len(s) == 8: s = "507" + s
        if s.startswith("00"): s = s[2:]
        return s

    def _hora_from_cols(row, kind):
        # Devuelve la hora como HH:MM (prioriza columna de hora; si no, la saca del datetime)
        if kind == "ci":
            if "checkin_time" in row and str(row["checkin_time"]).strip():
                return row["checkin_time"]
            if "checkin" in row and pd.notna(row["checkin"]):
                try: return row["checkin"].tz_convert(TZ_NAME).strftime("%H:%M")
                except Exception: return pd.to_datetime(row["checkin"]).strftime("%H:%M")
        else:
            if "checkout_time" in row and str(row["checkout_time"]).strip():
                return row["checkout_time"]
            if "checkout" in row and pd.notna(row["checkout"]):
                try: return row["checkout"].tz_convert(TZ_NAME).strftime("%H:%M")
                except Exception: return pd.to_datetime(row["checkout"]).strftime("%H:%M")
        return ""

    def _fecha_es_larga(iso_yyyy_mm_dd: str) -> str:
        try:
            d = datetime.strptime(iso_yyyy_mm_dd, "%Y-%m-%d")
        except Exception:
            return iso_yyyy_mm_dd
        meses = ["enero","febrero","marzo","abril","mayo","junio",
                 "julio","agosto","septiembre","octubre","noviembre","diciembre"]
        return f"{d.day} de {meses[d.month-1]}"

    def _hora_12h(hhmm: str) -> str:
        try:
            h, m = map(int, str(hhmm).split(":")[:2])
            suf = "am"
            if h == 0:
                h = 12; suf = "am"
            elif h == 12:
                suf = "pm"
            elif h > 12:
                h -= 12; suf = "pm"
            return f"{h}:{m:02d} {suf}"
        except Exception:
            return str(hhmm)

    def _extract_flight(notes: str) -> str:
        s = str(notes or "")
        m = re.search(r"(vuelo\s+[A-Za-z]{1,3}\s?\d{2,5})", s, flags=re.I)
        if m:
            t = m.group(1)
            return " ".join([w.upper() if re.fullmatch(r"[A-Za-z]{1,3}\s?\d{2,5}", w) else w for w in t.split()])
        m2 = re.search(r"\b([A-Za-z]{1,3}\s?\d{2,5})\b", s)
        return f"Vuelo {m2.group(1).upper()}" if m2 else ""

    # -------- Filtrado a TODO el MES --------
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
            df_in["Fecha"] = df_in["checkin_day"]
            df_in["Hora"]  = df_in.apply(lambda r: _hora_from_cols(r,"ci"), axis=1)
        else:
            df_in["Fecha"] = df_in["checkout_day"]
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

    # -------- Mensaje general para enviar (todo el mes) --------
    st.subheader("📝 Mensaje general para Motorista (mes)")
    titulo_mes = month_start_t.strftime("%B %Y").capitalize()
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
    driver_phone = re.sub(r"\D", "", wa_driver or "")
    if driver_phone and (ci_lines or co_lines):
        st.markdown(f"[Enviar por WhatsApp al **Motorista**](https://wa.me/{driver_phone}?text={_q.quote(msg_mes)})")

    # -------- Mensajes INDIVIDUALES por pasajero (todo el mes) --------
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

    # -------- Descarga CSV mensual --------
    st.download_button("⬇️ Descargar CSV mensual (transportes)",
                       data=month_view.sort_values(["Fecha","Hora","Acción","Apartamento"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"transportes_mes_{month_start_t.strftime('%Y-%m')}.csv",
                       mime="text/csv")
