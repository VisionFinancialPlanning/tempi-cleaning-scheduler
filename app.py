# 🧑‍🏫 Resumen para Conserje — Check-In de hoy (incluye Turnover)
st.subheader("🧑‍🏫 Resumen para Conserje — Check-In de hoy (incluye Turnover)")

# Todos los check-ins del día (sin filtrar por 'Solo Check-In')
conc_df = normalized[normalized["checkin_day"] == today_str].copy()

# Anexar el tipo desde el resumen (Turnover / Solo Check-In / etc.)
if not conc_df.empty and not summary_df.empty and "apartment" in conc_df.columns:
    conc_df = conc_df.merge(
        summary_df[["apartment","tipo"]],
        on="apartment",
        how="left"
    )
    conc_df["tipo"] = conc_df["tipo"].fillna("Solo Check-In")
else:
    conc_df["tipo"] = "Solo Check-In"

# Vista para conserje
keep_cols = [c for c in ["apartment", "guest_name", "checkin_time", "tipo"] if c in conc_df.columns]
conc_view = conc_df[keep_cols].sort_values(["checkin_time","apartment"]) if not conc_df.empty else conc_df

if conc_view.empty:
    st.info("No hay check-ins para hoy.")
else:
    st.dataframe(conc_view, use_container_width=True)

    # Mensaje de WhatsApp listo para enviar (marca Turnover cuando aplique)
    fecha_str = pd.Timestamp(work_date).strftime("%d/%m/%Y")
    lines = [f"Entradas de hoy – {fecha_str}:"]
    for _, r in conc_view.iterrows():
        hora = str(r.get("checkin_time","")).strip()
        apt  = str(r.get("apartment","")).strip()
        g    = str(r.get("guest_name","")).strip()
        tipo = str(r.get("tipo","")).strip()
        suf  = " (Turnover)" if tipo.lower().startswith("turnover") else ""
        lines.append(f"• {hora} – {apt} – {g}{suf}")
    concierge_msg = "\n".join(lines) + "\n\nGracias."

    st.text_area("Mensaje para Conserje", concierge_msg, height=160)

    # Link directo a WhatsApp si hay número en el sidebar
    phone_digits = re.sub(r"\D", "", wa_concierge or "")
    if phone_digits:
        wa_link = f"https://wa.me/{phone_digits}?text={_q.quote(concierge_msg)}"
        st.markdown(f"[Enviar por WhatsApp al **Conserje**]({wa_link})")

    # Descargar CSV
    csv_bytes = conc_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV para Conserje", data=csv_bytes,
                       file_name=f"checkins_todos_{today_str}.csv", mime="text/csv")
