"""
HAVI SENSE — Demo App
=====================
Motor de Inteligencia Personalizada para Hey Banco.
Demo interactiva para el pitch del datathon.

Cómo correrla:
    pip install streamlit pandas plotly
    streamlit run app_demo.py

Estructura de archivos esperada:
    ../data/df_clientes_kmeans.csv          (output del notebook clustering_kmeans)
    ../data/df_clientes_con_triggers.csv    (output del notebook cruce_triggers_hey)
    ../data/triggers_tabla.csv              (resumen de los 8 triggers)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(
    page_title="Havi Sense — Hey Banco",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta Hey Banco aproximada
COLOR_PRIMARY = "#00C896"   # verde Hey
COLOR_DARK = "#1D3557"
COLOR_RED = "#E63946"
COLOR_BG = "#F8F9FA"

st.markdown(f"""
    <style>
    .big-title {{ font-size: 32px; font-weight: 700; color: {COLOR_DARK}; }}
    .subtitle {{ font-size: 14px; color: #666; margin-bottom: 20px; }}
    .metric-card {{
        background: white !important; border-left: 4px solid {COLOR_PRIMARY};
        padding: 12px 16px; border-radius: 4px; margin-bottom: 8px;
        color: {COLOR_DARK} !important;
    }}
    .metric-card * {{ color: {COLOR_DARK} !important; }}
    .trigger-card {{
        background: white !important; border-left: 4px solid {COLOR_RED};
        padding: 16px; border-radius: 4px; margin-bottom: 12px;
        color: {COLOR_DARK} !important;
    }}
    .trigger-card * {{ color: {COLOR_DARK} !important; }}
    .message-box {{
        background: #E8F8F3 !important; border-radius: 8px; padding: 20px;
        border-left: 4px solid {COLOR_PRIMARY}; font-size: 16px;
        color: {COLOR_DARK} !important;
    }}
    .message-box * {{ color: {COLOR_DARK} !important; }}
    </style>
""", unsafe_allow_html=True)


# =====================================================================
# CARGA DE DATOS
# =====================================================================
@st.cache_data
def load_data():
    """Carga los CSVs. Cambia las rutas si los archivos están en otro lugar."""
    df = pd.read_csv("data/df_clientes_con_triggers.csv")

    # Si tienes el master con clusters K-Means, mergéalo
    try:
        kmeans = pd.read_csv("data/df_clientes_kmeans.csv")
        if 'cluster_label' in kmeans.columns:
            df = df.merge(
                kmeans[['user_id', 'cluster', 'cluster_label', 'pca_x', 'pca_y']],
                on='user_id', how='left'
            )
    except FileNotFoundError:
        st.sidebar.warning("⚠️ No se encontró df_clientes_kmeans.csv — algunas vistas no estarán disponibles.")

    triggers = pd.read_csv("data/triggers_tabla.csv")
    return df, triggers


df, triggers_df = load_data()

# =====================================================================
# SIDEBAR — NAVEGACIÓN
# =====================================================================
st.sidebar.markdown(f'<div class="big-title">💚 Havi Sense</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="subtitle">Motor de Inteligencia Personalizada</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

vista = st.sidebar.radio(
    "Vista",
    ["🧑 Vista Cliente", "📊 Vista Negocio", "💬 Mensaje Sugerido"],
    label_visibility='collapsed',
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Base activa**: {len(df):,} clientes")
if 'cubierto_por_trigger' in df.columns:
    cobertura = df['cubierto_por_trigger'].sum()
    st.sidebar.markdown(f"**Activables hoy**: {cobertura:,} ({100*cobertura/len(df):.1f}%)")

st.sidebar.markdown("---")
st.sidebar.caption("Hey Banco · Datathon 2026")


# =====================================================================
# VISTA 1: CLIENTE
# =====================================================================
if vista == "🧑 Vista Cliente":
    st.markdown('<div class="big-title">🧑 Vista Cliente</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Perfil 360° de un cliente individual con next-best-action sugerida</div>', unsafe_allow_html=True)

    # === Selector de cliente ===
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        # Por defecto ofrecer 3 casos representativos
        casos_demo = []
        for tid in ['T1_lead_credito_auto', 'T7_intercepta_cancelacion', 'T4_solicita_escalacion']:
            sub = df[df.get('next_best_action') == tid]
            if len(sub) > 0:
                casos_demo.append(sub.iloc[0]['user_id'])

        opciones = casos_demo + sorted(df['user_id'].unique().tolist())
        user_sel = st.selectbox(
            "Selecciona un cliente",
            opciones,
            help="Los primeros 3 son casos representativos para la demo del pitch"
        )

    u = df[df['user_id'] == user_sel].iloc[0]

    # === Encabezado del cliente ===
    st.markdown("---")
    col_h1, col_h2, col_h3 = st.columns([2, 2, 1])
    with col_h1:
        st.markdown(f"### {user_sel}")
        st.caption(f"{u['sexo']}, {u['edad']:.0f} años · {u['ciudad']}, {u['estado']}")
        st.caption(f"{u['ocupacion']} · {u['nivel_educativo']}")

    with col_h2:
        if u['es_hey_pro']:
            st.success("✓ Hey Pro activo")
        else:
            st.info("Hey Pro: no activo")
        st.caption(f"Antigüedad: {u['antiguedad_dias']/365:.1f} años · NPS: {u['satisfaccion_1_10']:.0f}/10")

    with col_h3:
        if 'cluster_label' in u and pd.notna(u.get('cluster_label')):
            st.metric("Segmento", u['cluster_label'])
        else:
            st.metric("Categoría", u.get('categoria', '—'))

    # === Métricas comportamentales ===
    st.markdown("### 📊 Comportamiento")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ingreso mensual", f"${u['ingreso_mensual_mxn']:,.0f}")
    m2.metric("Saldo actual", f"${u['saldo']:,.0f}",
              delta="negativo" if u['saldo'] < 0 else "positivo",
              delta_color="inverse" if u['saldo'] < 0 else "normal")
    m3.metric("Transacciones", f"{u['num_transacciones']:.0f}")
    m4.metric("No procesadas", f"{u['num_no_procesadas']:.0f}",
              delta=f"de {u['num_transacciones']:.0f}" if u['num_no_procesadas'] > 0 else None,
              delta_color="inverse")

    # === Voz del cliente ===
    st.markdown("### 💬 Voz del cliente (Havi)")
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        if pd.notna(u.get('cluster_label_dominante')):
            st.markdown(f'<div class="metric-card"><b>Tópico dominante:</b> {u["cluster_label_dominante"]}</div>',
                        unsafe_allow_html=True)
    with col_v2:
        st.metric("Tópicos distintos", int(u.get('num_clusters_distintos', 1)))

    # === Trigger activo + acción sugerida ===
    st.markdown("### 🎯 Next Best Action")
    if pd.notna(u.get('next_best_action')):
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.markdown(f"""
            <div class='trigger-card'>
            <b>{u['next_best_action_label']}</b><br>
            <small>Tipo: <i>{u['next_best_action_tipo']}</i></small>
            </div>
            """, unsafe_allow_html=True)

            n_aplicables = int(u.get('n_triggers_aplicables', 1))
            st.caption(f"📌 Este cliente tiene **{n_aplicables}** trigger(s) aplicable(s).")

        with col_t2:
            st.markdown(f"""
            <div class='message-box'>
            <b>📱 Mensaje sugerido:</b><br><br>
            {u['next_best_action_message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Cliente saludable — sin acción proactiva requerida en este momento.")


# =====================================================================
# VISTA 2: NEGOCIO
# =====================================================================
elif vista == "📊 Vista Negocio":
    st.markdown('<div class="big-title">📊 Vista Negocio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dashboard ejecutivo: oportunidades activables y impacto estimado</div>', unsafe_allow_html=True)

    # === KPIs ===
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Base de clientes", f"{len(df):,}")
    if 'cubierto_por_trigger' in df.columns:
        n_cob = df['cubierto_por_trigger'].sum()
        col2.metric("Activables hoy", f"{n_cob:,}", delta=f"{100*n_cob/len(df):.1f}% de la base")
    if 'impacto_estimado_mxn' in triggers_df.columns:
        impacto = triggers_df['impacto_estimado_mxn'].sum()
        col3.metric("Impacto anual estimado", f"${impacto/1e6:.1f}M MXN")
    col4.metric("Triggers activos", f"{len(triggers_df)}")

    st.markdown("---")

    # === Mapa de segmentos ===
    if 'pca_x' in df.columns and 'cluster_label' in df.columns:
        st.markdown("### 🗺️ Mapa de segmentos")
        fig = px.scatter(
            df.sample(min(5000, len(df))),  # sample para velocidad de render
            x='pca_x', y='pca_y',
            color='cluster_label',
            opacity=0.6,
            title=f'Segmentación K-Means — proyección PCA 2D',
            labels={'pca_x': 'PC1', 'pca_y': 'PC2', 'cluster_label': 'Segmento'},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # === Tabla de triggers ===
    st.markdown("### 🎯 Triggers activos")

    triggers_display = triggers_df.copy()
    if 'n_elegibles' in triggers_display.columns:
        triggers_display['n_elegibles'] = triggers_display['n_elegibles'].apply(lambda x: f'{x:,}')
    if 'pct_base' in triggers_display.columns:
        triggers_display['pct_base'] = triggers_display['pct_base'].apply(lambda x: f'{x:.1f}%')
    if 'conv_rate_pct' in triggers_display.columns:
        triggers_display['conv_rate_pct'] = triggers_display['conv_rate_pct'].apply(lambda x: f'{x:.0f}%')
    if 'valor_promedio_mxn' in triggers_display.columns:
        triggers_display['valor_promedio_mxn'] = triggers_display['valor_promedio_mxn'].apply(lambda x: f'${x:,.0f}')
    if 'impacto_estimado_mxn' in triggers_display.columns:
        triggers_display['impacto_estimado_mxn'] = triggers_display['impacto_estimado_mxn'].apply(lambda x: f'${x:,.0f}')

    cols_show = ['nombre', 'tipo', 'n_elegibles', 'pct_base',
                 'conv_rate_pct', 'valor_promedio_mxn', 'impacto_estimado_mxn']
    cols_show = [c for c in cols_show if c in triggers_display.columns]

    st.dataframe(
        triggers_display[cols_show],
        use_container_width=True,
        hide_index=True,
    )

    # === Distribución de triggers por tipo ===
    if 'tipo' in triggers_df.columns and 'impacto_estimado_mxn' in triggers_df.columns:
        st.markdown("### 💰 Impacto por trigger")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = px.bar(
                triggers_df.sort_values('n_elegibles', ascending=True),
                x='n_elegibles', y='nombre',
                color='tipo',
                title='Usuarios elegibles por trigger',
                orientation='h',
                color_discrete_map={
                    'cross_sell': '#2A9D8F',
                    'retencion': '#E63946',
                    'experiencia': '#457B9D',
                    'engagement': '#F4A261'
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            fig = px.bar(
                triggers_df.sort_values('impacto_estimado_mxn', ascending=True),
                x='impacto_estimado_mxn', y='nombre',
                color='tipo',
                title='Impacto estimado anual (MXN)',
                orientation='h',
                color_discrete_map={
                    'cross_sell': '#2A9D8F',
                    'retencion': '#E63946',
                    'experiencia': '#457B9D',
                    'engagement': '#F4A261'
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # === Distribución de Next Best Action ===
    if 'next_best_action_label' in df.columns:
        st.markdown("### 🎯 Distribución de Next Best Action")
        nba = df['next_best_action_label'].value_counts(dropna=False).reset_index()
        nba.columns = ['accion', 'usuarios']
        nba['accion'] = nba['accion'].fillna('Sin trigger activo')

        fig = px.bar(nba, x='usuarios', y='accion', orientation='h',
                     title='Cuántos clientes recibirían cada acción',
                     color='usuarios', color_continuous_scale='Greens')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# VISTA 3: MENSAJE SUGERIDO
# =====================================================================
elif vista == "💬 Mensaje Sugerido":
    st.markdown('<div class="big-title">💬 Mensaje Sugerido</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Vista previa de los mensajes que el motor enviaría hoy</div>', unsafe_allow_html=True)

    # Selector de tipo de trigger
    if 'next_best_action' in df.columns:
        triggers_disp = df['next_best_action'].dropna().unique()
        if len(triggers_disp) > 0:
            trigger_sel = st.selectbox(
                "Selecciona un trigger para ver ejemplo de mensaje",
                sorted(triggers_disp),
            )

            # Tomar un usuario representativo
            ejemplo = df[df['next_best_action'] == trigger_sel].iloc[0]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📋 Cliente ejemplo")
                st.markdown(f"**{ejemplo['user_id']}** · {ejemplo['edad']:.0f} años · {ejemplo['ciudad']}")
                st.markdown(f"Categoría: **{ejemplo['categoria']}**")
                st.markdown(f"Salud: **{ejemplo['categoria_salud']}**")
                st.markdown(f"Tópico Havi: *{ejemplo['cluster_label_dominante']}*")

                n_eleg = (df['next_best_action'] == trigger_sel).sum()
                st.metric("Total clientes con este trigger", f"{n_eleg:,}")

            with col2:
                st.markdown("### 📱 Mensaje que enviaría hoy el motor")
                st.markdown(f"""
                <div class='message-box'>
                <b>De:</b> Hey Banco<br>
                <b>Asunto:</b> {ejemplo['next_best_action_label']}<br><br>
                <hr style="border: none; border-top: 1px solid #ccc; margin: 8px 0;">
                {ejemplo['next_best_action_message']}
                </div>
                """, unsafe_allow_html=True)

                st.caption("ℹ️ Los mensajes son plantillas con slots; el motor decide *quién* y *cuándo*. Marketing aprueba el copy.")
        else:
            st.warning("No hay triggers asignados. Asegúrate de correr primero el notebook de cruce_triggers.")


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("💚 Havi Sense — Demo construida para el Datathon Hey Banco 2026 · "
           "Convertimos la base pasiva del banco en un motor de relación proactiva.")
