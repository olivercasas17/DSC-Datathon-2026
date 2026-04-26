"""Hey Banco - Datathon 2026 · Streamlit Dashboard

Reporta todos los KPIs y gráficas del EDA (eda_hey_banco.ipynb) sobre los
tres datasets: clientes, productos y transacciones.

Ejecutar con:
    streamlit run dashboard_hey_banco.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hey Banco · EDA Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2E86AB"
ACCENT = "#E63946"
PALETTE = [
    "#2E86AB", "#E63946", "#457B9D", "#2A9D8F", "#F4A261",
    "#264653", "#6A4C93", "#E76F51", "#1D3557", "#A8DADC",
]

DATA_DIR_CANDIDATES = [
    Path("Dataset/dataset_transacciones"),
    Path("ds"),
    Path("../Datathon_Working_Folder/Dataset/dataset_transacciones"),
]


# ---------------------------------------------------------------------------
# Data loading (cacheado)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datasets...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    data_dir = next((p for p in DATA_DIR_CANDIDATES if p.exists()), None)
    if data_dir is None:
        st.error(
            "No se encontró el directorio de datos. Esperado en "
            "`Dataset/dataset_transacciones/` o `ds/`."
        )
        st.stop()

    clientes = pd.read_csv(data_dir / "hey_clientes.csv")
    productos = pd.read_csv(data_dir / "hey_productos.csv")
    tx = pd.read_csv(data_dir / "hey_transacciones.csv", low_memory=False)

    productos["fecha_apertura"] = pd.to_datetime(
        productos["fecha_apertura"], errors="coerce"
    )
    productos["fecha_ultimo_movimiento"] = pd.to_datetime(
        productos["fecha_ultimo_movimiento"], errors="coerce"
    )
    tx["fecha_hora"] = pd.to_datetime(tx["fecha_hora"], errors="coerce")
    tx["mes"] = tx["fecha_hora"].dt.to_period("M").dt.to_timestamp()

    return clientes, productos, tx, data_dir


clientes, productos, tx, DATA_DIR = load_data()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def kpi_card(col, label: str, value: str, delta: str | None = None) -> None:
    col.metric(label, value, delta)


def bar(df, x, y, *, title, color=PRIMARY, orientation="v", sort=None):
    if sort is not None:
        df = df.sort_values(y if orientation == "v" else x, ascending=sort)
    fig = px.bar(df, x=x, y=y, orientation=orientation, title=title)
    fig.update_traces(marker_color=color)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )
    return fig


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })


# ---------------------------------------------------------------------------
# Sidebar · filtros globales
# ---------------------------------------------------------------------------
st.sidebar.title("💸 Hey Banco EDA")
st.sidebar.caption(f"Fuente: `{DATA_DIR}`")

sections = [
    "📊 Overview",

    "👥 Clientes · Demografía",
    "😊 Satisfacción & Hey Pro",
    "💳 Productos",
    "💱 Transacciones",
    "⚠️ Fallos operativos",
    "🧬 Segmentación",
    "🔗 Correlación",
    "🚨 Riesgo & Fraude",
    "📝 Resumen ejecutivo",
]
section = st.sidebar.radio("Sección", sections)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
min_date = pd.Timestamp(tx["fecha_hora"].min()).date()
max_date = pd.Timestamp(tx["fecha_hora"].max()).date()
date_range = st.sidebar.date_input(
    "Rango de fechas (transacciones)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
estados = sorted(clientes["estado"].dropna().unique().tolist())
sel_estados = st.sidebar.multiselect("Estados", estados, default=[])

# Filtrado
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    tx_f = tx[(tx["fecha_hora"] >= start) & (tx["fecha_hora"] < end)].copy()
else:
    tx_f = tx.copy()

if sel_estados:
    user_ids = clientes.loc[clientes["estado"].isin(sel_estados), "user_id"]
    clientes_f = clientes[clientes["estado"].isin(sel_estados)].copy()
    productos_f = productos[productos["user_id"].isin(user_ids)].copy()
    tx_f = tx_f[tx_f["user_id"].isin(user_ids)].copy()
else:
    clientes_f = clientes.copy()
    productos_f = productos.copy()

tx_completed = tx_f[tx_f["estatus"] == "completada"].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Usuarios: **{clientes_f['user_id'].nunique():,}**  \n"
    f"Productos: **{len(productos_f):,}**  \n"
    f"Transacciones: **{len(tx_f):,}**"
)


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
if section == sections[0]:
    st.title("📊 Overview · KPIs principales")
    st.caption("Todos los KPIs definidos en el EDA ejecutivo de Hey Banco.")

    n_users = clientes_f["user_id"].nunique()
    n_products = len(productos_f)
    n_tx = len(tx_f)
    total_monto = tx_completed["monto"].sum()
    avg_ticket = tx_completed["monto"].mean() if len(tx_completed) else 0
    median_ticket = tx_completed["monto"].median() if len(tx_completed) else 0

    pct_hey_pro = clientes_f["es_hey_pro"].mean() * 100
    pct_nomina = clientes_f["nomina_domiciliada"].mean() * 100
    pct_seguro = clientes_f["tiene_seguro"].mean() * 100
    pct_remesas = clientes_f["recibe_remesas"].mean() * 100
    pct_hey_shop = clientes_f["usa_hey_shop"].mean() * 100
    pct_atipico_user = clientes_f["patron_uso_atipico"].mean() * 100

    nps_avg = clientes_f["satisfaccion_1_10"].mean()
    promoters = (clientes_f["satisfaccion_1_10"] >= 9).mean()
    detractors = (clientes_f["satisfaccion_1_10"] <= 6).mean()
    nps_score = (promoters - detractors) * 100

    tx_status = tx_f["estatus"].value_counts(normalize=True) * 100
    approval_rate = tx_status.get("completada", 0)
    decline_rate = tx_status.get("no_procesada", 0)
    dispute_rate = tx_status.get("en_disputa", 0)
    reversal_rate = tx_status.get("revertida", 0)

    cashback_total = tx_f["cashback_generado"].sum()
    cashback_users = tx_f.loc[tx_f["cashback_generado"] > 0, "user_id"].nunique()

    st.subheader("Volúmenes")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Usuarios únicos", f"{n_users:,}")
    kpi_card(c2, "Productos", f"{n_products:,}")
    kpi_card(c3, "Transacciones", f"{n_tx:,}")
    kpi_card(c4, "Volumen completado (MXN)", f"${total_monto:,.0f}")

    st.subheader("Ticket y uso")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Ticket promedio", f"${avg_ticket:,.2f}")
    kpi_card(c2, "Ticket mediano", f"${median_ticket:,.2f}")
    kpi_card(c3, "Cashback generado", f"${cashback_total:,.0f}")
    kpi_card(c4, "Usuarios con cashback", f"{cashback_users:,}")

    st.subheader("Perfil de usuario")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "% Hey Pro", f"{pct_hey_pro:.1f}%")
    kpi_card(c2, "% Nómina domiciliada", f"{pct_nomina:.1f}%")
    kpi_card(c3, "% Con seguro", f"{pct_seguro:.1f}%")
    kpi_card(c4, "% Recibe remesas", f"{pct_remesas:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "% Usa Hey Shop", f"{pct_hey_shop:.1f}%")
    kpi_card(c2, "% Patrón atípico", f"{pct_atipico_user:.1f}%")
    kpi_card(c3, "Satisfacción prom. (1-10)", f"{nps_avg:.2f}")
    kpi_card(c4, "NPS score", f"{nps_score:.1f}")

    st.subheader("Estatus de transacciones")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Approval rate", f"{approval_rate:.2f}%")
    kpi_card(c2, "Decline rate", f"{decline_rate:.2f}%")
    kpi_card(c3, "Dispute rate", f"{dispute_rate:.2f}%")
    kpi_card(c4, "Reversal rate", f"{reversal_rate:.2f}%")

    





# ---------------------------------------------------------------------------
# 3. Clientes · Demografía
# ---------------------------------------------------------------------------
elif section == sections[1]:
    st.title("👥 Clientes · Demografía")

    # Edad
    st.subheader("Distribución de edad")
    fig = px.histogram(
        clientes_f, x="edad", nbins=40,
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(height=380, bargap=0.02)
    st.plotly_chart(fig, use_container_width=True)

    # Género
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Género")
        gender_counts = clientes_f["sexo"].value_counts().reset_index()
        gender_counts.columns = ["sexo", "n"]
        fig = px.bar(gender_counts, x="sexo", y="n",
                     color="sexo", color_discrete_sequence=PALETTE)
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Preferencia de canal digital")
        dev = clientes_f["preferencia_canal"].value_counts().reset_index()
        dev.columns = ["canal", "n"]
        fig = px.bar(dev, x="canal", y="n", color="canal",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Top estados
    st.subheader("Top 15 estados")
    top_estados = clientes_f["estado"].value_counts().head(15).reset_index()
    top_estados.columns = ["estado", "usuarios"]
    fig = px.bar(
        top_estados.sort_values("usuarios"),
        x="usuarios", y="estado", orientation="h",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Ingresos por educación / ocupación
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ingreso mensual mediano por nivel educativo")
        inc_edu = (
            clientes_f.groupby("nivel_educativo")["ingreso_mensual_mxn"]
            .agg(["mean", "median", "count"]).sort_values("median")
            .reset_index()
        )
        fig = px.bar(
            inc_edu, x="nivel_educativo", y="median",
            color_discrete_sequence=["#457B9D"],
            hover_data=["mean", "count"],
        )
        fig.update_layout(height=420, yaxis_title="MXN (mediana)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ingreso mensual mediano por ocupación")
        inc_occ = (
            clientes_f.groupby("ocupacion")["ingreso_mensual_mxn"]
            .agg(["mean", "median", "count"]).sort_values("median")
            .reset_index()
        )
        fig = px.bar(
            inc_occ, x="ocupacion", y="median",
            color_discrete_sequence=["#457B9D"],
            hover_data=["mean", "count"],
        )
        fig.update_layout(height=420, yaxis_title="MXN (mediana)")
        st.plotly_chart(fig, use_container_width=True)

    # Inactividad y antigüedad
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Días desde el último login")
        fig = px.histogram(
            clientes_f, x="dias_desde_ultimo_login", nbins=40,
            color_discrete_sequence=["#2A9D8F"],
        )
        fig.update_layout(height=380, bargap=0.02)
        st.plotly_chart(fig, use_container_width=True)

        inactivos_30 = (clientes_f["dias_desde_ultimo_login"] > 30).mean() * 100
        inactivos_90 = (clientes_f["dias_desde_ultimo_login"] > 90).mean() * 100
        c1, c2 = st.columns(2)
        c1.metric("Inactivos +30 días", f"{inactivos_30:.2f}%")
        c2.metric("Inactivos +90 días", f"{inactivos_90:.2f}%")

    with col2:
        st.subheader("Antigüedad del cliente (años)")
        antiguedad = clientes_f["antiguedad_dias"] / 365.25
        fig = px.histogram(
            antiguedad, nbins=40,
            color_discrete_sequence=["#F4A261"],
            labels={"value": "años"},
        )
        fig.update_layout(height=380, bargap=0.02, showlegend=False,
                          xaxis_title="Años")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Satisfacción & Hey Pro
# ---------------------------------------------------------------------------
elif section == sections[2]:
    st.title("😊 Satisfacción & Hey Pro")

    nps_avg = clientes_f["satisfaccion_1_10"].mean()
    promoters = (clientes_f["satisfaccion_1_10"] >= 9).mean()
    detractors = (clientes_f["satisfaccion_1_10"] <= 6).mean()
    neutrals = 1 - promoters - detractors
    nps_score = (promoters - detractors) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satisfacción promedio", f"{nps_avg:.2f}")
    c2.metric("Promotores (9-10)", f"{promoters*100:.1f}%")
    c3.metric("Detractores (≤6)", f"{detractors*100:.1f}%")
    c4.metric("NPS", f"{nps_score:.1f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Satisfacción: Hey Pro vs No")
        nps_by_pro = (
            clientes_f.groupby("es_hey_pro")["satisfaccion_1_10"]
            .mean().reset_index()
        )
        nps_by_pro["es_hey_pro"] = nps_by_pro["es_hey_pro"].map(
            {True: "Hey Pro", False: "Estándar"}
        )
        fig = px.bar(
            nps_by_pro, x="es_hey_pro", y="satisfaccion_1_10",
            color="es_hey_pro",
            color_discrete_sequence=["#2A9D8F", "#E76F51"],
        )
        fig.update_layout(height=380, showlegend=False,
                          yaxis_title="Satisfacción (1-10)",
                          xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mezcla Promotor / Neutral / Detractor")
        nps_mix = pd.DataFrame({
            "tipo": ["Promotores (9-10)", "Neutrales (7-8)", "Detractores (≤6)"],
            "pct": [promoters * 100, neutrals * 100, detractors * 100],
        })
        fig = px.pie(
            nps_mix, names="tipo", values="pct", hole=0.55,
            color_discrete_sequence=["#2A9D8F", "#F4A261", "#E63946"],
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfacción vs # productos activos")
    nps_by_nprod = (
        clientes_f.groupby("num_productos_activos")["satisfaccion_1_10"]
        .mean().reset_index()
    )
    fig = px.bar(
        nps_by_nprod, x="num_productos_activos", y="satisfaccion_1_10",
        color_discrete_sequence=["#457B9D"],
    )
    fig.update_layout(height=380, xaxis_title="# productos activos",
                      yaxis_title="Satisfacción promedio")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Productos
# ---------------------------------------------------------------------------
elif section == sections[3]:
    st.title("💳 Productos")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mix de productos contratados")
        prod_mix = productos_f["tipo_producto"].value_counts().reset_index()
        prod_mix.columns = ["tipo_producto", "cantidad"]
        fig = px.bar(
            prod_mix.sort_values("cantidad"),
            x="cantidad", y="tipo_producto", orientation="h",
            color_discrete_sequence=["#6A4C93"],
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Estatus de productos")
        prod_estatus = productos_f["estatus"].value_counts().reset_index()
        prod_estatus.columns = ["estatus", "n"]
        fig = px.bar(
            prod_estatus, x="estatus", y="n",
            color_discrete_sequence=["#1D3557"],
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    # Crédito: utilización y riesgo
    st.markdown("### Portafolio de crédito")
    cred_types = [
        "tarjeta_credito_hey", "tarjeta_credito_garantizada",
        "tarjeta_credito_negocios", "credito_personal",
        "credito_auto", "credito_nomina",
    ]
    creditos = productos_f[productos_f["tipo_producto"].isin(cred_types)].copy()

    deuda_total = creditos["saldo_actual"].sum()
    limite_total = creditos["limite_credito"].sum()
    util_cartera = deuda_total / limite_total if limite_total else np.nan
    high_util = (creditos["utilizacion_pct"] >= 0.8).mean() * 100 if len(creditos) else 0
    usuarios_alto_riesgo = (
        creditos.loc[creditos["utilizacion_pct"] >= 0.8, "user_id"].nunique()
    )

    inv = productos_f[productos_f["tipo_producto"] == "inversion_hey"]
    aum = inv["saldo_actual"].sum()
    n_users_all = clientes_f["user_id"].nunique() or 1
    pct_users_inv = inv["user_id"].nunique() / n_users_all * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deuda total", f"${deuda_total:,.0f}")
    c2.metric("Límite autorizado", f"${limite_total:,.0f}")
    c3.metric("Utilización cartera", f"{util_cartera*100:.2f}%")
    c4.metric("% productos con util ≥ 80%", f"{high_util:.2f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Usuarios alto riesgo (util ≥80%)", f"{usuarios_alto_riesgo:,}")
    c2.metric("AUM en Inversión Hey", f"${aum:,.0f}")
    c3.metric("% usuarios con inversión", f"{pct_users_inv:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Utilización promedio por tipo de crédito")
        util_by_type = (
            creditos.groupby("tipo_producto")["utilizacion_pct"].mean()
            .sort_values() * 100
        ).reset_index()
        util_by_type.columns = ["tipo_producto", "util_pct"]
        fig = px.bar(
            util_by_type, x="util_pct", y="tipo_producto", orientation="h",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(height=420, xaxis_title="% utilización")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tasa de interés anual promedio por tipo")
        tasa_by_type = (
            creditos.groupby("tipo_producto")["tasa_interes_anual"]
            .mean().sort_values().reset_index()
        )
        tasa_by_type.columns = ["tipo_producto", "tasa_prom"]
        fig = px.bar(
            tasa_by_type, x="tasa_prom", y="tipo_producto", orientation="h",
            color_discrete_sequence=["#F4A261"],
        )
        fig.update_layout(height=420, xaxis_title="Tasa anual promedio (%)")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 6. Transacciones
# ---------------------------------------------------------------------------
elif section == sections[4]:
    st.title("💱 Transacciones")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transacciones por canal")
        canal_counts = tx_f["canal"].value_counts().reset_index()
        canal_counts.columns = ["canal", "tx"]
        fig = px.bar(
            canal_counts.sort_values("tx"),
            x="tx", y="canal", orientation="h",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transacciones por tipo de operación")
        op_counts = tx_f["tipo_operacion"].value_counts().reset_index()
        op_counts.columns = ["tipo_operacion", "tx"]
        fig = px.bar(
            op_counts.sort_values("tx"),
            x="tx", y="tipo_operacion", orientation="h",
            color_discrete_sequence=["#457B9D"],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transacciones por hora del día")
        hora_dist = tx_f.groupby("hora_del_dia").size().reset_index(name="n")
        fig = px.bar(
            hora_dist, x="hora_del_dia", y="n",
            color_discrete_sequence=["#264653"],
        )
        fig.update_layout(height=380, xaxis_title="Hora", yaxis_title="Cantidad")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transacciones por día de la semana")
        dias_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        dia_dist = (
            tx_f["dia_semana"].value_counts().reindex(dias_order)
            .reset_index()
        )
        dia_dist.columns = ["dia_semana", "n"]
        fig = px.bar(
            dia_dist, x="dia_semana", y="n",
            color_discrete_sequence=["#2A9D8F"],
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Serie mensual
    st.subheader("Monto mensual (transacciones completadas)")
    monto_mes = (
        tx_completed
        .assign(mes=tx_completed["fecha_hora"].dt.to_period("M").dt.to_timestamp())
        .groupby("mes")["monto"].agg(["sum", "count"]).reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monto_mes["mes"], y=monto_mes["sum"],
        mode="lines+markers", name="Monto (MXN)",
        line=dict(color=ACCENT, width=3),
    ))
    fig.add_trace(go.Bar(
        x=monto_mes["mes"], y=monto_mes["count"],
        name="# transacciones", yaxis="y2",
        marker_color=PRIMARY, opacity=0.35,
    ))
    fig.update_layout(
        height=460,
        yaxis=dict(title="Monto MXN"),
        yaxis2=dict(title="# tx", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Categorías + comercios
    compras = tx_f[tx_f["tipo_operacion"] == "compra"]

    st.subheader("Gasto total por categoría (compras)")
    cat_spend = (
        compras.groupby("categoria_mcc")["monto"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
        .reset_index()
    )
    fig = px.bar(
        cat_spend.head(14).sort_values("sum"),
        x="sum", y="categoria_mcc", orientation="h",
        color_discrete_sequence=["#6A4C93"],
        hover_data=["count", "mean"],
    )
    fig.update_layout(height=500, xaxis_title="MXN")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 comercios")
    top_comercios = (
        compras.groupby("comercio_nombre")["monto"]
        .agg(["sum", "count"]).sort_values("sum", ascending=False)
        .head(20).reset_index()
    )
    st.dataframe(
        top_comercios.rename(columns={"sum": "monto", "count": "n_tx"}),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 7. Fallos operativos
# ---------------------------------------------------------------------------
elif section == sections[5]:
    st.title("⚠️ Fallos operativos y uso especial")

    compras = tx_f[tx_f["tipo_operacion"] == "compra"]
    reintentos_rate = (tx_f["intento_numero"] > 1).mean() * 100
    msi_usage = tx_f["meses_diferidos"].notna().sum()
    msi_pct = msi_usage / len(compras) * 100 if len(compras) else 0
    pct_intl = tx_f["es_internacional"].mean() * 100
    pct_tx_atipico = tx_f["patron_uso_atipico"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("% tx con reintentos", f"{reintentos_rate:.2f}%")
    c2.metric("% compras con MSI", f"{msi_pct:.2f}%")
    c3.metric("% tx internacionales", f"{pct_intl:.2f}%")
    c4.metric("% tx con patrón atípico", f"{pct_tx_atipico:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Motivos de transacciones no procesadas")
        no_proc = tx_f[tx_f["estatus"] == "no_procesada"]
        motivos = no_proc["motivo_no_procesada"].value_counts().reset_index()
        motivos.columns = ["motivo", "n"]
        fig = px.bar(
            motivos.sort_values("n"),
            x="n", y="motivo", orientation="h",
            color_discrete_sequence=["#E76F51"],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribución de meses diferidos (MSI)")
        msi_dist = (
            tx_f["meses_diferidos"].dropna().astype(int)
            .value_counts().sort_index().reset_index()
        )
        msi_dist.columns = ["meses", "n"]
        if len(msi_dist):
            fig = px.bar(
                msi_dist, x="meses", y="n",
                color_discrete_sequence=["#F4A261"],
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin compras a MSI en el rango seleccionado.")


# ---------------------------------------------------------------------------
# 8. Segmentación
# ---------------------------------------------------------------------------
elif section == sections[6]:
    st.title("🧬 Segmentación")

    tx_user = tx_completed.groupby("user_id").agg(
        n_tx=("transaccion_id", "count"),
        monto_total=("monto", "sum"),
        ticket_prom=("monto", "mean"),
    ).reset_index()

    clientes_enriched = clientes_f.merge(tx_user, on="user_id", how="left").fillna(
        {"n_tx": 0, "monto_total": 0, "ticket_prom": 0}
    )

    st.subheader("Segmento Hey Pro vs Estándar")
    seg_pro = clientes_enriched.groupby("es_hey_pro").agg(
        usuarios=("user_id", "count"),
        tx_prom=("n_tx", "mean"),
        gasto_prom=("monto_total", "mean"),
        ingreso_prom=("ingreso_mensual_mxn", "mean"),
        score_buro_prom=("score_buro", "mean"),
        satisfaccion_prom=("satisfaccion_1_10", "mean"),
    )
    seg_pro.index = seg_pro.index.map({True: "Hey Pro", False: "Estándar"})
    st.dataframe(seg_pro.round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            seg_pro.reset_index(),
            x="es_hey_pro", y="gasto_prom",
            color="es_hey_pro",
            color_discrete_sequence=["#E76F51", "#2A9D8F"],
            title="Gasto promedio por segmento",
        )
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            seg_pro.reset_index(),
            x="es_hey_pro", y="tx_prom",
            color="es_hey_pro",
            color_discrete_sequence=["#E76F51", "#2A9D8F"],
            title="# transacciones promedio por segmento",
        )
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Segmento por rango de ingreso
    st.subheader("Segmento por rango de ingreso mensual")
    bins = [0, 10000, 20000, 35000, 60000, 100000, np.inf]
    labels = ["<10k", "10-20k", "20-35k", "35-60k", "60-100k", ">100k"]
    clientes_enriched["rango_ingreso"] = pd.cut(
        clientes_enriched["ingreso_mensual_mxn"], bins=bins, labels=labels
    )
    seg_ing = clientes_enriched.groupby("rango_ingreso").agg(
        usuarios=("user_id", "count"),
        gasto_prom=("monto_total", "mean"),
        tx_prom=("n_tx", "mean"),
        nps_prom=("satisfaccion_1_10", "mean"),
        pct_heypro=("es_hey_pro", "mean"),
    )
    st.dataframe(seg_ing.round(2), use_container_width=True)

    fig = px.bar(
        seg_ing.reset_index(),
        x="rango_ingreso", y="gasto_prom",
        color_discrete_sequence=["#2A9D8F"],
        title="Gasto promedio por rango de ingreso",
    )
    fig.update_layout(height=420, yaxis_title="MXN")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 9. Correlación
# ---------------------------------------------------------------------------
elif section == sections[7]:
    st.title("🔗 Matriz de correlación (usuarios)")

    tx_user = tx_completed.groupby("user_id").agg(
        n_tx=("transaccion_id", "count"),
        monto_total=("monto", "sum"),
        ticket_prom=("monto", "mean"),
    ).reset_index()

    clientes_enriched = clientes_f.merge(tx_user, on="user_id", how="left").fillna(
        {"n_tx": 0, "monto_total": 0, "ticket_prom": 0}
    )

    num_cols = [
        "edad", "ingreso_mensual_mxn", "antiguedad_dias", "score_buro",
        "dias_desde_ultimo_login", "satisfaccion_1_10", "num_productos_activos",
        "n_tx", "monto_total", "ticket_prom",
    ]
    corr = clientes_enriched[num_cols].corr().round(3)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla"):
        st.dataframe(corr, use_container_width=True)


# ---------------------------------------------------------------------------
# 10. Riesgo & Fraude
# ---------------------------------------------------------------------------
elif section == sections[8]:
    st.title("🚨 Riesgo & Fraude")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("% patrón atípico por canal")
        atipico_canal = (
            tx_f.groupby("canal")["patron_uso_atipico"].mean()
            .sort_values() * 100
        ).reset_index()
        atipico_canal.columns = ["canal", "pct_atipico"]
        fig = px.bar(
            atipico_canal, x="pct_atipico", y="canal", orientation="h",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(height=420, xaxis_title="% atípico")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("% patrón atípico por tipo de operación")
        atipico_op = (
            tx_f.groupby("tipo_operacion")["patron_uso_atipico"].mean()
            .sort_values() * 100
        ).reset_index()
        atipico_op.columns = ["tipo_operacion", "pct_atipico"]
        fig = px.bar(
            atipico_op, x="pct_atipico", y="tipo_operacion", orientation="h",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(height=420, xaxis_title="% atípico")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 categorías con disputas")
    disputa_cat = (
        tx_f[tx_f["estatus"] == "en_disputa"]["categoria_mcc"]
        .value_counts().head(10).reset_index()
    )
    disputa_cat.columns = ["categoria_mcc", "n"]
    if len(disputa_cat):
        fig = px.bar(
            disputa_cat.sort_values("n"),
            x="n", y="categoria_mcc", orientation="h",
            color_discrete_sequence=["#E76F51"],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin disputas en el rango seleccionado.")

    st.subheader("Monto en tx atípicas vs normales")
    monto_atipico = tx_f.groupby("patron_uso_atipico")["monto"].agg(
        ["mean", "median", "count"]
    )
    monto_atipico.index = monto_atipico.index.map(
        {True: "Atípica", False: "Normal"}
    )
    st.dataframe(monto_atipico.round(2), use_container_width=True)


# ---------------------------------------------------------------------------
# 11. Resumen ejecutivo
# ---------------------------------------------------------------------------
elif section == sections[9]:
    st.title("📝 Resumen ejecutivo")

    st.markdown(
        """

**Hallazgos clave**

1. El NPS y la satisfacción promedio correlacionan positivamente con el
   número de productos activos y el segmento Hey Pro.
2. Hey Pro concentra mayor gasto y más transacciones por usuario.
3. Los canales digitales (apps) dominan el volumen — priorizar el roadmap móvil.
4. Supermercado, restaurante y servicios digitales lideran el gasto,
   abriendo oportunidades de partnerships y cashback dirigido.
5. `saldo_insuficiente` y `limite_excedido` son los principales motivos de
   declines — oportunidad para sobregiro o aumento de línea.
6. Las tarjetas con utilización ≥ 80 % son un segmento prioritario de
   early-warning.
7. El patrón atípico se concentra en ciertos canales/operaciones —
   base para reglas de fraude.
        """
    )

    md_path = Path("outputs/kpis.md")
    if md_path.exists():
        with st.expander("📄 kpis.md generado por el notebook"):
            st.markdown(md_path.read_text(encoding="utf-8"))

    st.caption(
        "Dashboard generado para el Datathon 2026."
    )
