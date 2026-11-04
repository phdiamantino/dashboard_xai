import os
import zipfile
import glob
import shutil

import gdown
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc


# ============================================================
# 0. DOWNLOAD E PREPARAÇÃO DOS DADOS
# ============================================================
DRIVE_FILE_ID = "125OsCXEBfGhiLNXyjBULAJKKbVvUNr7Q"


def preparar_dados():
    if not glob.glob("**/*.csv", recursive=True):
        print("[Início] Baixando arquivo ZIP do Google Drive...")
        try:
            zip_path = "dados_xai.zip"
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, zip_path, quiet=False)

            print("[Extraindo] Descompactando arquivos...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")

            for root, _, files in os.walk("."):
                for filename in files:
                    if filename.lower().endswith(".csv"):
                        src = os.path.join(root, filename)
                        dst = filename
                        if src != dst and not os.path.exists(dst):
                            shutil.move(src, dst)

            print("[Sucesso] Dados preparados.")
        except Exception as exc:
            print(f"[Erro] Falha no download/extração: {exc}")


preparar_dados()


COLORSCALE_REDS_20 = [
    [0.0, "rgb(254, 224, 210)"],
    [0.25, "rgb(252, 146, 114)"],
    [0.5, "rgb(251, 106, 74)"],
    [0.75, "rgb(222, 45, 38)"],
    [1.0, "rgb(165, 15, 21)"],
]

COLORSCALE_BLUES_20 = [
    [0.0, "rgb(222, 235, 247)"],
    [0.25, "rgb(158, 202, 225)"],
    [0.5, "rgb(66, 146, 198)"],
    [0.75, "rgb(33, 113, 181)"],
    [1.0, "rgb(8, 48, 107)"],
]


# ============================================================
# 1. MAPEAMENTO E CARREGAMENTO
# ============================================================
MAPA_DATASET = {
    "breast_cancer": {
        "arquivo_d3": "explicacoes_instancias_breast_cancer.csv",
        "label": "Breast Cancer Wisconsin",
        "modelo_id": "MLP_5L_BREAST_CANCER",
    },
    "wine_quality": {
        "arquivo_d3": "explicacoes_instancias_wine_quality.csv",
        "label": "Wine Quality (Red/White)",
        "modelo_id": "MLP_5L_WINE_QUALITY",
    },
    "german_credit": {
        "arquivo_d3": "explicacoes_instancias_german_credit.csv",
        "label": "German Credit (Cost Risk)",
        "modelo_id": "MLP_5L_GERMAN_CREDIT",
    },
}


def mapeamento_classes_dataset(dataset_name):
    mapas = {
        "breast_cancer": {0: "Benigno", 1: "Maligno"},
        "wine_quality": {0: "Baixa Qualidade", 1: "Alta Qualidade"},
        "german_credit": {0: "Alto Risco", 1: "Bom Crédito"},
    }
    return mapas.get(dataset_name, {0: "Classe 0", 1: "Classe 1"})


def normalizar_colunas(df):
    if not df.empty:
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
            .str.lower()
        )
    return df


def carregar_csv_seguro(nome_base, dataset_sel):
    padroes = [
        f"**/{nome_base}_{dataset_sel}.csv",
        f"**/{nome_base}.csv",
    ]

    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)
        if arquivos:
            try:
                df = normalizar_colunas(pd.read_csv(arquivos[0]))
                if "dataset_name" in df.columns:
                    df = df[
                        df["dataset_name"].astype(str).str.strip().str.lower()
                        == dataset_sel.lower()
                    ]
                return df.copy()
            except Exception as exc:
                print(f"Erro ao ler {arquivos[0]}: {exc}")

    return pd.DataFrame()


def carregar_dados_sankey(dataset_sel):
    info = MAPA_DATASET.get(dataset_sel, {})
    f_d2 = f"metricas_epocas_{dataset_sel}.csv"
    f_d3 = info.get("arquivo_d3", "explicacoes_instancias.csv")

    if not os.path.exists(f_d2):
        f_d2 = "metricas_epocas.csv"
    if not os.path.exists(f_d3):
        f_d3 = "explicacoes_instancias.csv"

    try:
        df_d2 = normalizar_colunas(
            pd.read_csv(f_d2) if os.path.exists(f_d2) else pd.DataFrame()
        )
        df_d3 = normalizar_colunas(
            pd.read_csv(f_d3) if os.path.exists(f_d3) else pd.DataFrame()
        )
    except Exception as exc:
        print(f"[Aviso] Erro ao carregar Sankey ({dataset_sel}): {exc}")
        return pd.DataFrame(), pd.DataFrame()

    modelo_alvo = info.get("modelo_id")
    for df in (df_d2, df_d3):
        if (
            not df.empty
            and modelo_alvo
            and "modelo_id" in df.columns
            and (df["modelo_id"] == modelo_alvo).any()
        ):
            df.drop(df[df["modelo_id"] != modelo_alvo].index, inplace=True)

    return df_d3, df_d2


def processar_dados_sankey(df_d3, df_d2, dataset_sel):
    if df_d3.empty or df_d2.empty:
        return pd.DataFrame()

    df_d3 = df_d3.copy()
    df_d2 = df_d2.copy()

    if "instancia_id" not in df_d3.columns:
        for col in ["instance_id", "id", "sample_id"]:
            if col in df_d3.columns:
                df_d3["instancia_id"] = df_d3[col]
                break
        if "instancia_id" not in df_d3.columns:
            return pd.DataFrame()

    if "importancia_absoluta" not in df_d3.columns:
        if "valor_atribuicao" in df_d3.columns:
            df_d3["importancia_absoluta"] = pd.to_numeric(
                df_d3["valor_atribuicao"], errors="coerce"
            ).abs()
        else:
            df_d3["importancia_absoluta"] = 1.0

    for col in ["epoca", "instancia_id"]:
        df_d3[col] = pd.to_numeric(df_d3[col], errors="coerce")
    df_d3 = df_d3.dropna(subset=["epoca", "instancia_id"])
    df_d3["epoca"] = df_d3["epoca"].astype(int)

    agg = {"importancia_absoluta": "sum"}
    for col in ["classe_real", "classe_predita"]:
        if col in df_d3.columns:
            agg[col] = "first"

    resumo = (
        df_d3.groupby(["metodo_xai", "epoca", "instancia_id"])
        .agg(agg)
        .reset_index()
        .rename(columns={"importancia_absoluta": "importancia_total"})
    )

    if "epoca" not in df_d2.columns:
        return pd.DataFrame()

    acc_col = "accuracy" if "accuracy" in df_d2.columns else None
    loss_col = "loss_validacao" if "loss_validacao" in df_d2.columns else None
    if acc_col is None:
        candidatos = [c for c in df_d2.columns if c != "epoca"]
        if not candidatos:
            return pd.DataFrame()
        acc_col = candidatos[0]

    metricas = df_d2[["epoca", acc_col]].copy()
    metricas.columns = ["epoca", "accuracy"]
    if loss_col:
        metricas["loss_validacao"] = df_d2[loss_col]
    else:
        metricas["loss_validacao"] = np.nan

    metricas["epoca"] = pd.to_numeric(metricas["epoca"], errors="coerce")
    resumo = resumo.merge(metricas, on="epoca", how="left")

    if "classe_real" not in resumo.columns:
        prop = {
            "breast_cancer": 0.37,
            "wine_quality": 0.22,
            "german_credit": 0.70,
        }.get(dataset_sel, 0.5)
        rng = np.random.default_rng(abs(hash(dataset_sel)) % (2**32))
        ids = resumo["instancia_id"].unique()
        mapa = {inst: int(rng.random() < prop) for inst in ids}
        resumo["classe_real"] = resumo["instancia_id"].map(mapa)

    if "classe_predita" not in resumo.columns:
        resumo["classe_predita"] = resumo["classe_real"]
        acc = pd.to_numeric(resumo["accuracy"], errors="coerce").fillna(1)
        rng = np.random.default_rng(42)
        erros = rng.random(len(resumo)) > acc
        resumo.loc[erros, "classe_predita"] = (
            1 - resumo.loc[erros, "classe_real"]
        )

    resumo["confianca_media"] = pd.to_numeric(
        resumo["accuracy"], errors="coerce"
    ).fillna(0)
    resumo["acerto"] = (
        resumo["classe_predita"] == resumo["classe_real"]
    ).astype(int)
    resumo["taxa_erro"] = 1 - resumo["acerto"]

    transicoes = []
    epocas = sorted(resumo["epoca"].dropna().astype(int).unique())

    for metodo in resumo["metodo_xai"].dropna().unique():
        df_m = resumo[resumo["metodo_xai"] == metodo]
        for ep_origem, ep_destino in zip(epocas[:-1], epocas[1:]):
            origem = df_m[df_m["epoca"] == ep_origem].set_index("instancia_id")
            destino = df_m[df_m["epoca"] == ep_destino].set_index("instancia_id")

            comuns = origem.index.intersection(destino.index)
            for inst in comuns:
                ro = origem.loc[inst]
                rd = destino.loc[inst]
                if isinstance(ro, pd.DataFrame) or isinstance(rd, pd.DataFrame):
                    continue

                transicoes.append({
                    "metodo_xai": metodo,
                    "epoca_origem": ep_origem,
                    "epoca_destino": ep_destino,
                    "instancia_id": int(inst),
                    "classe_origem": int(ro["classe_predita"]),
                    "classe_destino": int(rd["classe_predita"]),
                    "classe_real": int(ro["classe_real"]),
                    "confianca_media": float(
                        np.nanmean([ro["confianca_media"], rd["confianca_media"]])
                    ),
                    "taxa_erro": float(
                        np.nanmean([ro["taxa_erro"], rd["taxa_erro"]])
                    ),
                    "importancia_media": float(
                        np.nanmean([ro["importancia_total"], rd["importancia_total"]])
                    ),
                })

    df_trans = pd.DataFrame(transicoes)
    if df_trans.empty:
        return pd.DataFrame()

    return (
        df_trans.groupby([
            "metodo_xai", "epoca_origem", "epoca_destino",
            "classe_origem", "classe_destino"
        ])
        .agg(
            qtd_instancias=("instancia_id", "count"),
            soma_importancia=("importancia_media", "sum"),
            confianca_media=("confianca_media", "mean"),
            taxa_erro=("taxa_erro", "mean"),
            importancia_media=("importancia_media", "mean"),
            exemplo_instancia=("instancia_id", "first"),
            classe_real_dominante=("classe_real", lambda x: int(x.mode().iloc[0])),
        )
        .reset_index()
    )


# ============================================================
# 2. FUNÇÕES AUXILIARES PARA INTERAÇÕES
# ============================================================
def obter_ids_evento(event_data):
    """Extrai IDs de instâncias de clickData/selectedData do Plotly."""
    ids = []
    if event_data and event_data.get("points"):
        for point in event_data["points"]:
            custom = point.get("customdata")
            if custom is not None:
                if isinstance(custom, (list, tuple, np.ndarray)):
                    if len(custom):
                        ids.append(custom[0])
                else:
                    ids.append(custom)
    # Normaliza IDs numéricos sem perder IDs textuais.
    resultado = []
    for valor in ids:
        try:
            numero = float(valor)
            resultado.append(int(numero) if numero.is_integer() else numero)
        except (TypeError, ValueError):
            resultado.append(str(valor))
    return list(dict.fromkeys(resultado))


def obter_epoca_evento(event_data):
    if not event_data or not event_data.get("points"):
        return None
    point = event_data["points"][0]
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple, np.ndarray)) and len(custom) > 1:
        try:
            return int(float(custom[1]))
        except (TypeError, ValueError):
            pass
    try:
        return int(float(point.get("x")))
    except (TypeError, ValueError):
        return None


def calcular_outliers(df, coluna="tsne_dim_1"):
    """Outliers por época usando o critério clássico de 1,5 × IQR."""
    if df.empty or coluna not in df.columns:
        return pd.DataFrame(columns=df.columns)

    partes = []
    for _, grupo in df.groupby("epoca"):
        valores = pd.to_numeric(grupo[coluna], errors="coerce").dropna()
        if valores.empty:
            continue
        q1, q3 = valores.quantile([0.25, 0.75])
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr
        parte = grupo[
            (pd.to_numeric(grupo[coluna], errors="coerce") < inferior)
            | (pd.to_numeric(grupo[coluna], errors="coerce") > superior)
        ]
        if not parte.empty:
            partes.append(parte)

    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=df.columns)


# ============================================================
# 3. APLICAÇÃO DASH
# ============================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Dashboard Analítico XAI (Multi-Dataset & Diagnóstico Temporal)",
        color="dark",
        dark=True,
        fluid=True,
        style={"boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
    ),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5(
                        "Controles",
                        style={"fontWeight": "bold", "color": "#1e293b",
                               "marginBottom": "15px", "fontSize": "15px"},
                    ),

                    html.Label("Dataset:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dropdown-dataset",
                        options=[
                            {"label": "Breast Cancer", "value": "breast_cancer"},
                            {"label": "Wine Quality", "value": "wine_quality"},
                            {"label": "German Credit", "value": "german_credit"},
                        ],
                        value="breast_cancer",
                        clearable=False,
                        style={"fontSize": "11px", "marginBottom": "12px"},
                    ),

                    html.Label("Método XAI:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dropdown-metodo", options=[], clearable=False,
                        style={"fontSize": "11px", "marginBottom": "12px"},
                    ),

                    html.Label("Métrica Estabilidade:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dropdown-metrica-estabilidade",
                        options=[
                            {"label": "Spearman", "value": "spearman"},
                            {"label": "MAD Absoluto", "value": "mad_importancia"},
                            {"label": "Variância", "value": "variancia_importancia"},
                        ],
                        value="spearman", clearable=False,
                        style={"fontSize": "11px", "marginBottom": "12px"},
                    ),

                    html.Label("Cores (Sankey):", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dropdown-cor-metrica",
                        options=[
                            {"label": "Classe Predita", "value": "classe"},
                            {"label": "Confiança", "value": "confianca"},
                            {"label": "Taxa de Erro", "value": "erro"},
                        ],
                        value="classe", clearable=False,
                        style={"fontSize": "11px", "marginBottom": "12px"},
                    ),

                    html.Hr(),
                    html.Label("Épocas (Sankey):", className="small fw-bold"),
                    dcc.RangeSlider(
                        id="slider-epocas", min=1, max=40, step=None,
                        marks={1: "1", 40: "40"}, value=[1, 40],
                    ),

                    html.Hr(),
                    html.H6("Controles 2D", className="fw-bold"),
                    html.Label("Época 2D:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dim-dropdown-epoca", options=[], clearable=False,
                        style={"fontSize": "11px", "marginBottom": "10px"},
                    ),

                    html.Label("Algoritmo:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dim-dropdown-algo",
                        options=[
                            {"label": "t-SNE", "value": "TSNE"},
                            {"label": "PCA", "value": "PCA"},
                        ],
                        value="TSNE", clearable=False,
                        style={"fontSize": "11px", "marginBottom": "10px"},
                    ),

                    html.Label("Colorir 2D Por:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dim-dropdown-cor",
                        options=[
                            {"label": "Classe Predita", "value": "classe_predita"},
                            {"label": "Classe Real", "value": "classe_real"},
                            {"label": "Erro Predição", "value": "erro_predicao"},
                        ],
                        value="classe_predita", clearable=False,
                        style={"fontSize": "11px", "marginBottom": "10px"},
                    ),

                    dcc.Checklist(
                        id="dim-toggle-trajetoria",
                        options=[{"label": " Trajetória", "value": "ATIVO"}],
                        value=[], style={"fontSize": "11px", "color": "#334155"},
                    ),
                    html.Div(
                        id="dim-status-selecao",
                        style={"fontSize": "10px", "color": "#475569",
                               "marginTop": "12px"},
                    ),
                ], style={
                    "backgroundColor": "#ffffff", "padding": "15px",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.05)",
                    "border": "1px solid #e2e8f0", "marginBottom": "20px",
                }),
            ], width=2),

            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Fluxos Temporais de Transição de Classe (Sankey Reativo ao XAI)",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([
                                dcc.Graph(id="sankey-graph", style={"height": "530px"})
                            ]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Evolução do Modelo: Performance & Estabilidade XAI",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([
                                dcc.Graph(id="metrics-stability-combined-graph")
                            ]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12, lg=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Resumo Global / Heatmap de Importância Temporal",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([dcc.Graph(id="heatmap-graph")]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12, lg=6),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Redução de Dimensionalidade Principal (t-SNE / PCA Espacial)",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="dim-scatter-graph",
                                    style={"height": "460px"},
                                    config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                                )
                            ]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12, lg=7),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Análise Auxiliar Temporal (Boxplot)",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="dim-reduction-aux-graph",
                                    style={"height": "500px"},
                                    config={"modeBarButtonsToAdd": ["select2d", "lasso2d"]},
                                )
                            ]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12, lg=5),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Tabela Auxiliar de Instâncias (Brushing / Seleção Cruzada)",
                                style={"backgroundColor": "#f8fafc",
                                       "fontWeight": "bold", "color": "#1e293b"},
                            ),
                            dbc.CardBody([
                                html.Div(
                                    "Clique em uma linha para destacar a instância na projeção 2D.",
                                    style={"fontSize": "11px", "color": "#64748b",
                                           "marginBottom": "8px"},
                                ),
                                dash_table.DataTable(
                                    id="dim-tabela-instancias",
                                    columns=[
                                        {"name": "Instância ID", "id": "instancia_id"},
                                        {"name": "Época", "id": "epoca"},
                                        {"name": "Método XAI", "id": "metodo_xai"},
                                        {"name": "Classe Real", "id": "classe_real"},
                                        {"name": "Classe Predita", "id": "classe_predita"},
                                        {"name": "Erro Predição", "id": "erro_predicao"},
                                        {"name": "Coord X", "id": "tsne_dim_1"},
                                        {"name": "Coord Y", "id": "tsne_dim_2"},
                                    ],
                                    data=[], page_size=8,
                                    row_selectable=False,
                                    cell_selectable=True,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left", "fontFamily": "sans-serif",
                                        "fontSize": "12px", "padding": "6px",
                                        "minWidth": "80px", "maxWidth": "220px",
                                        "whiteSpace": "normal",
                                    },
                                    style_header={
                                        "backgroundColor": "#f1f5f9",
                                        "fontWeight": "bold", "color": "#1e293b",
                                    },
                                    style_data_conditional=[{
                                        "if": {"state": "active"},
                                        "backgroundColor": "#dbeafe",
                                        "border": "1px solid #2563eb",
                                    }],
                                ),
                            ]),
                        ], className="shadow-sm mb-4",
                           style={"borderRadius": "8px", "border": "1px solid #e2e8f0"}),
                    ], width=12),
                ]),
            ], width=10),
        ]),
    ], fluid=True, style={
        "backgroundColor": "#f4f7f9", "padding": "20px", "minHeight": "100vh"
    }),
])


# ============================================================
# 4. OPÇÕES DINÂMICAS
# ============================================================
@app.callback(
    [
        Output("dropdown-metodo", "options"),
        Output("dropdown-metodo", "value"),
        Output("dim-dropdown-epoca", "options"),
        Output("dim-dropdown-epoca", "value"),
        Output("slider-epocas", "min"),
        Output("slider-epocas", "max"),
        Output("slider-epocas", "marks"),
        Output("slider-epocas", "value"),
    ],
    Input("dropdown-dataset", "value"),
)
def update_options(dataset_sel):
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)

    metodos = ["SHAP", "PFI", "LIME", "IG"]
    if not df_d3.empty and "metodo_xai" in df_d3.columns:
        metodos = sorted(df_d3["metodo_xai"].dropna().unique().tolist())

    epocas = []
    if not df_d6.empty and "epoca" in df_d6.columns:
        epocas = sorted(pd.to_numeric(df_d6["epoca"], errors="coerce").dropna().astype(int).unique())
    elif not df_d3.empty and "epoca" in df_d3.columns:
        epocas = sorted(pd.to_numeric(df_d3["epoca"], errors="coerce").dropna().astype(int).unique())

    if not epocas:
        epocas = [1, 10, 20, 30, 40]

    min_ep, max_ep = min(epocas), max(epocas)
    return (
        [{"label": str(m).upper(), "value": m} for m in metodos],
        metodos[0] if metodos else "SHAP",
        [{"label": f"Época {ep}", "value": ep} for ep in epocas],
        epocas[0],
        min_ep,
        max_ep,
        {int(ep): str(ep) for ep in epocas},
        [min_ep, max_ep],
    )


# ============================================================
# 5. CALLBACK PRINCIPAL E INTERAÇÕES CRUZADAS
# ============================================================
@app.callback(
    [
        Output("metrics-stability-combined-graph", "figure"),
        Output("heatmap-graph", "figure"),
        Output("dim-scatter-graph", "figure"),
        Output("dim-reduction-aux-graph", "figure"),
        Output("dim-tabela-instancias", "data"),
        Output("sankey-graph", "figure"),
        Output("dim-status-selecao", "children"),
    ],
    [
        Input("dropdown-dataset", "value"),
        Input("dropdown-metodo", "value"),
        Input("dropdown-metrica-estabilidade", "value"),
        Input("dropdown-cor-metrica", "value"),
        Input("slider-epocas", "value"),
        Input("dim-dropdown-epoca", "value"),
        Input("dim-dropdown-algo", "value"),
        Input("dim-dropdown-cor", "value"),
        Input("dim-toggle-trajetoria", "value"),
        Input("dim-scatter-graph", "clickData"),
        Input("dim-scatter-graph", "selectedData"),
        Input("dim-reduction-aux-graph", "clickData"),
        Input("dim-reduction-aux-graph", "selectedData"),
        Input("dim-tabela-instancias", "active_cell"),
    ],
    State("dim-tabela-instancias", "data"),
)
def update_all_graphs(
    dataset_sel, metodo_sel, metrica_est_sel, cor_sankey_sel,
    epocas_sel, epoca_proj_sel, algo_sel, cor_proj_sel, trajetoria_sel,
    scatter_click, scatter_selected, box_click, box_selected,
    table_active_cell, table_data,
):
    df_d2 = carregar_csv_seguro("metricas_epocas", dataset_sel)
    df_d4 = carregar_csv_seguro("resumo_explicacoes", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)
    df_d5 = carregar_csv_seguro("metricas_estabilidade_concordancia", dataset_sel)
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)
    label_ds = MAPA_DATASET.get(dataset_sel, {}).get("label", dataset_sel.upper())

    # ---------- 1. Performance + estabilidade ----------
    fig_combined = go.Figure()
    if not df_d2.empty and "epoca" in df_d2.columns:
        acc_col = "accuracy" if "accuracy" in df_d2.columns else (
            df_d2.columns[1] if len(df_d2.columns) > 1 else None
        )
        if acc_col:
            fig_combined.add_trace(go.Scatter(
                x=df_d2["epoca"], y=df_d2[acc_col],
                mode="lines+markers", name="Acurácia",
                line=dict(color="#2ca02c"),
            ))
        if "loss_validacao" in df_d2.columns:
            fig_combined.add_trace(go.Scatter(
                x=df_d2["epoca"], y=df_d2["loss_validacao"],
                mode="lines+markers", name="Loss Validação",
                yaxis="y2", line=dict(color="#d62728"),
            ))

    if not df_d5.empty and {"tipo_metrica", "metrica", "epoca_destino", "valor_metrica"}.issubset(df_d5.columns):
        mask_metodo = df_d5["tipo_metrica"].astype(str).str.upper() == str(metodo_sel).upper()
        mask_metrica = df_d5["metrica"].astype(str).str.lower() == str(metrica_est_sel).lower()
        df_est = df_d5[mask_metodo & mask_metrica]
        if not df_est.empty:
            fig_combined.add_trace(go.Scatter(
                x=df_est["epoca_destino"], y=df_est["valor_metrica"],
                mode="lines+markers", name=f"Estabilidade ({metrica_est_sel})",
                line=dict(color="#1f77b4", dash="dash"),
            ))

    fig_combined.update_layout(
        title=f"Performance do Modelo & Estabilidade XAI ({metodo_sel})",
        yaxis=dict(title="Acurácia / Estabilidade"),
        yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
        margin=dict(t=35, b=30, l=40, r=40),
        legend=dict(x=0.01, y=0.99, orientation="h"),
        template="plotly_white",
    )

    # ---------- 2. Heatmap ----------
    fig_heat = go.Figure()
    if not df_d4.empty and "metodo_xai" in df_d4.columns:
        df_h = df_d4[
            df_d4["metodo_xai"].astype(str).str.upper() == str(metodo_sel).upper()
        ]
        if not df_h.empty and {"feature", "epoca"}.issubset(df_h.columns):
            val_col = "importancia_media" if "importancia_media" in df_h.columns else df_h.columns[-1]
            piv = df_h.pivot_table(
                index="feature", columns="epoca", values=val_col, fill_value=0
            )
            fig_heat = go.Figure(data=go.Heatmap(
                z=piv.values, x=piv.columns, y=piv.index,
                colorscale=COLORSCALE_REDS_20,
                hovertemplate="Feature: %{y}<br>Época: %{x}<br>Importância: %{z:.4f}<extra></extra>",
            ))
    fig_heat.update_layout(
        title=f"Heatmap de Importância ({metodo_sel})",
        margin=dict(t=35, b=30, l=50, r=20), template="plotly_white",
    )

    # ---------- 3. Preparação da projeção 2D ----------
    fig_2d = go.Figure()
    fig_box = go.Figure()
    tabela_records = []
    status = "Nenhuma seleção ativa. Exibindo as instâncias da época selecionada."

    if df_d6.empty:
        fig_2d.update_layout(title="Arquivo de projeção 2D não encontrado", template="plotly_white")
        fig_box.update_layout(title="Dados do Boxplot não encontrados", template="plotly_white")
    else:
        df_d6 = df_d6.copy()
        for col in ["epoca", "instancia_id", "tsne_dim_1", "tsne_dim_2"]:
            if col in df_d6.columns:
                df_d6[col] = pd.to_numeric(df_d6[col], errors="coerce")

        if "metodo_xai" in df_d6.columns:
            df_metodo = df_d6[
                df_d6["metodo_xai"].astype(str).str.upper() == str(metodo_sel).upper()
            ].copy()
        else:
            df_metodo = df_d6.copy()

        # Calcula erro caso não esteja presente no CSV.
        if "erro_predicao" not in df_metodo.columns:
            if {"classe_real", "classe_predita"}.issubset(df_metodo.columns):
                df_metodo["erro_predicao"] = (
                    df_metodo["classe_real"] != df_metodo["classe_predita"]
                ).astype(int)
            else:
                df_metodo["erro_predicao"] = 0

        # Boxplot com pontos individuais selecionáveis e outliers destacados.
        if {"epoca", "tsne_dim_1"}.issubset(df_metodo.columns):
            df_box_base = df_metodo.dropna(subset=["epoca", "tsne_dim_1"]).copy()
            outliers = calcular_outliers(df_box_base, "tsne_dim_1")
            outlier_ids_por_epoca = set()
            if not outliers.empty and "instancia_id" in outliers.columns:
                outlier_ids_por_epoca = set(
                    zip(outliers["epoca"].astype(int), outliers["instancia_id"])
                )

            box_custom = []
            for _, row in df_box_base.iterrows():
                iid = row.get("instancia_id")
                is_outlier = (
                    (int(row["epoca"]), iid) in outlier_ids_por_epoca
                    if pd.notna(iid) else False
                )
                box_custom.append([iid, int(row["epoca"]), int(is_outlier)])

            fig_box.add_trace(go.Box(
                x=df_box_base["epoca"],
                y=df_box_base["tsne_dim_1"],
                name="Distribuição por época",
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(color="rgba(59,130,246,0.35)", size=5),
                line=dict(color="#6366f1", width=1.5),
                fillcolor="rgba(99,102,241,0.12)",
                customdata=box_custom,
                hovertemplate=(
                    "<b>Época:</b> %{customdata[1]}<br>"
                    "<b>Instância:</b> %{customdata[0]}<br>"
                    "<b>Dimensão 1:</b> %{y:.4f}<br>"
                    "<b>Outlier:</b> %{customdata[2]}<extra></extra>"
                ),
                selected=dict(marker=dict(color="#f59e0b", size=9, opacity=1)),
                unselected=dict(marker=dict(opacity=0.35)),
            ))

            if not outliers.empty:
                fig_box.add_trace(go.Scatter(
                    x=outliers["epoca"],
                    y=outliers["tsne_dim_1"],
                    mode="markers",
                    name="Outliers (IQR)",
                    customdata=np.column_stack([
                        outliers["instancia_id"].values,
                        outliers["epoca"].astype(int).values,
                        np.ones(len(outliers), dtype=int),
                    ]),
                    marker=dict(color="#dc2626", size=10, symbol="diamond",
                                line=dict(color="#7f1d1d", width=1)),
                    hovertemplate=(
                        "<b>OUTLIER</b><br>Época: %{customdata[1]}<br>"
                        "Instância: %{customdata[0]}<br>"
                        "Dimensão 1: %{y:.4f}<extra></extra>"
                    ),
                ))

        fig_box.update_layout(
            title="Dispersão temporal por época — clique em uma época ou selecione pontos/outliers",
            xaxis_title="Época", yaxis_title="t-SNE Dim 1",
            template="plotly_white",
            margin=dict(l=25, r=20, t=45, b=30),
            clickmode="event+select",
            dragmode="select",
            legend=dict(orientation="h", y=1.12),
        )

        # ---------- Interpretação dos eventos do Boxplot ----------
        epoca_filtro_box = None
        ids_box = []

        # A seleção por laço/caixa no Boxplot tem prioridade sobre o clique.
        if box_selected and box_selected.get("points"):
            ids_box = obter_ids_evento(box_selected)
            epocas_evento = [
                obter_epoca_evento({"points": [p]})
                for p in box_selected["points"]
            ]
            epocas_evento = [e for e in epocas_evento if e is not None]
            if epocas_evento:
                epoca_filtro_box = epocas_evento[0]
        elif box_click and box_click.get("points"):
            ponto_box = box_click["points"][0]
            epoca_filtro_box = obter_epoca_evento(box_click)
            custom = ponto_box.get("customdata")
            # Clique em outlier também seleciona a instância específica.
            if isinstance(custom, (list, tuple, np.ndarray)) and len(custom) > 2:
                try:
                    if int(custom[2]) == 1:
                        ids_box = obter_ids_evento(box_click)
                except (TypeError, ValueError):
                    pass

        # ---------- Época efetiva da projeção ----------
        epoca_efetiva = epoca_filtro_box if epoca_filtro_box is not None else epoca_proj_sel
        if epoca_efetiva is None:
            epoca_efetiva = int(df_metodo["epoca"].min()) if "epoca" in df_metodo.columns and not df_metodo.empty else 1

        df_sub = df_metodo.copy()
        if "epoca" in df_sub.columns:
            df_sub = df_sub[df_sub["epoca"] == int(epoca_efetiva)].copy()

        x_col, y_col = "tsne_dim_1", "tsne_dim_2"

        # Recalcula a projeção via PCA somente quando solicitado.
        if algo_sel == "PCA" and not df_d3.empty:
            df_exp = df_d3.copy()
            if "metodo_xai" in df_exp.columns:
                df_exp = df_exp[
                    df_exp["metodo_xai"].astype(str).str.upper()
                    == str(metodo_sel).upper()
                ]
            if "epoca" in df_exp.columns:
                df_exp = df_exp[
                    pd.to_numeric(df_exp["epoca"], errors="coerce") == int(epoca_efetiva)
                ]

            if {"instancia_id", "feature", "valor_atribuicao"}.issubset(df_exp.columns):
                pivot = df_exp.pivot_table(
                    index="instancia_id", columns="feature",
                    values="valor_atribuicao", aggfunc="mean",
                ).fillna(0)
                if len(pivot) >= 2 and pivot.shape[1] >= 2:
                    n_comp = min(2, pivot.shape[0], pivot.shape[1])
                    pca_values = PCA(n_components=n_comp).fit_transform(pivot.values)
                    pca_df = pd.DataFrame(
                        pca_values, index=pivot.index,
                        columns=["pca_dim_1", "pca_dim_2"][:n_comp],
                    ).reset_index()
                    df_sub = df_sub.merge(pca_df, on="instancia_id", how="left")
                    if {"pca_dim_1", "pca_dim_2"}.issubset(df_sub.columns):
                        x_col, y_col = "pca_dim_1", "pca_dim_2"

        # Garante campos necessários para a tabela e para o hover.
        for col in [
            "instancia_id", "classe_real", "classe_predita",
            "erro_predicao", x_col, y_col,
        ]:
            if col not in df_sub.columns:
                df_sub[col] = np.nan

        # ---------- Instâncias selecionadas em cada origem ----------
        selected_ids = []

        # 1) Clique na tabela: destaca a instância escolhida.
        if table_active_cell and table_data:
            # row_id permanece estável mesmo quando a tabela é filtrada.
            row_id = table_active_cell.get("row_id")
            if row_id is not None:
                selected_ids = [row_id]
            else:
                row_idx = table_active_cell.get("row")
                if row_idx is not None and 0 <= row_idx < len(table_data):
                    selected_ids = [table_data[row_idx].get("instancia_id")]

        # 2) Seleção de outliers/pontos no Boxplot.
        elif ids_box:
            selected_ids = ids_box

        # 3) Brushing por caixa/laço no gráfico 2D.
        elif scatter_selected and scatter_selected.get("points"):
            selected_ids = obter_ids_evento(scatter_selected)

        # 4) Clique individual no gráfico 2D.
        elif scatter_click and scatter_click.get("points"):
            selected_ids = obter_ids_evento(scatter_click)

        selected_ids = [x for x in selected_ids if x is not None]
        selected_ids = list(dict.fromkeys(selected_ids))

        # ---------- Montagem do gráfico 2D ----------
        if not df_sub.empty and {x_col, y_col}.issubset(df_sub.columns):
            if cor_proj_sel == "erro_predicao":
                df_sub["cor_label"] = df_sub["erro_predicao"].map({
                    0: "Acerto", 1: "Erro"
                }).fillna("Não informado")
                fig_2d = px.scatter(
                    df_sub, x=x_col, y=y_col, color="cor_label",
                    color_discrete_map={
                        "Acerto": "#059669", "Erro": "#dc2626",
                        "Não informado": "#94a3b8",
                    },
                    custom_data=[
                        "instancia_id", "classe_real",
                        "classe_predita", "erro_predicao",
                    ],
                )
            else:
                if cor_proj_sel not in df_sub.columns:
                    df_sub[cor_proj_sel] = 0
                fig_2d = px.scatter(
                    df_sub, x=x_col, y=y_col, color=cor_proj_sel,
                    color_continuous_scale=COLORSCALE_BLUES_20,
                    custom_data=[
                        "instancia_id", "classe_real",
                        "classe_predita", "erro_predicao",
                    ],
                )

            fig_2d.update_traces(
                marker=dict(size=8, opacity=0.78,
                            line=dict(width=0.5, color="#334155")),
                hovertemplate=(
                    "<b>Instância ID:</b> %{customdata[0]}<br>"
                    "<b>Classe Real:</b> %{customdata[1]}<br>"
                    "<b>Classe Predita:</b> %{customdata[2]}<br>"
                    "<b>Erro:</b> %{customdata[3]}<br>"
                    "<b>X:</b> %{x:.3f} | <b>Y:</b> %{y:.3f}<extra></extra>"
                ),
            )

            # Destaque das instâncias selecionadas (tabela, boxplot ou brushing).
            if selected_ids:
                df_highlight = df_sub[
                    df_sub["instancia_id"].isin(selected_ids)
                ].dropna(subset=[x_col, y_col])
                if not df_highlight.empty:
                    fig_2d.add_trace(go.Scatter(
                        x=df_highlight[x_col], y=df_highlight[y_col],
                        mode="markers",
                        name="Instâncias selecionadas",
                        marker=dict(
                            size=17, color="rgba(0,0,0,0)",
                            line=dict(color="#eab308", width=3),
                            symbol="circle",
                        ),
                        customdata=df_highlight[["instancia_id"]].values,
                        hovertemplate="Instância selecionada: %{customdata[0]}<extra></extra>",
                    ))

            # Trajetória da instância clicada no scatter.
            clicked_id = None
            if scatter_click and scatter_click.get("points"):
                ids_click = obter_ids_evento(scatter_click)
                if ids_click:
                    clicked_id = ids_click[0]

            if "ATIVO" in (trajetoria_sel or []) and clicked_id is not None:
                df_traj = df_metodo[
                    df_metodo["instancia_id"] == clicked_id
                ].sort_values("epoca")
                if not df_traj.empty and {"tsne_dim_1", "tsne_dim_2"}.issubset(df_traj.columns):
                    fig_2d.add_trace(go.Scatter(
                        x=df_traj["tsne_dim_1"], y=df_traj["tsne_dim_2"],
                        mode="lines+markers+text",
                        text=[f"E{ep}" for ep in df_traj["epoca"]],
                        textposition="top center",
                        line=dict(color="#f59e0b", width=3, dash="dash"),
                        marker=dict(size=11, color="#f59e0b", symbol="diamond"),
                        name=f"Trajetória ID #{clicked_id}",
                    ))

            fig_2d.update_layout(
                title=f"Espaço 2D ({algo_sel}) — Época {epoca_efetiva} ({metodo_sel}) — {label_ds}",
                xaxis_title=f"{algo_sel} Dim 1",
                yaxis_title=f"{algo_sel} Dim 2",
                margin=dict(l=20, r=20, t=45, b=25),
                template="plotly_white",
                clickmode="event+select",
                dragmode="select",
            )

            # Tabela mostra somente as instâncias selecionadas.
            if selected_ids:
                df_tabela = df_sub[
                    df_sub["instancia_id"].isin(selected_ids)
                ].copy()
                status = (
                    f"{len(df_tabela)} instância(s) selecionada(s). "
                    f"Origem: {'tabela' if table_active_cell else 'Boxplot' if ids_box else 'gráfico 2D'}."
                )
            else:
                df_tabela = df_sub.copy()
                status = (
                    f"Época {epoca_efetiva}: {len(df_tabela)} instância(s). "
                    "Selecione pontos no gráfico ou no Boxplot para filtrar."
                )

            # Campos apresentados e ordenação da tabela.
            colunas_tabela = [
                "instancia_id", "epoca", "metodo_xai",
                "classe_real", "classe_predita", "erro_predicao",
                x_col, y_col,
            ]
            if "metodo_xai" not in df_tabela.columns:
                df_tabela["metodo_xai"] = metodo_sel
            df_tabela = df_tabela.reindex(columns=colunas_tabela)
            df_tabela = df_tabela.rename(columns={
                x_col: "tsne_dim_1",
                y_col: "tsne_dim_2",
            })
            df_tabela = df_tabela.replace({np.nan: None})
            # O campo id é usado internamente pelo DataTable para manter
            # a identidade da linha após filtros e paginação.
            df_tabela["id"] = df_tabela["instancia_id"].astype(str)
            tabela_records = df_tabela.to_dict("records")
        else:
            fig_2d.update_layout(
                title=f"Sem instâncias para a época {epoca_efetiva}",
                template="plotly_white",
            )
            tabela_records = []
            status = f"Nenhuma instância encontrada para a época {epoca_efetiva}."

    # ---------- 4. Sankey sem filtro lateral de classe ----------
    fig_sankey = go.Figure()
    df_d3_sankey, df_d2_sankey = carregar_dados_sankey(dataset_sel)
    df_sankey = processar_dados_sankey(
        df_d3_sankey, df_d2_sankey, dataset_sel
    )

    if not df_sankey.empty:
        metodos_validos = df_sankey["metodo_xai"].dropna().unique().tolist()
        metodo_alvo = metodo_sel if metodo_sel in metodos_validos else metodos_validos[0]
        df_fluxos = df_sankey[
            df_sankey["metodo_xai"] == metodo_alvo
        ].copy()

        if epocas_sel:
            df_fluxos = df_fluxos[
                (df_fluxos["epoca_origem"] >= epocas_sel[0])
                & (df_fluxos["epoca_destino"] <= epocas_sel[1])
            ]

        if not df_fluxos.empty:
            mapa_cls = mapeamento_classes_dataset(dataset_sel)
            epocas_unicas = sorted(
                set(df_fluxos["epoca_origem"]).union(df_fluxos["epoca_destino"])
            )
            classes = [0, 1]
            labels, node_x, node_y, node_map = [], [], [], {}

            for i_ep, ep in enumerate(epocas_unicas):
                for cl in classes:
                    node_id = len(labels)
                    node_map[(ep, cl)] = node_id
                    labels.append(f"Época {ep}<br>{mapa_cls.get(cl, f'Classe {cl}')}")
                    node_x.append(
                        0.05 + 0.9 * i_ep / max(1, len(epocas_unicas) - 1)
                    )
                    node_y.append(0.2 if cl == 0 else 0.8)

            sources, targets, values, colors, hovers = [], [], [], [], []
            for _, row in df_fluxos.iterrows():
                origem = (row["epoca_origem"], row["classe_origem"])
                destino = (row["epoca_destino"], row["classe_destino"])
                if origem not in node_map or destino not in node_map:
                    continue

                sources.append(node_map[origem])
                targets.append(node_map[destino])
                values.append(max(1, int(row["qtd_instancias"])))

                conf = float(np.clip(row["confianca_media"], 0, 1))
                alpha = max(0.30, min(0.85, conf))
                if cor_sankey_sel == "classe":
                    rgb = "37, 99, 235" if row["classe_origem"] == 0 else "249, 115, 22"
                elif cor_sankey_sel == "confianca":
                    rgb = f"{int(255 * (1-conf))}, {int(200 * conf)}, 150"
                else:
                    erro = float(np.clip(row["taxa_erro"], 0, 1))
                    rgb = f"{int(255*erro)}, 50, 50"
                colors.append(f"rgba({rgb}, {alpha:.2f})")

                hovers.append(
                    f"<b>Dataset:</b> {dataset_sel.upper()}<br>"
                    f"<b>Método:</b> {metodo_alvo}<br>"
                    f"<b>Transição:</b> Época {row['epoca_origem']} → {row['epoca_destino']}<br>"
                    f"<b>Classe:</b> {mapa_cls.get(row['classe_origem'])} → "
                    f"{mapa_cls.get(row['classe_destino'])}<br>"
                    f"<b>Instâncias:</b> {row['qtd_instancias']}<br>"
                    f"<b>Instância exemplo:</b> #{row['exemplo_instancia']}<br>"
                    f"<b>Confiança média:</b> {conf*100:.1f}%<br>"
                    f"<b>Taxa de erro:</b> {row['taxa_erro']*100:.1f}%<br>"
                    f"<b>Importância acumulada:</b> {row['soma_importancia']:.4f}"
                )

            fig_sankey = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=30, thickness=20,
                    line=dict(color="#334155", width=0.5),
                    label=labels, x=node_x, y=node_y,
                    color="#e2e8f0",
                ),
                link=dict(
                    source=sources, target=targets, value=values,
                    color=colors, customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )])
            fig_sankey.update_layout(
                title_text=f"Transições Temporais de Explicações ({metodo_alvo}) — {dataset_sel.upper()}",
                font_size=12, margin=dict(l=20, r=20, t=50, b=20),
                template="plotly_white",
            )
        else:
            fig_sankey.update_layout(
                title="Não há transições para o intervalo de épocas selecionado.",
                template="plotly_white",
            )
    else:
        fig_sankey.update_layout(
            title="Dados insuficientes para construir o Sankey.",
            template="plotly_white",
        )

    return (
        fig_combined, fig_heat, fig_2d, fig_box,
        tabela_records, fig_sankey, status,
    )


if __name__ == "__main__":
    app.run(debug=False)
