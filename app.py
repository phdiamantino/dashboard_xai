
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
        try:
            zip_path = "dados_xai.zip"
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, zip_path, quiet=False)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")

            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".csv"):
                        src = os.path.join(root, file)
                        dst = file
                        if src != dst and not os.path.exists(dst):
                            shutil.move(src, dst)

            print("[Sucesso] Dados preparados.")

        except Exception as e:
            print(f"[Erro] Preparação dos dados: {e}")


preparar_dados()


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

COLORSCALE_REDS_20 = [
    [0.0, "rgb(254,224,210)"],
    [0.25, "rgb(252,146,114)"],
    [0.5, "rgb(251,106,74)"],
    [0.75, "rgb(222,45,38)"],
    [1.0, "rgb(165,15,21)"]
]

COLORSCALE_BLUES_20 = [
    [0.0, "rgb(222,235,247)"],
    [0.25, "rgb(158,202,225)"],
    [0.5, "rgb(66,146,198)"],
    [0.75, "rgb(33,113,181)"],
    [1.0, "rgb(8,48,107)"]
]

MAPA_DATASET = {
    "breast_cancer": {
        "arquivo_d3": "explicacoes_instancias_breast_cancer.csv",
        "label": "Breast Cancer Wisconsin",
        "modelo_id": "MLP_5L_BREAST_CANCER"
    },
    "wine_quality": {
        "arquivo_d3": "explicacoes_instancias_wine_quality.csv",
        "label": "Wine Quality (Red/White)",
        "modelo_id": "MLP_5L_WINE_QUALITY"
    },
    "german_credit": {
        "arquivo_d3": "explicacoes_instancias_german_credit.csv",
        "label": "German Credit (Cost Risk)",
        "modelo_id": "MLP_5L_GERMAN_CREDIT"
    }
}


def normalizar_id(valor):
    """Padroniza IDs para comparação entre arquivos CSV."""
    if pd.isna(valor):
        return None

    try:
        numero = float(valor)
        if numero.is_integer():
            return str(int(numero))
    except (ValueError, TypeError):
        pass

    return str(valor).strip()


def carregar_csv_seguro(nome_base, dataset_sel):
    padroes = [
        f"**/{nome_base}_{dataset_sel}.csv",
        f"**/{nome_base}.csv"
    ]

    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)

        if arquivos:
            try:
                df = pd.read_csv(arquivos[0])

                df.columns = (
                    df.columns.astype(str)
                    .str.replace("\ufeff", "")
                    .str.replace("ï»¿", "")
                    .str.strip()
                    .str.lower()
                )

                if "dataset_name" in df.columns:
                    df = df[
                        df["dataset_name"].astype(str).str.lower()
                        == dataset_sel.lower()
                    ]

                return df.copy()

            except Exception as e:
                print(f"[Erro] Leitura de {arquivos[0]}: {e}")

    return pd.DataFrame()


def carregar_dados_sankey(dataset_sel):
    info = MAPA_DATASET.get(dataset_sel, {})

    f_d2 = f"metricas_epocas_{dataset_sel}.csv"
    f_d3 = info.get("arquivo_d3", "explicacoes_instancias.csv")

    if not os.path.exists(f_d2):
        f_d2 = "metricas_epocas.csv"

    if not os.path.exists(f_d3):
        f_d3 = f"explicacoes_instancias_{dataset_sel}.csv"

    if not os.path.exists(f_d3):
        f_d3 = "explicacoes_instancias.csv"

    try:
        df_d2 = pd.read_csv(f_d2) if os.path.exists(f_d2) else pd.DataFrame()
        df_d3 = pd.read_csv(f_d3) if os.path.exists(f_d3) else pd.DataFrame()

        for df in [df_d2, df_d3]:
            if not df.empty:
                df.columns = (
                    df.columns.astype(str)
                    .str.strip()
                    .str.lower()
                )

    except Exception as e:
        print(f"[Erro] Sankey: {e}")
        return pd.DataFrame(), pd.DataFrame()

    modelo = info.get("modelo_id")

    for df in [df_d2, df_d3]:
        if not df.empty and "modelo_id" in df.columns:
            if modelo and (df["modelo_id"] == modelo).any():
                df.drop(
                    df[df["modelo_id"] != modelo].index,
                    inplace=True
                )

    return df_d3, df_d2


# ============================================================
# 2. PROCESSAMENTO DO SANKEY
# ============================================================

def processar_dados_sankey(df_d3, df_d2, dataset_sel):

    if df_d3.empty or df_d2.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    df_d3 = df_d3.copy()
    df_d2 = df_d2.copy()

    if "instancia_id" not in df_d3.columns:
        for col in ["instance_id", "id", "sample_id"]:
            if col in df_d3.columns:
                df_d3["instancia_id"] = df_d3[col]
                break

    if "instancia_id" not in df_d3.columns:
        return pd.DataFrame(), pd.DataFrame(), []

    if "importancia_absoluta" not in df_d3.columns:
        if "valor_atribuicao" in df_d3.columns:
            df_d3["importancia_absoluta"] = (
                df_d3["valor_atribuicao"].abs()
            )
        else:
            df_d3["importancia_absoluta"] = 1.0

    df_d3["epoca"] = pd.to_numeric(
        df_d3["epoca"], errors="coerce"
    )

    df_d3 = df_d3.dropna(subset=["epoca"])
    df_d3["epoca"] = df_d3["epoca"].astype(int)

    agg_dict = {
        "importancia_absoluta": "sum"
    }

    for col in ["classe_real", "classe_predita"]:
        if col in df_d3.columns:
            agg_dict[col] = "first"

    df_inst = df_d3.groupby(
        ["metodo_xai", "epoca", "instancia_id"]
    ).agg(agg_dict).reset_index()

    df_inst.rename(
        columns={"importancia_absoluta": "importancia_total"},
        inplace=True
    )

    acc_col = (
        "accuracy" if "accuracy" in df_d2.columns
        else df_d2.columns[1]
    )

    loss_col = (
        "loss_validacao"
        if "loss_validacao" in df_d2.columns
        else acc_col
    )

    df_metricas = df_d2[
        ["epoca", acc_col, loss_col]
    ].copy()

    df_metricas.columns = [
        "epoca", "accuracy", "loss_validacao"
    ]

    df_metricas["epoca"] = pd.to_numeric(
        df_metricas["epoca"], errors="coerce"
    )

    df_metricas = df_metricas.dropna(subset=["epoca"])
    df_metricas["epoca"] = df_metricas["epoca"].astype(int)

    df_inst = df_inst.merge(
        df_metricas,
        on="epoca",
        how="left"
    )

    if "classe_real" not in df_inst.columns:
        prop = {
            "breast_cancer": 0.37,
            "wine_quality": 0.22,
            "german_credit": 0.70
        }.get(dataset_sel, 0.5)

        rng = np.random.default_rng(
            abs(hash(dataset_sel)) % (2**32)
        )

        ids = df_inst["instancia_id"].unique()

        mapa = {
            inst: int(rng.random() < prop)
            for inst in ids
        }

        df_inst["classe_real"] = (
            df_inst["instancia_id"].map(mapa)
        )

    if "classe_predita" not in df_inst.columns:
        df_inst["classe_predita"] = df_inst["classe_real"]

        rng = np.random.default_rng(42)

        erros = (
            rng.random(len(df_inst))
            > df_inst["accuracy"].fillna(0.9)
        )

        df_inst.loc[erros, "classe_predita"] = (
            1 - df_inst.loc[erros, "classe_real"]
        )

    df_inst["confianca_media"] = df_inst["accuracy"]

    df_inst["acerto"] = (
        df_inst["classe_predita"]
        == df_inst["classe_real"]
    ).astype(int)

    df_inst["taxa_erro"] = 1 - df_inst["acerto"]

    epocas = sorted(df_inst["epoca"].unique())
    metodos = df_inst["metodo_xai"].dropna().unique()

    transicoes = []

    for metodo in metodos:

        df_m = df_inst[
            df_inst["metodo_xai"] == metodo
        ]

        for i in range(len(epocas) - 1):

            ep_o = epocas[i]
            ep_d = epocas[i + 1]

            orig = df_m[
                df_m["epoca"] == ep_o
            ].set_index("instancia_id")

            dest = df_m[
                df_m["epoca"] == ep_d
            ].set_index("instancia_id")

            comuns = orig.index.intersection(dest.index)

            for inst in comuns:

                ro = orig.loc[inst]
                rd = dest.loc[inst]

                if isinstance(ro, pd.DataFrame):
                    ro = ro.iloc[0]

                if isinstance(rd, pd.DataFrame):
                    rd = rd.iloc[0]

                transicoes.append({
                    "metodo_xai": metodo,
                    "epoca_origem": int(ep_o),
                    "epoca_destino": int(ep_d),
                    "instancia_id": inst,
                    "classe_origem": int(ro["classe_predita"]),
                    "classe_destino": int(rd["classe_predita"]),
                    "classe_real": int(ro["classe_real"]),
                    "confianca_media": (
                        ro["confianca_media"]
                        + rd["confianca_media"]
                    ) / 2,
                    "taxa_erro": (
                        ro["taxa_erro"] + rd["taxa_erro"]
                    ) / 2,
                    "importancia_media": (
                        ro["importancia_total"]
                        + rd["importancia_total"]
                    ) / 2
                })

    df_trans = pd.DataFrame(transicoes)

    if df_trans.empty:
        return pd.DataFrame(), df_trans, epocas

    # IMPORTANTE:
    # Cada fluxo mantém a lista dos IDs individuais.
    df_sankey = df_trans.groupby(
        [
            "metodo_xai",
            "epoca_origem",
            "epoca_destino",
            "classe_origem",
            "classe_destino"
        ]
    ).agg(
        qtd_instancias=("instancia_id", "count"),
        instancias_ids=(
            "instancia_id",
            lambda x: list(pd.unique(x))
        ),
        soma_importancia=("importancia_media", "sum"),
        confianca_media=("confianca_media", "mean"),
        taxa_erro=("taxa_erro", "mean"),
        importancia_media=("importancia_media", "mean"),
        exemplo_instancia=("instancia_id", "first"),
        classe_real_dominante=(
            "classe_real",
            lambda x: int(x.mode().iloc[0])
        )
    ).reset_index()

    return df_sankey, df_trans, epocas


# ============================================================
# 3. APLICAÇÃO DASH
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server


app.layout = html.Div([

    dbc.NavbarSimple(
        brand="Dashboard Analítico XAI (Multi-Dataset & Diagnóstico Temporal)",
        color="dark",
        dark=True,
        fluid=True
    ),

    # ESTADO COMPARTILHADO ENTRE VISUALIZAÇÕES
    dcc.Store(
        id="estado-selecao",
        data={
            "instancias": [],
            "epoca": None,
            "origem": None,
            "transicao": None
        }
    ),

    dbc.Container([

        dbc.Row([

            # =================================================
            # SIDEBAR
            # =================================================

            dbc.Col([

                html.Div([

                    html.H5(
                        "Controles",
                        style={"fontWeight": "bold"}
                    ),

                    html.Label("Dataset:"),

                    dcc.Dropdown(
                        id="dropdown-dataset",
                        options=[
                            {"label": "Breast Cancer", "value": "breast_cancer"},
                            {"label": "Wine Quality", "value": "wine_quality"},
                            {"label": "German Credit", "value": "german_credit"}
                        ],
                        value="breast_cancer",
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Método XAI:"),

                    dcc.Dropdown(
                        id="dropdown-metodo",
                        options=[],
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Métrica Estabilidade:"),

                    dcc.Dropdown(
                        id="dropdown-metrica-estabilidade",
                        options=[
                            {"label": "Spearman", "value": "spearman"},
                            {"label": "MAD Absoluto", "value": "mad_importancia"},
                            {"label": "Variância", "value": "variancia_importancia"}
                        ],
                        value="spearman",
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Cores (Sankey):"),

                    dcc.Dropdown(
                        id="dropdown-cor-metrica",
                        options=[
                            {"label": "Classe Predita", "value": "classe"},
                            {"label": "Confiança", "value": "confianca"},
                            {"label": "Taxa de Erro", "value": "erro"}
                        ],
                        value="classe",
                        clearable=False
                    ),

                    html.Hr(),

                    html.Label("Épocas (Sankey):"),

                    dcc.RangeSlider(
                        id="slider-epocas",
                        min=1,
                        max=40,
                        step=None,
                        marks={1: "1", 40: "40"},
                        value=[1, 40]
                    ),

                    html.Hr(),

                    html.H6("Controles 2D"),

                    html.Label("Época 2D:"),

                    dcc.Dropdown(
                        id="dim-dropdown-epoca",
                        options=[],
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Algoritmo:"),

                    dcc.Dropdown(
                        id="dim-dropdown-algo",
                        options=[
                            {"label": "t-SNE", "value": "TSNE"},
                            {"label": "PCA", "value": "PCA"}
                        ],
                        value="TSNE",
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Colorir 2D Por:"),

                    dcc.Dropdown(
                        id="dim-dropdown-cor",
                        options=[
                            {"label": "Classe Predita", "value": "classe_predita"},
                            {"label": "Classe Real", "value": "classe_real"},
                            {"label": "Erro Predição", "value": "erro_predicao"}
                        ],
                        value="classe_predita",
                        clearable=False
                    ),

                    html.Br(),

                    dcc.Checklist(
                        id="dim-toggle-trajetoria",
                        options=[
                            {"label": " Trajetória", "value": "ATIVO"}
                        ],
                        value=[]
                    )

                ], style={
                    "backgroundColor": "#ffffff",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "border": "1px solid #e2e8f0"
                })

            ], width=2),

            # =================================================
            # CONTEÚDO PRINCIPAL
            # =================================================

            dbc.Col([

                # SANKEY

                dbc.Card([
                    dbc.CardHeader(
                        "Fluxos Temporais de Transição de Classe (Sankey Reativo ao XAI)"
                    ),
                    dbc.CardBody([
                        dcc.Graph(
                            id="sankey-graph",
                            style={"height": "530px"}
                        )
                    ])
                ], className="shadow-sm mb-4"),

                # PERFORMANCE E HEATMAP

                dbc.Row([

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Evolução do Modelo: Performance & Estabilidade XAI"
                            ),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="metrics-stability-combined-graph"
                                )
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=12, lg=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Resumo Global / Heatmap de Importância Temporal"
                            ),
                            dbc.CardBody([
                                dcc.Graph(id="heatmap-graph")
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=12, lg=6)

                ]),

                # REDUÇÃO 2D E BOXPLOT

                dbc.Row([

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Redução de Dimensionalidade Principal (t-SNE / PCA)"
                            ),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="dim-scatter-graph",
                                    style={"height": "460px"},
                                    config={
                                        "modeBarButtonsToAdd": [
                                            "drawrect",
                                            "drawlasso",
                                            "eraseshape"
                                        ]
                                    }
                                )
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=12, lg=7),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "Análise Auxiliar Temporal (Boxplot)"
                            ),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="dim-reduction-aux-graph",
                                    style={"height": "500px"},
                                    config={
                                        "modeBarButtonsToAdd": [
                                            "select2d",
                                            "lasso2d"
                                        ]
                                    }
                                )
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=12, lg=5)

                ]),

                # TABELA

                dbc.Card([
                    dbc.CardHeader(
                        "Tabela Auxiliar de Instâncias — Seleções Coordenadas"
                    ),
                    dbc.CardBody([

                        html.Div(
                            id="status-selecao",
                            style={
                                "fontSize": "12px",
                                "color": "#475569",
                                "marginBottom": "10px"
                            }
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
                                {"name": "Coord Y", "id": "tsne_dim_2"}
                            ],

                            data=[],
                            page_size=10,
                            sort_action="native",
                            filter_action="native",
                            row_selectable="single",
                            active_cell=None,

                            style_table={
                                "overflowX": "auto"
                            },

                            style_cell={
                                "textAlign": "left",
                                "fontFamily": "sans-serif",
                                "fontSize": "12px",
                                "padding": "8px"
                            },

                            style_header={
                                "backgroundColor": "#f1f5f9",
                                "fontWeight": "bold",
                                "color": "#1e293b"
                            },

                            style_data_conditional=[
                                {
                                    "if": {"state": "active"},
                                    "backgroundColor": "#dbeafe",
                                    "border": "1px solid #2563eb"
                                }
                            ]
                        )

                    ])
                ], className="shadow-sm mb-4")

            ], width=10)

        ])

    ], fluid=True, style={
        "backgroundColor": "#f4f7f9",
        "padding": "20px",
        "minHeight": "100vh"
    })

])


# ============================================================
# 4. ATUALIZAÇÃO DOS DROPDOWNS
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
        Output("slider-epocas", "value")
    ],
    Input("dropdown-dataset", "value")
)
def update_options(dataset_sel):

    df_d3 = carregar_csv_seguro(
        "explicacoes_instancias", dataset_sel
    )

    df_d6 = carregar_csv_seguro(
        "projecao_tsne_epocas", dataset_sel
    )

    metodos = ["SHAP", "PFI", "LIME", "IG"]

    if not df_d3.empty and "metodo_xai" in df_d3.columns:
        metodos = sorted(
            df_d3["metodo_xai"].dropna().unique().tolist()
        )

    epocas = [1, 10, 20, 30, 40]

    if not df_d6.empty and "epoca" in df_d6.columns:
        epocas = sorted(
            pd.to_numeric(
                df_d6["epoca"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )

    elif not df_d3.empty and "epoca" in df_d3.columns:
        epocas = sorted(
            pd.to_numeric(
                df_d3["epoca"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )

    minimo = min(epocas) if epocas else 1
    maximo = max(epocas) if epocas else 40

    return (
        [{"label": str(m).upper(), "value": m} for m in metodos],
        metodos[0] if metodos else "SHAP",
        [{"label": f"Época {e}", "value": e} for e in epocas],
        epocas[0] if epocas else 1,
        minimo,
        maximo,
        {e: str(e) for e in epocas},
        [minimo, maximo]
    )


# ============================================================
# 5. CALLBACK CENTRAL DE SELEÇÕES
# ============================================================

@app.callback(
    Output("estado-selecao", "data"),

    [
        Input("dropdown-dataset", "value"),
        Input("dropdown-metodo", "value"),
        Input("dim-dropdown-epoca", "value"),

        Input("sankey-graph", "clickData"),

        Input("dim-scatter-graph", "selectedData"),
        Input("dim-scatter-graph", "clickData"),

        Input("dim-reduction-aux-graph", "clickData"),
        Input("dim-reduction-aux-graph", "selectedData"),

        Input("dim-tabela-instancias", "active_cell")
    ],

    [
        State("dim-tabela-instancias", "data"),
        State("estado-selecao", "data")
    ]
)
def atualizar_selecao(
    dataset,
    metodo,
    epoca_controle,
    sankey_click,
    scatter_selected,
    scatter_click,
    box_click,
    box_selected,
    tabela_active,
    tabela_data,
    estado_atual
):

    estado_atual = estado_atual or {}

    gatilho = ctx.triggered_id

    estado_vazio = {
        "instancias": [],
        "epoca": epoca_controle,
        "origem": None,
        "transicao": None
    }

    # Alterou dataset ou método: limpa seleções anteriores.
    if gatilho in ["dropdown-dataset", "dropdown-metodo"]:
        return estado_vazio

    # Alterou manualmente a época 2D.
    if gatilho == "dim-dropdown-epoca":
        return estado_vazio

    # --------------------------------------------------------
    # A. SANKEY -> INSTÂNCIAS DA TRANSIÇÃO
    # --------------------------------------------------------

    if gatilho == "sankey-graph":

        if not sankey_click or not sankey_click.get("points"):
            return estado_vazio

        ponto = sankey_click["points"][0]
        custom = ponto.get("customdata")

        if custom is None:
            return estado_vazio

        # Estrutura:
        # [lista_ids, epoca_origem, epoca_destino,
        #  classe_origem, classe_destino]

        ids = custom[0] if len(custom) > 0 else []

        if not isinstance(ids, (list, tuple, np.ndarray)):
            ids = []

        return {
            "instancias": [normalizar_id(x) for x in ids],
            "epoca": None,
            "origem": "sankey",
            "transicao": {
                "epoca_origem": custom[1],
                "epoca_destino": custom[2],
                "classe_origem": custom[3],
                "classe_destino": custom[4]
            }
        }

    # --------------------------------------------------------
    # B. SCATTER -> BRUSHING / LASSO
    # --------------------------------------------------------

    if gatilho == "dim-scatter-graph":

        if scatter_selected and scatter_selected.get("points"):

            ids = []

            for ponto in scatter_selected["points"]:
                custom = ponto.get("customdata")

                if custom is not None and len(custom) > 0:
                    ids.append(normalizar_id(custom[0]))

            ids = list(dict.fromkeys(
                x for x in ids if x is not None
            ))

            return {
                "instancias": ids,
                "epoca": epoca_controle,
                "origem": "scatter",
                "transicao": None
            }

        if scatter_click and scatter_click.get("points"):

            ponto = scatter_click["points"][0]
            custom = ponto.get("customdata")

            if custom is not None:
                return {
                    "instancias": [normalizar_id(custom[0])],
                    "epoca": epoca_controle,
                    "origem": "scatter",
                    "transicao": None
                }

        return estado_vazio

    # --------------------------------------------------------
    # C. BOXPLOT -> ÉPOCA / OUTLIERS
    # --------------------------------------------------------

    if gatilho == "dim-reduction-aux-graph":

        if not box_click or not box_click.get("points"):
            return estado_vazio

        ponto = box_click["points"][0]

        epoca = ponto.get("x")

        custom = ponto.get("customdata")

        # Clique em um ponto individual do Boxplot.
        # Recupera o ID da instância.
        if custom is not None and len(custom) > 0:
            instancia = normalizar_id(custom[0])

            return {
                "instancias": [instancia],
                "epoca": int(epoca) if epoca is not None else epoca_controle,
                "origem": "boxplot_instancia",
                "transicao": None
            }

        # Clique na época.
        return {
            "instancias": [],
            "epoca": int(epoca) if epoca is not None else epoca_controle,
            "origem": "boxplot_epoca",
            "transicao": None
        }

    # --------------------------------------------------------
    # D. BOXPLOT -> SELEÇÃO DE OUTLIERS / PONTOS
    # --------------------------------------------------------

    if gatilho == "dim-reduction-aux-graph" and box_selected:

        ids = []

        for ponto in box_selected.get("points", []):

            custom = ponto.get("customdata")

            if custom is not None and len(custom) > 0:
                ids.append(normalizar_id(custom[0]))

        if ids:
            return {
                "instancias": list(dict.fromkeys(ids)),
                "epoca": epoca_controle,
                "origem": "boxplot_instancia",
                "transicao": None
            }

    # --------------------------------------------------------
    # E. TABELA -> SCATTER
    # --------------------------------------------------------

    if gatilho == "dim-tabela-instancias":

        if tabela_active and tabela_data:

            indice = tabela_active.get("row")

            if indice is not None and indice < len(tabela_data):

                registro = tabela_data[indice]

                return {
                    "instancias": [
                        normalizar_id(registro["instancia_id"])
                    ],
                    "epoca": registro.get("epoca", epoca_controle),
                    "origem": "tabela",
                    "transicao": None
                }

        return estado_vazio

    return estado_atual


# ============================================================
# 6. CALLBACK PRINCIPAL DAS VISUALIZAÇÕES
# ============================================================

@app.callback(

    [
        Output("metrics-stability-combined-graph", "figure"),
        Output("heatmap-graph", "figure"),
        Output("dim-scatter-graph", "figure"),
        Output("dim-reduction-aux-graph", "figure"),
        Output("dim-tabela-instancias", "data"),
        Output("sankey-graph", "figure"),
        Output("status-selecao", "children")
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
        Input("estado-selecao", "data")
    ]
)
def atualizar_visualizacoes(
    dataset_sel,
    metodo_sel,
    metrica_est_sel,
    cor_sankey_sel,
    epocas_sel,
    epoca_proj_sel,
    algo_sel,
    cor_proj_sel,
    trajetoria_sel,
    estado
):

    estado = estado or {}

    ids_selecionados = estado.get("instancias", [])
    epoca_estado = estado.get("epoca")
    origem = estado.get("origem")

    ids_selecionados = [
        normalizar_id(x)
        for x in ids_selecionados
        if x is not None
    ]

    # --------------------------------------------------------
    # CARREGAMENTO
    # --------------------------------------------------------

    df_d2 = carregar_csv_seguro(
        "metricas_epocas", dataset_sel
    )

    df_d4 = carregar_csv_seguro(
        "resumo_explicacoes", dataset_sel
    )

    df_d6 = carregar_csv_seguro(
        "projecao_tsne_epocas", dataset_sel
    )

    df_d5 = carregar_csv_seguro(
        "metricas_estabilidade_concordancia", dataset_sel
    )

    df_d3 = carregar_csv_seguro(
        "explicacoes_instancias", dataset_sel
    )

    label_ds = MAPA_DATASET.get(
        dataset_sel, {}
    ).get("label", dataset_sel.upper())

    # Padronização dos identificadores.
    for df in [df_d3, df_d6]:
        if not df.empty and "instancia_id" in df.columns:
            df["id_normalizado"] = (
                df["instancia_id"].map(normalizar_id)
            )

    # --------------------------------------------------------
    # 1. PERFORMANCE + ESTABILIDADE
    # --------------------------------------------------------

    fig_combined = go.Figure()

    if not df_d2.empty:

        acc_col = (
            "accuracy" if "accuracy" in df_d2.columns
            else df_d2.columns[1]
        )

        loss_col = (
            "loss_validacao"
            if "loss_validacao" in df_d2.columns
            else None
        )

        if "epoca" in df_d2.columns:

            fig_combined.add_trace(
                go.Scatter(
                    x=df_d2["epoca"],
                    y=df_d2[acc_col],
                    mode="lines+markers",
                    name="Acurácia",
                    line=dict(color="#2ca02c")
                )
            )

            if loss_col:
                fig_combined.add_trace(
                    go.Scatter(
                        x=df_d2["epoca"],
                        y=df_d2[loss_col],
                        mode="lines+markers",
                        name="Loss Validação",
                        yaxis="y2",
                        line=dict(color="#d62728")
                    )
                )

        if (
            not df_d5.empty
            and "tipo_metrica" in df_d5.columns
            and "metrica" in df_d5.columns
        ):

            mask = (
                df_d5["tipo_metrica"].astype(str).str.upper()
                == str(metodo_sel).upper()
            )

            mask &= (
                df_d5["metrica"].astype(str).str.lower()
                == str(metrica_est_sel).lower()
            )

            df_est = df_d5[mask]

            if not df_est.empty:

                fig_combined.add_trace(
                    go.Scatter(
                        x=df_est["epoca_destino"],
                        y=df_est["valor_metrica"],
                        mode="lines+markers",
                        name=f"Estabilidade ({metrica_est_sel})",
                        line=dict(color="#1f77b4", dash="dash")
                    )
                )

    fig_combined.update_layout(
        title=f"Performance e Estabilidade XAI ({metodo_sel})",
        yaxis_title="Acurácia / Estabilidade",
        yaxis2=dict(
            title="Loss",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        margin=dict(t=40, b=30)
    )

    # --------------------------------------------------------
    # 2. HEATMAP
    # --------------------------------------------------------

    fig_heat = go.Figure()

    if (
        not df_d4.empty
        and "metodo_xai" in df_d4.columns
    ):

        df_h = df_d4[
            df_d4["metodo_xai"].astype(str).str.upper()
            == str(metodo_sel).upper()
        ]

        if not df_h.empty:

            val_col = (
                "importancia_media"
                if "importancia_media" in df_h.columns
                else df_h.columns[-1]
            )

            piv = df_h.pivot_table(
                index="feature",
                columns="epoca",
                values=val_col,
                fill_value=0
            )

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=piv.values,
                    x=piv.columns,
                    y=piv.index,
                    colorscale=COLORSCALE_REDS_20,
                    hovertemplate=(
                        "Feature: %{y}<br>"
                        "Época: %{x}<br>"
                        "Importância: %{z:.4f}"
                        "<extra></extra>"
                    )
                )
            )

            fig_heat.update_layout(
                title=f"Heatmap de Importância ({metodo_sel})",
                template="plotly_white",
                margin=dict(t=40, b=30)
            )

    # --------------------------------------------------------
    # 3. REDUÇÃO DE DIMENSIONALIDADE
    # --------------------------------------------------------

    fig_scatter = go.Figure()
    tabela_records = []

    epoca_alvo = (
        epoca_estado
        if epoca_estado is not None
        else epoca_proj_sel
    )

    if not df_d6.empty:

        df_sub = df_d6.copy()

        df_sub["epoca"] = pd.to_numeric(
            df_sub["epoca"], errors="coerce"
        )

        df_sub = df_sub[
            (
                df_sub["metodo_xai"].astype(str).str.upper()
                == str(metodo_sel).upper()
            )
            &
            (df_sub["epoca"] == int(epoca_alvo))
        ].copy()

        if not df_sub.empty:

            # PCA calculado a partir das explicações.
            if algo_sel == "PCA" and not df_d3.empty:

                df_exp = df_d3[
                    (
                        df_d3["metodo_xai"].astype(str).str.upper()
                        == str(metodo_sel).upper()
                    )
                    &
                    (
                        pd.to_numeric(
                            df_d3["epoca"], errors="coerce"
                        ) == int(epoca_alvo)
                    )
                ].copy()

                if not df_exp.empty:

                    pivot = df_exp.pivot_table(
                        index="instancia_id",
                        columns="feature",
                        values="valor_atribuicao"
                    ).fillna(0)

                    if len(pivot) >= 2:

                        pca = PCA(n_components=2)

                        resultado = pca.fit_transform(
                            pivot.values
                        )

                        pca_df = pd.DataFrame(
                            resultado,
                            columns=["dim_1", "dim_2"]
                        )

                        pca_df["id_normalizado"] = [
                            normalizar_id(x)
                            for x in pivot.index
                        ]

                        df_sub = df_sub.merge(
                            pca_df,
                            on="id_normalizado",
                            how="left"
                        )

                        x_col = "dim_1"
                        y_col = "dim_2"

                    else:
                        x_col = "tsne_dim_1"
                        y_col = "tsne_dim_2"

                else:
                    x_col = "tsne_dim_1"
                    y_col = "tsne_dim_2"

            else:
                x_col = "tsne_dim_1"
                y_col = "tsne_dim_2"

            # ------------------------------------------------
            # COLORAÇÃO
            # ------------------------------------------------

            if cor_proj_sel == "erro_predicao":

                if "erro_predicao" not in df_sub.columns:
                    df_sub["erro_predicao"] = (
                        df_sub["classe_real"]
                        != df_sub["classe_predita"]
                    ).astype(int)

                df_sub["cor_label"] = (
                    df_sub["erro_predicao"]
                    .map({0: "Acerto", 1: "Erro"})
                )

                fig_scatter = px.scatter(
                    df_sub,
                    x=x_col,
                    y=y_col,
                    color="cor_label",
                    color_discrete_map={
                        "Acerto": "#059669",
                        "Erro": "#dc2626"
                    },
                    custom_data=[
                        "instancia_id",
                        "classe_real",
                        "classe_predita",
                        "erro_predicao"
                    ]
                )

            else:

                fig_scatter = px.scatter(
                    df_sub,
                    x=x_col,
                    y=y_col,
                    color=cor_proj_sel,
                    color_continuous_scale=COLORSCALE_BLUES_20,
                    custom_data=[
                        "instancia_id",
                        "classe_real",
                        "classe_predita",
                        "erro_predicao"
                    ]
                )

            fig_scatter.update_traces(
                marker=dict(
                    size=8,
                    opacity=0.65,
                    line=dict(width=0.5, color="#334155")
                ),
                hovertemplate=(
                    "<b>Instância:</b> #%{customdata[0]}<br>"
                    "<b>Classe Real:</b> %{customdata[1]}<br>"
                    "<b>Classe Predita:</b> %{customdata[2]}<br>"
                    "<b>Erro:</b> %{customdata[3]}<br>"
                    "<b>X:</b> %{x:.3f}<br>"
                    "<b>Y:</b> %{y:.3f}"
                    "<extra></extra>"
                )
            )

            # ------------------------------------------------
            # DESTAQUE DAS INSTÂNCIAS SELECIONADAS
            # ------------------------------------------------

            if ids_selecionados:

                df_destaque = df_sub[
                    df_sub["id_normalizado"].isin(ids_selecionados)
                ]

                if not df_destaque.empty:

                    fig_scatter.add_trace(
                        go.Scatter(
                            x=df_destaque[x_col],
                            y=df_destaque[y_col],
                            mode="markers",
                            marker=dict(
                                size=17,
                                color="rgba(0,0,0,0)",
                                line=dict(
                                    color="#f59e0b",
                                    width=3
                                ),
                                symbol="circle"
                            ),
                            customdata=df_destaque[
                                ["instancia_id"]
                            ].values,
                            name="Instâncias Selecionadas",
                            hovertemplate=(
                                "Instância selecionada: %{customdata[0]}"
                                "<extra></extra>"
                            )
                        )
                    )

            # ------------------------------------------------
            # TRAJETÓRIA INDIVIDUAL
            # ------------------------------------------------

            if "ATIVO" in (trajetoria_sel or []) and ids_selecionados:

                df_traj = df_d6[
                    (
                        df_d6["metodo_xai"].astype(str).str.upper()
                        == str(metodo_sel).upper()
                    )
                    &
                    (
                        df_d6["id_normalizado"].isin(ids_selecionados)
                    )
                ].copy()

                df_traj["epoca"] = pd.to_numeric(
                    df_traj["epoca"], errors="coerce"
                )

                df_traj = df_traj.sort_values("epoca")

                for inst in ids_selecionados:

                    traj = df_traj[
                        df_traj["id_normalizado"] == inst
                    ]

                    if not traj.empty:

                        fig_scatter.add_trace(
                            go.Scatter(
                                x=traj["tsne_dim_1"],
                                y=traj["tsne_dim_2"],
                                mode="lines+markers+text",
                                text=[
                                    f"E{int(e)}"
                                    for e in traj["epoca"]
                                ],
                                textposition="top center",
                                line=dict(
                                    color="#f59e0b",
                                    width=2,
                                    dash="dash"
                                ),
                                marker=dict(
                                    size=9,
                                    color="#f59e0b"
                                ),
                                name=f"Trajetória #{inst}"
                            )
                        )

            fig_scatter.update_layout(
                title=(
                    f"Espaço 2D ({algo_sel}) - "
                    f"Época {epoca_alvo} - {metodo_sel}"
                ),
                xaxis_title=f"{algo_sel} Dim 1",
                yaxis_title=f"{algo_sel} Dim 2",
                template="plotly_white",
                margin=dict(t=40, b=30)
            )

            # ------------------------------------------------
            # TABELA: SOMENTE INSTÂNCIAS SELECIONADAS
            # ------------------------------------------------

            if ids_selecionados:

                df_tabela = df_sub[
                    df_sub["id_normalizado"].isin(ids_selecionados)
                ].copy()

            else:

                df_tabela = df_sub.copy()

            if "erro_predicao" not in df_tabela.columns:
                df_tabela["erro_predicao"] = (
                    df_tabela["classe_real"]
                    != df_tabela["classe_predita"]
                ).astype(int)

            df_tabela["tsne_dim_1"] = df_tabela[x_col]
            df_tabela["tsne_dim_2"] = df_tabela[y_col]

            colunas_tabela = [
                "instancia_id",
                "epoca",
                "metodo_xai",
                "classe_real",
                "classe_predita",
                "erro_predicao",
                "tsne_dim_1",
                "tsne_dim_2"
            ]

            colunas_existentes = [
                c for c in colunas_tabela
                if c in df_tabela.columns
            ]

            tabela_records = (
                df_tabela[colunas_existentes]
                .replace({np.nan: None})
                .to_dict("records")
            )

    # --------------------------------------------------------
    # 4. BOXPLOT COM PONTOS INDIVIDUAIS SELECIONÁVEIS
    # --------------------------------------------------------

    fig_box = go.Figure()

    if (
        not df_d6.empty
        and "tsne_dim_1" in df_d6.columns
    ):

        df_box = df_d6[
            (
                df_d6["metodo_xai"].astype(str).str.upper()
                == str(metodo_sel).upper()
            )
        ].copy()

        df_box["epoca"] = pd.to_numeric(
            df_box["epoca"], errors="coerce"
        )

        df_box = df_box.dropna(
            subset=["epoca", "tsne_dim_1"]
        )

        if not df_box.empty:

            for ep in sorted(df_box["epoca"].unique()):

                grupo = df_box[
                    df_box["epoca"] == ep
                ]

                fig_box.add_trace(
                    go.Box(
                        x=[int(ep)] * len(grupo),
                        y=grupo["tsne_dim_1"],
                        name=f"Época {int(ep)}",
                        boxpoints="all",
                        jitter=0.35,
                        pointpos=0,
                        marker=dict(
                            size=5,
                            color="#6366f1",
                            opacity=0.65
                        ),
                        line=dict(color="#6366f1"),
                        customdata=grupo[
                            ["instancia_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Época:</b> %{x}<br>"
                            "<b>Instância:</b> %{customdata[0]}<br>"
                            "<b>Dimensão 1:</b> %{y:.4f}"
                            "<extra></extra>"
                        )
                    )
                )

            # Destaque da época selecionada.
            if epoca_estado is not None:

                fig_box.add_vline(
                    x=int(epoca_estado),
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text="Época selecionada"
                )

            fig_box.update_layout(
                title="Dispersão Temporal (t-SNE Dim 1)",
                xaxis_title="Época",
                yaxis_title="t-SNE Dim 1",
                showlegend=False,
                template="plotly_white",
                clickmode="event+select",
                dragmode="select",
                margin=dict(t=40, b=30)
            )

    # --------------------------------------------------------
    # 5. SANKEY COM IDENTIFICAÇÃO DAS INSTÂNCIAS
    # --------------------------------------------------------

    fig_sankey = go.Figure()

    df_d3_s, df_d2_s = carregar_dados_sankey(dataset_sel)

    df_sankey, df_trans, _ = processar_dados_sankey(
        df_d3_s,
        df_d2_s,
        dataset_sel
    )

    if not df_sankey.empty:

        metodos_validos = df_sankey["metodo_xai"].unique()

        metodo_alvo = (
            metodo_sel
            if metodo_sel in metodos_validos
            else metodos_validos[0]
        )

        df_filtered = df_sankey[
            df_sankey["metodo_xai"] == metodo_alvo
        ].copy()

        if epocas_sel:

            df_filtered = df_filtered[
                (
                    df_filtered["epoca_origem"]
                    >= epocas_sel[0]
                )
                &
                (
                    df_filtered["epoca_destino"]
                    <= epocas_sel[1]
                )
            ]

        if not df_filtered.empty:

            classes = [0, 1]

            mapa_classes = {
                "breast_cancer": {
                    0: "Benigno",
                    1: "Maligno"
                },
                "wine_quality": {
                    0: "Baixa Qualidade",
                    1: "Alta Qualidade"
                },
                "german_credit": {
                    0: "Alto Risco",
                    1: "Bom Crédito"
                }
            }.get(dataset_sel, {
                0: "Classe 0",
                1: "Classe 1"
            })

            epocas = sorted(
                set(df_filtered["epoca_origem"])
                .union(df_filtered["epoca_destino"])
            )

            labels = []
            node_x = []
            node_y = []
            node_dict = {}

            for i, ep in enumerate(epocas):

                for j, classe in enumerate(classes):

                    node_id = len(labels)

                    node_dict[f"E{ep}_C{classe}"] = node_id

                    labels.append(
                        f"Época {ep}<br>{mapa_classes[classe]}"
                    )

                    node_x.append(
                        0.05 + 0.9 * (
                            i / max(1, len(epocas) - 1)
                        )
                    )

                    node_y.append(
                        0.2 if classe == 0 else 0.8
                    )

            sources = []
            targets = []
            values = []
            colors = []
            customdata = []
            hovertexts = []

            for _, row in df_filtered.iterrows():

                origem_node = (
                    f"E{row['epoca_origem']}_C{row['classe_origem']}"
                )

                destino_node = (
                    f"E{row['epoca_destino']}_C{row['classe_destino']}"
                )

                if (
                    origem_node not in node_dict
                    or destino_node not in node_dict
                ):
                    continue

                sources.append(node_dict[origem_node])
                targets.append(node_dict[destino_node])

                values.append(int(row["qtd_instancias"]))

                conf = float(row["confianca_media"])
                erro = float(row["taxa_erro"])

                if cor_sankey_sel == "classe":
                    rgb = (
                        "37,99,235"
                        if row["classe_origem"] == 0
                        else "249,115,22"
                    )

                elif cor_sankey_sel == "confianca":
                    rgb = (
                        f"{int(255*(1-conf))},"
                        f"{int(200*conf)},150"
                    )

                else:
                    rgb = (
                        f"{int(255*erro)},50,50"
                    )

                colors.append(
                    f"rgba({rgb},0.75)"
                )

                # CUSTOMDATA CONTÉM OS IDS DO FLUXO.
                ids_fluxo = [
                    normalizar_id(x)
                    for x in row["instancias_ids"]
                ]

                customdata.append([
                    ids_fluxo,
                    int(row["epoca_origem"]),
                    int(row["epoca_destino"]),
                    int(row["classe_origem"]),
                    int(row["classe_destino"])
                ])

                hovertexts.append(
                    f"<b>Dataset:</b> {dataset_sel}<br>"
                    f"<b>Método:</b> {metodo_alvo}<br>"
                    f"<b>Transição:</b> "
                    f"{row['epoca_origem']} → "
                    f"{row['epoca_destino']}<br>"
                    f"<b>Classe:</b> "
                    f"{mapa_classes[row['classe_origem']]} → "
                    f"{mapa_classes[row['classe_destino']]}<br>"
                    f"<b>Instâncias:</b> {row['qtd_instancias']}<br>"
                    f"<b>Confiança:</b> {conf*100:.1f}%<br>"
                    f"<b>Taxa de erro:</b> {erro*100:.1f}%<br>"
                    f"<b>Importância acumulada:</b> "
                    f"{row['soma_importancia']:.4f}"
                )

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="snap",

                        node=dict(
                            pad=30,
                            thickness=20,
                            line=dict(
                                color="#334155",
                                width=0.5
                            ),
                            label=labels,
                            x=node_x,
                            y=node_y,
                            color="#e2e8f0"
                        ),

                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color=colors,
                            customdata=customdata,
                            hovertemplate=(
                                "%{customdata[0]}"
                                "<extra></extra>"
                            )
                        )
                    )
                ]
            )

            fig_sankey.update_layout(
                title=(
                    f"Transições Temporais de Explicações "
                    f"({metodo_alvo}) - {dataset_sel.upper()}"
                ),
                font_size=12,
                template="plotly_white",
                margin=dict(t=50, b=20)
            )

    # --------------------------------------------------------
    # 6. STATUS DAS SELEÇÕES
    # --------------------------------------------------------

    if origem == "sankey":

        status = (
            f"Seleção originada no Sankey: "
            f"{len(ids_selecionados)} instâncias pertencentes "
            f"ao fluxo selecionado."
        )

    elif origem == "boxplot_epoca":

        status = (
            f"Época {epoca_estado} selecionada no Boxplot. "
            f"A projeção 2D foi filtrada para essa época."
        )

    elif origem == "boxplot_instancia":

        status = (
            f"Instâncias selecionadas no Boxplot: "
            f"{len(ids_selecionados)}."
        )

    elif origem == "scatter":

        status = (
            f"Brushing/Lasso: "
            f"{len(ids_selecionados)} instâncias selecionadas."
        )

    elif origem == "tabela":

        status = (
            "Instância selecionada na tabela e destacada "
            "na projeção 2D."
        )

    else:

        status = (
            f"Exibindo todas as instâncias da época {epoca_alvo}. "
            "Selecione elementos nas visualizações para filtrar."
        )

    return (
        fig_combined,
        fig_heat,
        fig_scatter,
        fig_box,
        tabela_records,
        fig_sankey,
        status
    )


# ============================================================
# 7. EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    app.run(debug=False)
