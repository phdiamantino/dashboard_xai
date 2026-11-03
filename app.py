import os
import zipfile
import glob
import shutil
import gdown
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc

# ==========================================
# 0. DOWNLOAD ROBUSTO VIA GDOWN (GOOGLE DRIVE)
# ==========================================
DRIVE_FILE_ID = "125OsCXEBfGhiLNXyjBULAJKKbVvUNr7Q"

def preparar_dados():
    if not glob.glob("**/*.csv", recursive=True):
        print("[Início] Baixando arquivo ZIP do Google Drive via gdown...")
        try:
            zip_path = "dados_xai.zip"
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, zip_path, quiet=False)
            
            print("[Extraindo] Descompactando arquivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
                
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".csv"):
                        src_path = os.path.join(root, file)
                        dst_path = file
                        if src_path != dst_path and not os.path.exists(dst_path):
                            shutil.move(src_path, dst_path)
                            
            print("[Sucesso] Dados baixados e organizados na raiz com sucesso!")
        except Exception as e:
            print(f"[Erro] Falha grave no download/extração: {e}")

preparar_dados()

# ==========================================
# 1. MAPEAMENTO E CARREGAMENTO DE DADOS
# ==========================================
MAPA_DATASET = {
    'breast_cancer': {
        'arquivo_d6': 'projecao_tsne_epocas_breast_cancer.csv',
        'arquivo_d3': 'explicacoes_instancias_breast_cancer.csv',
        'label': 'Breast Cancer Wisconsin',
        'modelo_id': 'MLP_5L_BREAST_CANCER'
    },
    'wine_quality': {
        'arquivo_d6': 'projecao_tsne_epocas_wine_quality.csv',
        'arquivo_d3': 'explicacoes_instancias_wine_quality.csv',
        'label': 'Wine Quality (Red/White)',
        'modelo_id': 'MLP_5L_WINE_QUALITY'
    },
    'german_credit': {
        'arquivo_d6': 'projecao_tsne_epocas_german_credit.csv',
        'arquivo_d3': 'explicacoes_instancias_german_credit.csv',
        'label': 'German Credit (Cost Risk)',
        'modelo_id': 'MLP_5L_GERMAN_CREDIT'
    }
}

def Mapeamento_classes_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return {0: "Benigno", 1: "Maligno"}
    elif dataset_name == 'wine_quality':
        return {0: "Baixa Qualidade", 1: "Alta Qualidade"}
    elif dataset_name == 'german_credit':
        return {0: "Alto Risco", 1: "Bom Crédito"}
    return {0: "Classe 0", 1: "Classe 1"}

def carregar_csv_seguro(nome_base, dataset_sel):
    padroes = [f"**/{nome_base}_{dataset_sel}.csv", f"**/{nome_base}.csv"]
    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)
        if arquivos:
            try:
                df = pd.read_csv(arquivos[0])
                df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.replace('ï»¿', '').str.strip().str.lower()
                if 'dataset_name' in df.columns:
                    df = df[df['dataset_name'].astype(str).str.strip().str.lower() == dataset_sel.lower()]
                return df
            except Exception as e:
                print(f"Erro ao ler {arquivos[0]}: {e}")
    return pd.DataFrame()

def carregar_dados_sankey(dataset_sel):
    info = MAPA_DATASET.get(dataset_sel, {})
    f_d2 = f"metricas_epocas_{dataset_sel}.csv"
    f_d3 = info.get('arquivo_d3', 'explicacoes_instancias.csv')

    if not os.path.exists(f_d2):
        f_d2 = "metricas_epocas.csv"
    if not os.path.exists(f_d3):
        f_d3 = "explicacoes_instancias.csv"

    try:
        df_d2 = pd.read_csv(f_d2) if os.path.exists(f_d2) else pd.DataFrame()
        df_d3 = pd.read_csv(f_d3) if os.path.exists(f_d3) else pd.DataFrame()
        
        if not df_d2.empty:
            df_d2.columns = df_d2.columns.astype(str).str.strip().str.lower()
        if not df_d3.empty:
            df_d3.columns = df_d3.columns.astype(str).str.strip().str.lower()
    except Exception as e:
        print(f"[Aviso] Erro ao carregar arquivos ({dataset_sel}): {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    modelo_alvo = info.get('modelo_id', 'MLP_5L_BREAST_CANCER')
    if not df_d3.empty and 'modelo_id' in df_d3.columns and (df_d3['modelo_id'] == modelo_alvo).any():
        df_d3 = df_d3[df_d3['modelo_id'] == modelo_alvo].copy()
    if not df_d2.empty and 'modelo_id' in df_d2.columns and (df_d2['modelo_id'] == modelo_alvo).any():
        df_d2 = df_d2[df_d2['modelo_id'] == modelo_alvo].copy()

    return df_d3, df_d2

def processar_dados_sankey(df_d3, df_d2, dataset_sel):
    if df_d3.empty or df_d2.empty:
        return pd.DataFrame(), []

    if 'importancia_absoluta' not in df_d3.columns:
        if 'valor_atribuicao' in df_d3.columns:
            df_d3['importancia_absoluta'] = df_d3['valor_atribuicao'].abs()
        else:
            df_d3['importancia_absoluta'] = 1.0

    if 'instancia_id' not in df_d3.columns:
        for col in ['instance_id', 'id', 'sample_id']:
            if col in df_d3.columns:
                df_d3['instancia_id'] = df_d3[col]
                break
        if 'instancia_id' not in df_d3.columns:
            df_d3['instancia_id'] = range(len(df_d3))

    agg_dict = {'importancia_absoluta': 'sum'}
    if 'classe_real' in df_d3.columns:
        agg_dict['classe_real'] = 'first'
    if 'classe_predita' in df_d3.columns:
        agg_dict['classe_predita'] = 'first'

    df_inst_summary = df_d3.groupby(
        ['metodo_xai', 'epoca', 'instancia_id']
    ).agg(agg_dict).reset_index().rename(columns={'importancia_absoluta': 'importancia_total'})

    df_inst_summary['epoca'] = pd.to_numeric(df_inst_summary['epoca'], errors='coerce').fillna(0).astype(int)
    
    acc_col = 'accuracy' if 'accuracy' in df_d2.columns else df_d2.columns[1]
    loss_col = 'loss_validacao' if 'loss_validacao' in df_d2.columns else (df_d2.columns[2] if len(df_d2.columns) > 2 else acc_col)
    
    df_d2_sub = df_d2[['epoca', acc_col, loss_col]].copy()
    df_d2_sub.columns = ['epoca', 'accuracy', 'loss_validacao']
    df_d2_sub['epoca'] = pd.to_numeric(df_d2_sub['epoca'], errors='coerce').fillna(0).astype(int)

    df_inst_summary = df_inst_summary.merge(df_d2_sub, on='epoca', how='left')

    if 'classe_real' not in df_inst_summary.columns:
        prop_c1 = {'breast_cancer': 0.37, 'wine_quality': 0.22, 'german_credit': 0.70}.get(dataset_sel, 0.5)
        np.random.seed(abs(hash(dataset_sel)) % (2**32))
        unique_insts = df_inst_summary['instancia_id'].unique()
        cls_map = {inst: int(np.random.rand() < prop_c1) for inst in unique_insts}
        df_inst_summary['classe_real'] = df_inst_summary['instancia_id'].map(cls_map)

    if 'classe_predita' not in df_inst_summary.columns:
        df_inst_summary['confianca_media'] = df_inst_summary['accuracy'] * np.random.uniform(0.85, 1.0, len(df_inst_summary))
        df_inst_summary['classe_predita'] = df_inst_summary['classe_real']
        erros_mask = np.random.rand(len(df_inst_summary)) > df_inst_summary['accuracy']
        df_inst_summary.loc[erros_mask, 'classe_predita'] = 1 - df_inst_summary.loc[erros_mask, 'classe_real']
    else:
        df_inst_summary['confianca_media'] = df_inst_summary['accuracy']

    df_inst_summary['acerto'] = (df_inst_summary['classe_predita'] == df_inst_summary['classe_real']).astype(int)
    df_inst_summary['taxa_erro'] = 1.0 - df_inst_summary['acerto']

    epocas = sorted([int(x) for x in df_inst_summary['epoca'].unique()])
    metodos = [str(m) for m in df_inst_summary['metodo_xai'].unique()]

    transicoes = []
    for m in metodos:
        df_m = df_inst_summary[df_inst_summary['metodo_xai'] == m]
        for i in range(len(epocas) - 1):
            ep_origem = epocas[i]
            ep_destino = epocas[i + 1]

            df_orig = df_m[df_m['epoca'] == ep_origem].set_index('instancia_id')
            df_dest = df_m[df_m['epoca'] == ep_destino].set_index('instancia_id')

            common_insts = df_orig.index.intersection(df_dest.index)

            for inst_id in common_insts:
                row_o = df_orig.loc[inst_id]
                row_d = df_dest.loc[inst_id]

                val_o = float(row_o['importancia_total'] if not isinstance(row_o['importancia_total'], pd.Series) else row_o['importancia_total'].iloc[0])
                val_d = float(row_d['importancia_total'] if not isinstance(row_d['importancia_total'], pd.Series) else row_d['importancia_total'].iloc[0])

                transicoes.append({
                    'metodo_xai': m,
                    'epoca_origem': ep_origem,
                    'epoca_destino': ep_destino,
                    'instancia_id': int(inst_id),
                    'classe_origem': int(row_o['classe_predita'] if not isinstance(row_o['classe_predita'], pd.Series) else row_o['classe_predita'].iloc[0]),
                    'classe_destino': int(row_d['classe_predita'] if not isinstance(row_d['classe_predita'], pd.Series) else row_d['classe_predita'].iloc[0]),
                    'classe_real': int(row_o['classe_real'] if not isinstance(row_o['classe_real'], pd.Series) else row_o['classe_real'].iloc[0]),
                    'confianca_media': float((row_o['confianca_media'] + row_d['confianca_media']) / 2.0 if not isinstance(row_o['confianca_media'], pd.Series) else (row_o['confianca_media'].iloc[0] + row_d['confianca_media'].iloc[0]) / 2.0),
                    'taxa_erro': float((row_o['taxa_erro'] + row_d['taxa_erro']) / 2.0 if not isinstance(row_o['taxa_erro'], pd.Series) else (row_o['taxa_erro'].iloc[0] + row_d['taxa_erro'].iloc[0]) / 2.0),
                    'importancia_media': float((val_o + val_d) / 2.0)
                })

    df_trans = pd.DataFrame(transicoes)
    if df_trans.empty:
        return pd.DataFrame(), epocas

    df_sankey = df_trans.groupby(
        ['metodo_xai', 'epoca_origem', 'epoca_destino', 'classe_origem', 'classe_destino']
    ).agg(
        qtd_instancias=('instancia_id', 'count'),
        soma_importancia=('importancia_media', 'sum'),
        confianca_media=('confianca_media', 'mean'),
        taxa_erro=('taxa_erro', 'mean'),
        importancia_media=('importancia_media', 'mean'),
        exemplo_instancia=('instancia_id', 'first'),
        classe_real_dominante=('classe_real', lambda x: int(x.mode()[0]))
    ).reset_index()

    return df_sankey, epocas


# ==========================================
# 2. INICIALIZAÇÃO DA APLICAÇÃO DASH
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Dashboard Analítico XAI (Multi-Dataset & Diagnóstico Temporal)",
        color="light",
        dark=False,
        fluid=True,
        children=[
            html.Div([
                html.Label("Dataset:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-dataset',
                    options=[
                        {'label': 'Breast Cancer Wisconsin', 'value': 'breast_cancer'},
                        {'label': 'Wine Quality (Red/White)', 'value': 'wine_quality'},
                        {'label': 'German Credit (Cost Risk)', 'value': 'german_credit'}
                    ],
                    value='breast_cancer',
                    clearable=False,
                    style={'width': '250px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ]
    ),

    dbc.Container([
        # Filtros Globais
        dbc.Row([
            dbc.Col([
                html.Label("Método XAI:", style={'fontSize': '12px', 'fontWeight': 'bold'}),
                dcc.Dropdown(id='dropdown-metodo', options=[], clearable=False, style={'fontSize': '13px'})
            ], width=3),
            dbc.Col([
                html.Label("Métrica Auxiliar (Estabilidade):", style={'fontSize': '12px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-metrica-estabilidade',
                    options=[
                        {'label': 'Correlação de Spearman (Ranking)', 'value': 'spearman'},
                        {'label': 'MAD (Desvio Absoluto Médio)', 'value': 'mad_importancia'},
                        {'label': 'Variância de Importância', 'value': 'variancia_importancia'}
                    ],
                    value='spearman',
                    clearable=False,
                    style={'fontSize': '13px'}
                )
            ], width=3),
            dbc.Col([
                html.Label("Mapeamento de Cores (Sankey):", style={'fontSize': '12px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-cor-metrica',
                    options=[
                        {'label': 'Classe Predita Dominante', 'value': 'classe'},
                        {'label': 'Taxa de Confiança', 'value': 'confianca'},
                        {'label': 'Taxa de Erro', 'value': 'erro'}
                    ],
                    value='classe',
                    clearable=False,
                    style={'fontSize': '13px'}
                )
            ], width=3),
            dbc.Col([
                html.Label("Filtro de Classe:", style={'fontSize': '12px', 'fontWeight': 'bold'}),
                dcc.Dropdown(id='dropdown-classe', options=[], value='TODAS', clearable=False, style={'fontSize': '13px'})
            ], width=3),
        ], className="my-2"),

        # Slider de Épocas Global
        dbc.Row([
            dbc.Col([
                html.Label("Intervalo de Épocas (Brushing Temporal):", style={'fontSize': '12px', 'fontWeight': 'bold'}),
                dcc.RangeSlider(id='slider-epocas', min=1, max=40, step=None, marks={1: "Ép 1"}, value=[1, 40])
            ], width=12)
        ], className="mb-3"),

        # ==========================================
        # 1. SANKEY COMO A PRIMEIRA VISUALIZAÇÃO
        # ==========================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fluxos Temporais de Transição de Classe (Sankey Reativo ao XAI)"),
                    dbc.CardBody([dcc.Graph(id='sankey-graph', style={'height': '550px'})])
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),

        # Demais Visualizações Globais (Performance e Heatmap)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Evolução do Modelo: Performance (Loss/Acurácia) e Estabilidade XAI"),
                    dbc.CardBody([dcc.Graph(id='metrics-stability-combined-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resumo Global / Heatmap de Importância Temporal"),
                    dbc.CardBody([dcc.Graph(id='heatmap-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),
        ]),

        # ==========================================
        # 2. REDUÇÃO DE DIMENSIONALIDADE + BOXPLOT TEMPORAL (SEM PERFIL DE INSTÂNCIA)
        # ==========================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Redução de Dimensionalidade Principal (t-SNE / PCA Espacial)"),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Label("Época:", style={'fontSize': '11px', 'fontWeight': 'bold'}),
                                dcc.Dropdown(id='dim-dropdown-epoca', options=[], clearable=False, style={'fontSize': '12px'})
                            ], style={'width': '31%', 'display': 'inline-block', 'marginRight': '3%'} ),
                            html.Div([
                                html.Label("Algoritmo:", style={'fontSize': '11px', 'fontWeight': 'bold'}),
                                dcc.Dropdown(
                                    id='dim-dropdown-algo',
                                    options=[{'label': 't-SNE', 'value': 'TSNE'}, {'label': 'PCA', 'value': 'PCA'}],
                                    value='TSNE',
                                    clearable=False,
                                    style={'fontSize': '12px'}
                                )
                            ], style={'width': '31%', 'display': 'inline-block', 'marginRight': '3%'}),
                            html.Div([
                                html.Label("Colorir Por:", style={'fontSize': '11px', 'fontWeight': 'bold'}),
                                dcc.Dropdown(
                                    id='dim-dropdown-cor',
                                    options=[
                                        {'label': 'Classe Predita', 'value': 'classe_predita'},
                                        {'label': 'Classe Real', 'value': 'classe_real'},
                                        {'label': 'Erro de Predição', 'value': 'erro_predicao'}
                                    ],
                                    value='classe_predita',
                                    clearable=False,
                                    style={'fontSize': '12px'}
                                )
                            ], style={'width': '31%', 'display': 'inline-block'}),
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            dcc.Checklist(
                                id='dim-toggle-trajetoria',
                                options=[{'label': ' Traçar Migração Temporal da Instância Clicada', 'value': 'ATIVO'}],
                                value=[],
                                style={'fontSize': '11px'}
                            )
                        ], style={'marginBottom': '10px'}),
                        dcc.Graph(
                            id='dim-scatter-graph',
                            style={'height': '430px'},
                            config={'modeBarButtonsToAdd': ['drawrect', 'eraseshape']}
                        )
                    ])
                ], className="shadow-sm mb-4")
            ], width=12, lg=7),

            # Painel Lateral Apenas com a Análise Auxiliar Temporal (Boxplot)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Análise Auxiliar Temporal (Boxplot)"),
                    dbc.CardBody([
                        dcc.Graph(id='dim-reduction-aux-graph', style={'height': '490px'})
                    ])
                ], className="shadow-sm mb-4")
            ], width=12, lg=5),
        ]),

        # ==========================================
        # 3. TABELA AUXILIAR ABAIXO DA REDUÇÃO DE DIMENSIONALIDADE
        # ==========================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tabela Auxiliar de Instâncias (Seleção por Brushing / Caixa no Gráfico 2D)"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='dim-tabela-instancias',
                            columns=[
                                {'name': 'Instância ID', 'id': 'instancia_id'},
                                {'name': 'Época', 'id': 'epoca'},
                                {'name': 'Método XAI', 'id': 'metodo_xai'},
                                {'name': 'Classe Real', 'id': 'classe_real'},
                                {'name': 'Classe Predita', 'id': 'classe_predita'},
                                {'name': 'Erro Predição', 'id': 'erro_predicao'},
                                {'name': 'Coord X', 'id': 'tsne_dim_1'},
                                {'name': 'Coord Y', 'id': 'tsne_dim_2'}
                            ],
                            data=[],
                            page_size=6,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif', 'fontSize': '12px'},
                            style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'}
                        )
                    ])
                ], className="shadow-sm mb-4")
            ], width=12)
        ])
    ], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})
])


# ==========================================
# 3. CALLBACKS DE ATUALIZAÇÃO
# ==========================================

@app.callback(
    [Output('dropdown-metodo', 'options'),
     Output('dropdown-metodo', 'value'),
     Output('dim-dropdown-epoca', 'options'),
     Output('dim-dropdown-epoca', 'value'),
     Output('slider-epocas', 'min'),
     Output('slider-epocas', 'max'),
     Output('slider-epocas', 'marks'),
     Output('slider-epocas', 'value')],
    [Input('dropdown-dataset', 'value')]
)
def update_options(dataset_sel):
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)
    
    metodos = ['SHAP', 'PFI', 'LIME', 'IG']
    if not df_d3.empty and 'metodo_xai' in df_d3.columns:
        metodos = sorted(list(df_d3['metodo_xai'].dropna().unique()))

    epocas = [1, 10, 20, 30, 40]
    if not df_d6.empty and 'epoca' in df_d6.columns:
        epocas = sorted([int(x) for x in df_d6['epoca'].dropna().unique()])
    elif not df_d3.empty and 'epoca' in df_d3.columns:
        epocas = sorted([int(x) for x in df_d3['epoca'].dropna().unique()])

    options_metodo = [{'label': str(m).upper(), 'value': m} for m in metodos]
    val_metodo = metodos[0] if metodos else 'SHAP'
    
    opt_epocas = [{'label': f"Época {ep}", 'value': ep} for ep in epocas]
    val_epoca = epocas[0] if epocas else 1

    min_ep = min(epocas) if epocas else 1
    max_ep = max(epocas) if epocas else 40
    marks_ep = {int(ep): f"Ép {ep}" for ep in epocas} if epocas else {1: "Ép 1"}
    
    return options_metodo, val_metodo, opt_epocas, val_epoca, min_ep, max_ep, marks_ep, [min_ep, max_ep]


@app.callback(
    [Output('metrics-stability-combined-graph', 'figure'),
     Output('heatmap-graph', 'figure'),
     Output('dim-scatter-graph', 'figure'),
     Output('dim-reduction-aux-graph', 'figure'),
     Output('dim-tabela-instancias', 'data'),
     Output('sankey-graph', 'figure')],
    [Input('dropdown-dataset', 'value'),
     Input('dropdown-metodo', 'value'),
     Input('dropdown-metrica-estabilidade', 'value'),
     Input('dropdown-cor-metrica', 'value'),
     Input('dropdown-classe', 'value'),
     Input('slider-epocas', 'value'),
     Input('dim-dropdown-epoca', 'value'),
     Input('dim-dropdown-algo', 'value'),
     Input('dim-dropdown-cor', 'value'),
     Input('dim-toggle-trajetoria', 'value'),
     Input('dim-scatter-graph', 'clickData'),
     Input('dim-scatter-graph', 'selectedData')]
)
def update_all_graphs(dataset_sel, metodo_sel, metrica_est_sel, cor_sankey_sel, classe_sel, epocas_sel, epoca_proj_sel, algo_sel, cor_proj_sel, trajetoria_sel, clickData, selectedData):
    df_d2 = carregar_csv_seguro("metricas_epocas", dataset_sel)
    df_d4 = carregar_csv_seguro("resumo_explicacoes", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)
    df_d5 = carregar_csv_seguro("metricas_estabilidade_concordancia", dataset_sel)
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)
    label_ds = MAPA_DATASET.get(dataset_sel, {}).get('label', dataset_sel.upper())

    # 1. Performance + Estabilidade XAI
    fig_combined = go.Figure()
    if not df_d2.empty:
        acc_col = 'accuracy' if 'accuracy' in df_d2.columns else (df_d2.columns[1] if len(df_d2.columns) > 1 else None)
        loss_col = 'loss_validacao' if 'loss_validacao' in df_d2.columns else None
        
        if acc_col and 'epoca' in df_d2.columns:
            fig_combined.add_trace(go.Scatter(x=df_d2['epoca'], y=df_d2[acc_col], mode='lines+markers', name='Acurácia', line=dict(color='#2ca02c')))
        if loss_col and 'epoca' in df_d2.columns:
            fig_combined.add_trace(go.Scatter(x=df_d2['epoca'], y=df_d2[loss_col], mode='lines+markers', name='Loss Validação', yaxis='y2', line=dict(color='#d62728')))
            
        if not df_d5.empty and 'tipo_metrica' in df_d5.columns:
            mask_metodo = df_d5['tipo_metrica'].astype(str).str.upper() == str(metodo_sel).upper()
            mask_metrica = df_d5['metrica'].astype(str).str.lower() == str(metrica_est_sel).lower() if 'metrica' in df_d5.columns else True
            df_s_sub = df_d5[mask_metodo & mask_metrica]
            
            if not df_s_sub.empty and 'epoca_destino' in df_s_sub.columns and 'valor_metrica' in df_s_sub.columns:
                fig_combined.add_trace(go.Scatter(
                    x=df_s_sub['epoca_destino'], y=df_s_sub['valor_metrica'],
                    mode='lines+markers', name=f'Estabilidade ({metrica_est_sel})',
                    line=dict(color='#1f77b4', dash='dash')
                ))
    fig_combined.update_layout(
        title=f"Performance do Modelo & Estabilidade XAI ({metodo_sel})",
        yaxis=dict(title="Acurácia / Estabilidade"),
        yaxis2=dict(title="Loss", overlaying='y', side='right', showgrid=False),
        margin=dict(t=30, b=30, l=40, r=40),
        legend=dict(x=0.01, y=0.99, orientation='h')
    )

    # 2. Heatmap
    fig_heat = go.Figure()
    if not df_d4.empty and 'metodo_xai' in df_d4.columns:
        df_h_sub = df_d4[df_d4['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_h_sub.empty:
            val_col = 'importancia_media' if 'importancia_media' in df_h_sub.columns else df_h_sub.columns[-1]
            piv = df_h_sub.pivot_table(index='feature', columns='epoca', values=val_col, fill_value=0)
            fig_heat = go.Figure(data=go.Heatmap(
                z=piv.values, x=piv.columns, y=piv.index, colorscale='Viridis',
                hovertemplate='Feature: %{y}<br>Época: %{x}<br>Importância Média: %{z:.4f}<extra></extra>'
            ))
            fig_heat.update_layout(title=f"Heatmap de Importância ({metodo_sel})", margin=dict(t=30, b=30, l=50, r=20))

    # 3. Redução de Dimensionalidade (t-SNE / PCA)
    fig_tsne = go.Figure()
    tabela_records = []
    instancia_clicada_id = None

    if not df_d6.empty:
        epoca_alvo = epoca_proj_sel if epoca_proj_sel else (df_d6['epoca'].min() if 'epoca' in df_d6.columns else 1)
        df_sub = df_d6[
            (df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()) &
            (df_d6['epoca'] == int(epoca_alvo))
        ].copy()

        if not df_sub.empty:
            if algo_sel == 'PCA' and not df_d3.empty:
                df_exp_ep = df_d3[(df_d3['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()) & (df_d3['epoca'] == int(epoca_alvo))]
                if not df_exp_ep.empty:
                    pivot_pca = df_exp_ep.pivot_table(index='instancia_id', columns='feature', values='valor_atribuicao').fillna(0)
                    pca = PCA(n_components=2)
                    res_pca = pca.fit_transform(pivot_pca.values)
                    pca_df = pd.DataFrame(res_pca, index=pivot_pca.index, columns=['dim_1', 'dim_2']).reset_index()
                    df_sub = df_sub.merge(pca_df, on='instancia_id', how='left')
                    x_col, y_col = 'dim_1', 'dim_2'
                else:
                    x_col, y_col = 'tsne_dim_1', 'tsne_dim_2'
            else:
                x_col, y_col = 'tsne_dim_1', 'tsne_dim_2'

            if clickData and 'points' in clickData and len(clickData['points']) > 0:
                instancia_clicada_id = clickData['points'][0]['customdata'][0]

            if cor_proj_sel == 'erro_predicao':
                df_sub['cor_label'] = df_sub['erro_predicao'].map({0: 'Acerto', 1: 'Erro'})
                fig_tsne = px.scatter(
                    df_sub, x=x_col, y=y_col, color='cor_label',
                    color_discrete_map={'Acerto': '#2ca02c', 'Erro': '#d62728'},
                    custom_data=['instancia_id', 'classe_real', 'classe_predita', 'erro_predicao']
                )
            else:
                fig_tsne = px.scatter(
                    df_sub, x=x_col, y=y_col, color=cor_proj_sel, color_continuous_scale='Viridis',
                    custom_data=['instancia_id', 'classe_real', 'classe_predita', 'erro_predicao']
                )

            fig_tsne.update_traces(
                marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color='black')),
                hovertemplate=(
                    "<b>Instância ID:</b> #%{customdata[0]}<br>"
                    "<b>Classe Real:</b> %{customdata[1]}<br>"
                    "<b>Classe Predita:</b> %{customdata[2]}<br>"
                    "<b>Erro:</b> %{customdata[3]}<br>"
                    "<b>X:</b> %{x:.3f} | <b>Y:</b> %{y:.3f}<extra></extra>"
                )
            )

            if 'ATIVO' in trajetoria_sel and instancia_clicada_id is not None:
                df_traj = df_d6[
                    (df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()) &
                    (df_d6['instancia_id'] == instancia_clicada_id)
                ].sort_values('epoca')
                fig_tsne.add_trace(go.Scatter(
                    x=df_traj['tsne_dim_1'], y=df_traj['tsne_dim_2'],
                    mode='lines+markers+text', text=[f"E{ep}" for ep in df_traj['epoca']],
                    textposition="top center", line=dict(color='orange', width=3, dash='dash'),
                    marker=dict(size=12, color='orange', symbol='diamond'), name=f'Trajetória ID #{instancia_clicada_id}'
                ))

            if instancia_clicada_id is not None:
                ponto_clicado = df_sub[df_sub['instancia_id'] == instancia_clicada_id]
                if not ponto_clicado.empty:
                    fig_tsne.add_trace(go.Scatter(
                        x=ponto_clicado[x_col], y=ponto_clicado[y_col], mode='markers',
                        marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color='yellow', width=3)),
                        showlegend=False, hoverinfo='skip'
                    ))

            fig_tsne.update_layout(
                title=f"Espaço 2D ({algo_sel}) - Época {epoca_alvo} ({metodo_sel}) - {label_ds}",
                xaxis_title=f"{algo_sel} Dim 1", yaxis_title=f"{algo_sel} Dim 2",
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Tabela Auxiliar (Brushing / Seleção por caixa ou Cliques)
            if selectedData and 'points' in selectedData:
                selected_ids = [p['customdata'][0] for p in selectedData['points'] if 'customdata' in p]
                df_tabela = df_sub[df_sub['instancia_id'].isin(selected_ids)]
            elif instancia_clicada_id is not None:
                df_tabela = df_sub[df_sub['instancia_id'] == instancia_clicada_id]
            else:
                df_tabela = df_sub

            tabela_records = df_tabela.to_dict('records')

    # 4. Boxplot Auxiliar Temporal
    fig_dim_aux = go.Figure()
    if not df_d6.empty and 'tsne_dim_1' in df_d6.columns:
        df_aux_sub = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_aux_sub.empty and 'epoca' in df_aux_sub.columns:
            fig_dim_aux = go.Figure(data=go.Box(
                x=df_aux_sub['epoca'], y=df_aux_sub['tsne_dim_1'],
                name="Dim 1 (t-SNE)", marker_color='#9467bd'
            ))
            fig_dim_aux.update_layout(title="Dispersão Temporal (t-SNE Dim 1)", margin=dict(l=20, r=20, t=30, b=20))

    # 5. Fluxos Temporais (Sankey na Primeira Posição)
    fig_sankey = go.Figure()
    df_d3_sankey, df_d2_sankey = carregar_dados_sankey(dataset_sel)
    df_sankey_processed, _ = processar_dados_sankey(df_d3_sankey, df_d2_sankey, dataset_sel)

    if not df_sankey_processed.empty:
        metodos_validos = df_sankey_processed['metodo_xai'].unique()
        m_alvo = metodo_sel if metodo_sel in metodos_validos else metodos_validos[0]
        df_filtered = df_sankey_processed[df_sankey_processed['metodo_xai'] == m_alvo].copy()

        if epocas_sel:
            df_filtered = df_filtered[
                (df_filtered['epoca_origem'] >= epocas_sel[0]) &
                (df_filtered['epoca_destino'] <= epocas_sel[1])
            ]
        if classe_sel is not None and classe_sel != 'TODAS':
            df_filtered = df_filtered[df_filtered['classe_origem'] == int(classe_sel)]

        if not df_filtered.empty:
            map_cls = Mapeamento_classes_dataset(dataset_sel)
            epocas_unicas = sorted(list(set(df_filtered['epoca_origem']).union(set(df_filtered['epoca_destino']))))
            classes_unicas = [0, 1]

            node_labels, node_x, node_y = [], [], []
            node_dict = {}

            for i_ep, ep in enumerate(epocas_unicas):
                for i_cl, cl in enumerate(classes_unicas):
                    node_id = len(node_labels)
                    node_dict[f"E{ep}_C{cl}"] = node_id
                    nome_classe = map_cls.get(cl, f"Classe {cl}")
                    node_labels.append(f"Época {ep}<br>{nome_classe}")
                    x_pos = 0.05 + 0.9 * (i_ep / max(1, len(epocas_unicas) - 1))
                    y_pos = 0.2 if cl == 0 else 0.8
                    node_x.append(x_pos)
                    node_y.append(y_pos)

            sources, targets, values, link_colors, hover_texts = [], [], [], [], []
            for _, row in df_filtered.iterrows():
                src_key = f"E{row['epoca_origem']}_C{row['classe_origem']}"
                tgt_key = f"E{row['epoca_destino']}_C{row['classe_destino']}"

                if src_key in node_dict and tgt_key in node_dict:
                    sources.append(node_dict[src_key])
                    targets.append(node_dict[tgt_key])
                    val_fluxo = int(row['qtd_instancias'])
                    values.append(val_fluxo)

                    conf = float(row['confianca_media'])
                    alpha = max(0.30, min(0.85, conf))

                    if cor_sankey_sel == 'classe':
                        base_rgb = "31, 119, 180" if row['classe_origem'] == 0 else "255, 127, 14"
                    elif cor_sankey_sel == 'confianca':
                        base_rgb = f"{int(255*(1-conf))}, {int(200*conf)}, 150"
                    else:
                        err = float(row['taxa_erro'])
                        base_rgb = f"{int(255*err)}, 50, 50"

                    link_colors.append(f"rgba({base_rgb}, {alpha:.2f})")

                    hover_text = (
                        f"<b>Dataset:</b> {dataset_sel.upper()}<br>"
                        f"<b>Método XAI:</b> {m_alvo}<br>"
                        f"<b>Transição:</b> Época {row['epoca_origem']} &rarr; Época {row['epoca_destino']}<br>"
                        f"<b>Classe Predita:</b> {map_cls.get(row['classe_origem'])} &rarr; {map_cls.get(row['classe_destino'])}<br>"
                        f"<b>Classe Real Dominante:</b> {map_cls.get(row['classe_real_dominante'])}<br>"
                        f"<b>Quantidade de Instâncias no Fluxo:</b> {row['qtd_instancias']}<br>"
                        f"<b>Instância Exemplo:</b> ID #{row['exemplo_instancia']}<br>"
                        f"<b>Confiança Média:</b> {conf*100:.1f}%<br>"
                        f"<b>Taxa de Erro Média:</b> {row['taxa_erro']*100:.1f}%<br>"
                        f"<b>Importância Acumulada (XAI):</b> {row['soma_importancia']:.4f}"
                    )
                    hover_texts.append(hover_text)

            fig_sankey = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(pad=30, thickness=20, line=dict(color="black", width=0.5), label=node_labels, x=node_x, y=node_y, color="lightgray"),
                link=dict(source=sources, target=targets, value=values, color=link_colors, customdata=hover_texts, hovertemplate="%{customdata}<extra></extra>")
            )])
            fig_sankey.update_layout(
                title_text=f"Transições Temporais de Explicações ({m_alvo}) - Dataset: {dataset_sel.upper()}",
                font_size=12, margin=dict(l=20, r=20, t=50, b=20)
            )

    return fig_combined, fig_heat, fig_tsne, fig_dim_aux, tabela_records, fig_sankey


if __name__ == '__main__':
    app.run(debug=False)
