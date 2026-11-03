import os
import zipfile
import requests
import glob
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# ==========================================
# 0. DOWNLOAD ROBUSTO DO GOOGLE DRIVE
# ==========================================
DRIVE_FILE_ID = "16E8yydOSbV6RwlhVPW5in_lIQr7CcjZm"

def baixar_drive_robusto(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def preparar_dados():
    if not glob.glob("**/*metricas_epocas*.csv", recursive=True):
        print("[Início] Baixando ZIP do Google Drive...")
        try:
            zip_path = "dados_xai.zip"
            baixar_drive_robusto(DRIVE_FILE_ID, zip_path)
            
            print("[Extraindo] Descompactando arquivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("[Sucesso] Dados prontos para leitura!")
        except Exception as e:
            print(f"[Erro] Falha no download ou extração: {e}")

preparar_dados()

# ==========================================
# 1. FUNÇÕES DE CARREGAMENTO ULTRA-SEGURO
# ==========================================
def carregar_csv_seguro(nome_base, dataset_sel):
    padroes = [f"**/{nome_base}_{dataset_sel}.csv", f"**/{nome_base}.csv"]
    
    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)
        if arquivos:
            try:
                df = pd.read_csv(arquivos[0])
                # Limpeza profunda: remove BOM, espaços e padroniza para minúsculo
                df.columns = df.columns.str.replace('\ufeff', '').str.replace('ï»¿', '').str.strip().str.lower()
                
                if 'dataset_name' in df.columns:
                    df = df[df['dataset_name'].astype(str).str.strip().str.lower() == dataset_sel.lower()]
                return df
            except Exception as e:
                print(f"Erro ao ler {arquivos[0]}: {e}")
    return pd.DataFrame()

def Mapeamento_classes_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return {0: "Benigno", 1: "Maligno"}
    elif dataset_name == 'wine_quality':
        return {0: "Baixa Qualidade", 1: "Alta Qualidade"}
    elif dataset_name == 'german_credit':
        return {0: "Alto Risco", 1: "Bom Crédito"}
    return {0: "Classe 0", 1: "Classe 1"}

# ==========================================
# 2. INICIALIZAÇÃO DA APLICAÇÃO DASH
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Dashboard Analítico XAI (Multi-Dataset)",
        color="light",
        dark=False,
        fluid=True,
        children=[
            html.Div([
                html.Label("Dataset:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-dataset',
                    options=[
                        {'label': 'Breast Cancer', 'value': 'breast_cancer'},
                        {'label': 'Wine Quality', 'value': 'wine_quality'},
                        {'label': 'German Credit', 'value': 'german_credit'}
                    ],
                    value='breast_cancer',
                    clearable=False,
                    style={'width': '200px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ]
    ),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Método XAI:"),
                dcc.Dropdown(id='dropdown-metodo', options=[], clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Métrica Auxiliar (Estabilidade):"),
                dcc.Dropdown(
                    id='dropdown-metrica-estabilidade',
                    options=[
                        {'label': 'Correlação de Spearman (Ranking)', 'value': 'spearman'},
                        {'label': 'MAD (Desvio Absoluto Médio)', 'value': 'mad_importancia'},
                        {'label': 'Variância de Importância', 'value': 'variancia_importancia'}
                    ],
                    value='spearman',
                    clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("Filtro de Classe:"),
                dcc.Dropdown(id='dropdown-classe', options=[], value='TODAS', clearable=False)
            ], width=4),
        ], className="my-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Intervalo de Épocas (Brushing Temporal):"),
                dcc.RangeSlider(id='slider-epocas', min=1, max=40, step=None, marks={1: "Ép 1"}, value=[1, 40])
            ], width=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance do Modelo ao Longo das Épocas (Loss & Accuracy)"),
                    dbc.CardBody([dcc.Graph(id='metrics-epochs-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Resumo Global / Heatmap de Importância"),
                    dbc.CardBody([dcc.Graph(id='heatmap-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Redução de Dimensionalidade (t-SNE / PCA)"),
                    dbc.CardBody([dcc.Graph(id='tsne-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Métricas de Estabilidade e Concordância XAI"),
                    dbc.CardBody([dcc.Graph(id='stability-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fluxos Temporais de Transições (Sankey)"),
                    dbc.CardBody([dcc.Graph(id='sankey-graph', style={'height': '600px'})])
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
     Output('dropdown-classe', 'options'),
     Output('slider-epocas', 'min'),
     Output('slider-epocas', 'max'),
     Output('slider-epocas', 'marks'),
     Output('slider-epocas', 'value')],
    [Input('dropdown-dataset', 'value')]
)
def update_options(dataset_sel):
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)
    
    metodos = ['SHAP', 'PFI', 'LIME', 'IG']
    if not df_d3.empty and 'metodo_xai' in df_d3.columns:
        metodos = sorted(list(df_d3['metodo_xai'].dropna().unique()))

    epocas = [1, 10, 20, 30, 40]
    if not df_d3.empty and 'epoca' in df_d3.columns:
        epocas = sorted([int(x) for x in df_d3['epoca'].dropna().unique()])

    map_cls = Mapeamento_classes_dataset(dataset_sel)
    options_metodo = [{'label': str(m).upper(), 'value': m} for m in metodos]
    val_metodo = metodos[0] if metodos else 'SHAP'
    
    options_classe = [
        {'label': 'Todas as Classes', 'value': 'TODAS'},
        {'label': f"Classe 0 ({map_cls[0]})", 'value': 0},
        {'label': f"Classe 1 ({map_cls[1]})", 'value': 1}
    ]
    
    min_ep = min(epocas) if epocas else 1
    max_ep = max(epocas) if epocas else 40
    marks_ep = {int(ep): f"Ép {ep}" for ep in epocas} if epocas else {1: "Ép 1"}
    
    return options_metodo, val_metodo, options_classe, min_ep, max_ep, marks_ep, [min_ep, max_ep]

@app.callback(
    [Output('metrics-epochs-graph', 'figure'),
     Output('heatmap-graph', 'figure'),
     Output('tsne-graph', 'figure'),
     Output('stability-graph', 'figure'),
     Output('sankey-graph', 'figure')],
    [Input('dropdown-dataset', 'value'),
     Input('dropdown-metodo', 'value'),
     Input('dropdown-metrica-estabilidade', 'value'),
     Input('dropdown-classe', 'value'),
     Input('slider-epocas', 'value')]
)
def update_graphs(dataset_sel, metodo_sel, metrica_est_sel, classe_sel, epocas_sel):
    df_d2 = carregar_csv_seguro("metricas_epocas", dataset_sel)
    df_d4 = carregar_csv_seguro("resumo_explicacoes", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)
    df_d5 = carregar_csv_seguro("metricas_estabilidade_concordancia", dataset_sel)

    # 1. Gráfico de Métricas
    fig_metrics = go.Figure()
    if df_d2.empty:
        fig_metrics.add_annotation(text="Arquivo CSV de métricas não encontrado no ZIP.", showarrow=False)
    elif 'epoca' not in df_d2.columns:
        fig_metrics.add_annotation(text=f"Coluna 'epoca' ausente. Encontradas: {list(df_d2.columns)}", showarrow=False)
    else:
        acc_col = 'accuracy' if 'accuracy' in df_d2.columns else (df_d2.columns[1] if len(df_d2.columns) > 1 else None)
        loss_col = 'loss_validacao' if 'loss_validacao' in df_d2.columns else None
        
        if acc_col:
            fig_metrics.add_trace(go.Scatter(x=df_d2['epoca'], y=df_d2[acc_col], mode='lines+markers', name='Acurácia'))
        if loss_col:
            fig_metrics.add_trace(go.Scatter(x=df_d2['epoca'], y=df_d2[loss_col], mode='lines+markers', name='Loss Validação', yaxis='y2'))
            
        fig_metrics.update_layout(title="Evolução da Performance por Época", yaxis=dict(title="Acurácia"), yaxis2=dict(title="Loss", overlaying='y', side='right', showgrid=False), margin=dict(t=30, b=30, l=40, r=40), legend=dict(x=0.01, y=0.99))

    # 2. Heatmap
    fig_heat = go.Figure()
    if df_d4.empty:
        fig_heat.add_annotation(text="Arquivo CSV de resumo_explicacoes não encontrado.", showarrow=False)
    elif 'metodo_xai' not in df_d4.columns or 'feature' not in df_d4.columns or 'epoca' not in df_d4.columns:
        fig_heat.add_annotation(text=f"Colunas XAI/feature/epoca ausentes. Encontradas: {list(df_d4.columns)}", showarrow=False)
    else:
        df_h_sub = df_d4[df_d4['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_h_sub.empty:
            val_col = 'importancia_media' if 'importancia_media' in df_h_sub.columns else df_h_sub.columns[-1]
            piv = df_h_sub.pivot_table(index='feature', columns='epoca', values=val_col, fill_value=0)
            fig_heat = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale='Viridis'))
            fig_heat.update_layout(title=f"Heatmap de Importância ({metodo_sel})", margin=dict(t=30, b=30, l=50, r=20))

    # 3. t-SNE
    fig_tsne = go.Figure()
    if df_d6.empty:
        fig_tsne.add_annotation(text="Arquivo CSV de projecao_tsne não encontrado.", showarrow=False)
    elif 'metodo_xai' not in df_d6.columns or 'tsne_dim_1' not in df_d6.columns:
        fig_tsne.add_annotation(text=f"Colunas t-SNE ausentes. Encontradas: {list(df_d6.columns)}", showarrow=False)
    else:
        df_t_sub = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if epocas_sel and 'epoca' in df_t_sub.columns:
            df_t_sub = df_t_sub[(df_t_sub['epoca'] >= epocas_sel[0]) & (df_t_sub['epoca'] <= epocas_sel[1])]
        if not df_t_sub.empty:
            color_col = 'classe_predita' if 'classe_predita' in df_t_sub.columns else ('cluster_id' if 'cluster_id' in df_t_sub.columns else df_t_sub.columns[-1])
            fig_tsne = go.Figure(data=go.Scatter(x=df_t_sub['tsne_dim_1'], y=df_t_sub['tsne_dim_2'], mode='markers', marker=dict(color=df_t_sub[color_col], colorscale='Plasma', opacity=0.7)))
            fig_tsne.update_layout(title=f"Projeção t-SNE ({metodo_sel})", margin=dict(t=30, b=30, l=30, r=20))

    # 4. Estabilidade
    fig_stab = go.Figure()
    if df_d5.empty:
        fig_stab.add_annotation(text="Arquivo CSV de estabilidade não encontrado.", showarrow=False)
    elif 'tipo_metrica' not in df_d5.columns or 'epoca_destino' not in df_d5.columns:
        fig_stab.add_annotation(text=f"Colunas de estabilidade ausentes. Encontradas: {list(df_d5.columns)}", showarrow=False)
    else:
        mask_metodo = df_d5['tipo_metrica'].astype(str).str.upper() == str(metodo_sel).upper()
        mask_metrica = df_d5['metrica'].astype(str).str.lower() == str(metrica_est_sel).lower() if 'metrica' in df_d5.columns else True
        df_s_sub = df_d5[mask_metodo & mask_metrica]
        if not df_s_sub.empty and 'valor_metrica' in df_s_sub.columns:
            fig_stab = go.Figure(data=go.Scatter(x=df_s_sub['epoca_destino'], y=df_s_sub['valor_metrica'], mode='lines+markers', name=metrica_est_sel))
            fig_stab.update_layout(title=f"Estabilidade ({metrica_est_sel}) - {metodo_sel}", margin=dict(t=30, b=30, l=40, r=20))

    # 5. Sankey Placeholder
    fig_sankey = go.Figure()
    fig_sankey.add_annotation(text="Sankey pronto - Aguardando processamento temporal", showarrow=False, x=0.5, y=0.5, font=dict(size=14))
    fig_sankey.update_layout(title="Transições Temporais (Sankey)", margin=dict(t=30, b=30, l=30, r=30))

    return fig_metrics, fig_heat, fig_tsne, fig_stab, fig_sankey

if __name__ == '__main__':
    app.run(debug=False)
