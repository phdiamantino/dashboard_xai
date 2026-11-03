import os
import zipfile
import glob
import shutil
import gdown
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc

# ==========================================
# 0. DOWNLOAD ROBUSTO VIA GDOWN (GOOGLE DRIVE)
# ==========================================
DRIVE_FILE_ID = "16E8yydOSbV6RwlhVPW5in_lIQr7CcjZm"

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
                            
            print("[Sucesso] Dados baixados e organizados com sucesso!")
        except Exception as e:
            print(f"[Erro] Falha grave no download/extração: {e}")

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
                df.columns = df.columns.astype(str).str.replace('\ufeff', '').str.replace('ï»¿', '').str.strip().str.lower()
                
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
        brand="Dashboard Analítico XAI & Diagnóstico Temporal de Dimensionalidade",
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
                        {'label': 'German Credit (Pré-processado)', 'value': 'german_credit'}
                    ],
                    value='breast_cancer',
                    clearable=False,
                    style={'width': '220px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ]
    ),

    dbc.Container([
        # Filtros Globais
        dbc.Row([
            dbc.Col([
                html.Label("Método XAI:"),
                dcc.Dropdown(id='dropdown-metodo', options=[], clearable=False)
            ], width=3),
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
            ], width=3),
            dbc.Col([
                html.Label("Algoritmo de Projeção (2D):"),
                dcc.Dropdown(
                    id='dropdown-algo-proj',
                    options=[
                        {'label': 't-SNE', 'value': 'TSNE'},
                        {'label': 'PCA (Recalculado)', 'value': 'PCA'}
                    ],
                    value='TSNE',
                    clearable=False
                )
            ], width=3),
            dbc.Col([
                html.Label("Filtro de Classe:"),
                dcc.Dropdown(id='dropdown-classe', options=[], value='TODAS', clearable=False)
            ], width=3),
        ], className="my-3"),

        # Slider de Épocas
        dbc.Row([
            dbc.Col([
                html.Label("Intervalo de Épocas (Brushing Temporal):"),
                dcc.RangeSlider(id='slider-epocas', min=1, max=40, step=None, marks={1: "Ép 1"}, value=[1, 40])
            ], width=12)
        ], className="mb-4"),

        # Linha 1: Performance + Estabilidade XAI integradas E Heatmap
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

        # Linha 2: Redução de Dimensionalidade Principal e Inspeção Auxiliar Local
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Redução de Dimensionalidade 2D (Clique em um ponto para inspecionar)"),
                    dbc.CardBody([dcc.Graph(id='tsne-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=7),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualização Auxiliar: Perfil de Importância da Instância Selecionada"),
                    dbc.CardBody([
                        html.Div(id='info-instancia-selecionada', style={'marginBottom': '10px', 'fontSize': '13px'}),
                        dcc.Graph(id='local-xai-bar-graph', style={'height': '350px'})
                    ])
                ], className="shadow-sm mb-4")
            ], width=12, lg=5),
        ]),

        # Linha 3: Sankey
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fluxos Temporais de Transições entre Épocas (Sankey)"),
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
    [Output('metrics-stability-combined-graph', 'figure'),
     Output('heatmap-graph', 'figure'),
     Output('tsne-graph', 'figure'),
     Output('info-instancia-selecionada', 'children'),
     Output('local-xai-bar-graph', 'figure'),
     Output('sankey-graph', 'figure')],
    [Input('dropdown-dataset', 'value'),
     Input('dropdown-metodo', 'value'),
     Input('dropdown-metrica-estabilidade', 'value'),
     Input('dropdown-algo-proj', 'value'),
     Input('dropdown-classe', 'value'),
     Input('slider-epocas', 'value'),
     Input('tsne-graph', 'clickData')]
)
def update_graphs(dataset_sel, metodo_sel, metrica_est_sel, algo_proj_sel, classe_sel, epocas_sel, clickData):
    df_d2 = carregar_csv_seguro("metricas_epocas", dataset_sel)
    df_d4 = carregar_csv_seguro("resumo_explicacoes", dataset_sel)
    df_d6 = carregar_csv_seguro("projecao_tsne_epocas", dataset_sel)
    df_d5 = carregar_csv_seguro("metricas_estabilidade_concordancia", dataset_sel)
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)

    # 1. Performance do Modelo + Estabilidade XAI Integradas
    fig_combined = go.Figure()
    if df_d2.empty:
        fig_combined.add_annotation(text="Dados de métricas de treino não encontrados.", showarrow=False)
    else:
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

    # 2. Heatmap de Importância
    fig_heat = go.Figure()
    if df_d4.empty:
        fig_heat.add_annotation(text="Arquivo CSV de resumo_explicacoes não encontrado.", showarrow=False)
    elif 'metodo_xai' not in df_d4.columns or 'feature' not in df_d4.columns or 'epoca' not in df_d4.columns:
        fig_heat.add_annotation(text=f"Colunas XAI/feature/epoca ausentes.", showarrow=False)
    else:
        df_h_sub = df_d4[df_d4['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_h_sub.empty:
            val_col = 'importancia_media' if 'importancia_media' in df_h_sub.columns else df_h_sub.columns[-1]
            piv = df_h_sub.pivot_table(index='feature', columns='epoca', values=val_col, fill_value=0)
            fig_heat = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale='Viridis'))
            fig_heat.update_layout(title=f"Heatmap de Importância ({metodo_sel})", margin=dict(t=30, b=30, l=50, r=20))

    # 3. Projeção Espacial 2D (t-SNE / PCA Recalculado)
    fig_proj = go.Figure()
    instancia_clicada_id = None
    if clickData and 'points' in clickData and len(clickData['points']) > 0:
        instancia_clicada_id = clickData['points'][0].get('customdata', [None])[0]

    if df_d6.empty:
        fig_proj.add_annotation(text="Arquivo CSV de projeção não encontrado.", showarrow=False)
    else:
        df_p_sub = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()].copy()
        
        if epocas_sel and 'epoca' in df_p_sub.columns:
            df_p_sub = df_p_sub[(df_p_sub['epoca'] >= epocas_sel[0]) & (df_p_sub['epoca'] <= epocas_sel[1])]

        if not df_p_sub.empty:
            x_col, y_col = 'tsne_dim_1', 'tsne_dim_2'
            
            # Recálculo de PCA se selecionado e dados do Dataset 3 estiverem disponíveis
            if algo_proj_sel == 'PCA' and not df_d3.empty and 'feature' in df_d3.columns and 'valor_atribuicao' in df_d3.columns:
                id_inst_col = next((c for c in ['instancia_id', 'instance_id', 'id'] if c in df_d3.columns), None)
                if id_inst_col:
                    piv_pca = df_d3.pivot_table(index=id_inst_col, columns='feature', values='valor_atribuicao').fillna(0)
                    if len(piv_pca) > 2:
                        pca = PCA(n_components=2)
                        res_pca = pca.fit_transform(piv_pca.values)
                        df_pca = pd.DataFrame(res_pca, index=piv_pca.index, columns=['pca_dim_1', 'pca_dim_2']).reset_index()
                        df_p_sub = df_p_sub.merge(df_pca, left_on=next((c for c in ['instancia_id', 'instance_id', 'id'] if c in df_p_sub.columns), id_inst_col), right_on=id_inst_col, how='left')
                        x_col, y_col = 'pca_dim_1', 'pca_dim_2'

            color_col = 'classe_predita' if 'classe_predita' in df_p_sub.columns else ('cluster_id' if 'cluster_id' in df_p_sub.columns else df_p_sub.columns[-1])
            id_custom_col = next((c for c in ['instancia_id', 'instance_id', 'id'] if c in df_p_sub.columns), color_col)
            
            fig_proj = px.scatter(
                df_p_sub, x=x_col, y=y_col, color=color_col,
                color_continuous_scale='Plasma', custom_data=[id_custom_col]
            )
            fig_proj.update_traces(marker=dict(size=8, opacity=0.8))
            fig_proj.update_layout(title=f"Projeção Espacial 2D ({algo_proj_sel}) - {metodo_sel}", margin=dict(t=30, b=30, l=30, r=20))

    # 4. Visualização Auxiliar Local: Top Features da Instância Clicada
    fig_local_bar = go.Figure()
    info_html = "Clique em um ponto do gráfico 2D para inspecionar os detalhes da instância."
    
    if instancia_clicada_id is not None and not df_d3.empty:
        id_inst_d3 = next((c for c in ['instancia_id', 'instance_id', 'id'] if c in df_d3.columns), None)
        if id_inst_d3:
            df_local = df_d3[
                (df_d3['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()) &
                (df_d3[id_inst_d3] == instancia_clicada_id)
            ]
            if not df_local.empty and 'valor_atribuicao' in df_local.columns and 'feature' in df_local.columns:
                df_local_top = df_local.sort_values('valor_atribuicao', key=abs, ascending=True).tail(10)
                colors_bar = ['#d62728' if v < 0 else '#1f77b4' for v in df_local_top['valor_atribuicao']]
                
                fig_local_bar.add_trace(go.Bar(
                    x=df_local_top['valor_atribuicao'], y=df_local_top['feature'],
                    orientation='h', marker_color=colors_bar
                ))
                fig_local_bar.update_layout(
                    title=f"Top 10 Features (Instância #{instancia_clicada_id})",
                    xaxis_title="Atribuição de Importância", yaxis_title="Feature",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                info_html = html.Div([
                    html.Span(f"Instância Selecionada: #{instancia_clicada_id}", style={'fontWeight': 'bold', 'color': '#1f77b4'})
                ])

    if fig_local_bar.data == ():
        fig_local_bar.add_annotation(text="Nenhuma instância selecionada.", showarrow=False)

    # 5. Fluxos Temporais (Sankey Robusto)
    fig_sankey = go.Figure()
    if df_d3.empty:
        fig_sankey.add_annotation(text="Arquivo CSV explicacoes_instancias não encontrado.", showarrow=False)
    else:
        df_sankey_sub = df_d3.copy()
        if 'metodo_xai' in df_sankey_sub.columns:
            df_sankey_sub = df_sankey_sub[df_sankey_sub['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]

        id_col = next((c for c in ['instance_id', 'instancia_id', 'instancia', 'id_instancia', 'id', 'sample_id'] if c in df_sankey_sub.columns), None)
        if not id_col:
            df_sankey_sub['auto_instance_id'] = df_sankey_sub.groupby('epoca').cumcount() if 'epoca' in df_sankey_sub.columns else range(len(df_sankey_sub))
            id_col = 'auto_instance_id'

        target_col = next((c for c in ['classe_predita', 'cluster_id', 'predicao', 'y_pred', 'target_pred', 'pred', 'classe_real'] if c in df_sankey_sub.columns), None)

        if not target_col:
            fig_sankey.add_annotation(text="Coluna de classe predita/cluster não localizada.", showarrow=False)
        else:
            if epocas_sel and 'epoca' in df_sankey_sub.columns:
                df_sankey_sub = df_sankey_sub[(df_sankey_sub['epoca'] >= epocas_sel[0]) & (df_sankey_sub['epoca'] <= epocas_sel[1])]

            epocas_disponiveis = sorted(df_sankey_sub['epoca'].unique()) if 'epoca' in df_sankey_sub.columns else []

            if len(epocas_disponiveis) < 2:
                fig_sankey.add_annotation(text="Selecione um intervalo com pelo menos 2 épocas no slider para exibir o fluxo.", showarrow=False)
            else:
                nodes = []
                sources, targets, values = [], [], []
                node_map = {}
                map_cls = Mapeamento_classes_dataset(dataset_sel)

                for ep in epocas_disponiveis:
                    grupos = df_sankey_sub[df_sankey_sub['epoca'] == ep][target_col].unique()
                    for g in grupos:
                        try:
                            g_int = int(g)
                            label_g = map_cls.get(g_int, f"Classe {g}")
                        except (ValueError, TypeError):
                            label_g = f"Grupo {g}"
                        node_name = f"Ép {ep}: {label_g}"
                        node_map[(ep, g)] = len(nodes)
                        nodes.append(node_name)

                for i in range(len(epocas_disponiveis) - 1):
                    ep1, ep2 = epocas_disponiveis[i], epocas_disponiveis[i+1]
                    df_e1 = df_sankey_sub[df_sankey_sub['epoca'] == ep1][[id_col, target_col]]
                    df_e2 = df_sankey_sub[df_sankey_sub['epoca'] == ep2][[id_col, target_col]]

                    merged = pd.merge(df_e1, df_e2, on=id_col, suffixes=('_src', '_tgt'))
                    trans = merged.groupby([f'{target_col}_src', f'{target_col}_tgt']).size().reset_index(name='count')

                    for _, row in trans.iterrows():
                        src_key = (ep1, row[f'{target_col}_src'])
                        tgt_key = (ep2, row[f'{target_col}_tgt'])
                        if src_key in node_map and tgt_key in node_map:
                            sources.append(node_map[src_key])
                            targets.append(node_map[tgt_key])
                            values.append(row['count'])

                if sources and targets:
                    fig_sankey = go.Figure(data=[go.Sankey(
                        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color="#1f77b4"),
                        link=dict(source=sources, target=targets, value=values, color="rgba(160, 160, 160, 0.4)")
                    )])
                    fig_sankey.update_layout(title=f"Fluxos Temporais de Transição de Classe ({metodo_sel})", margin=dict(t=30, b=30, l=30, r=30))
                else:
                    fig_sankey.add_annotation(text="Sem transições registradas para o intervalo de épocas selecionado.", showarrow=False)

    return fig_combined, fig_heat, fig_proj, info_html, fig_local_bar, fig_sankey

if __name__ == '__main__':
    app.run(debug=False)
