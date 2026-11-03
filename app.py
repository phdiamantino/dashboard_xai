import os
import zipfile
import glob
import shutil
import gdown
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
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
                            
            print("[Sucesso] Dados baixados e organizados na raiz com sucesso!")
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
        # Filtros Globais
        dbc.Row([
            dbc.Col([
                html.Label("Método XAI:"),
                dcc.Dropdown(id='dropdown-metodo', options=[], clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Métrica Auxiliar (Estabilidade XAI):"),
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

        # Slider de Épocas
        dbc.Row([
            dbc.Col([
                html.Label("Intervalo de Épocas (Brushing Temporal):"),
                dcc.RangeSlider(id='slider-epocas', min=1, max=40, step=None, marks={1: "Ép 1"}, value=[1, 40])
            ], width=12)
        ], className="mb-4"),

        # Linha 1: Performance + Estabilidade XAI integradas e Heatmap
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

        # Linha 2: Redução de Dimensionalidade Principal (t-SNE) e Visualização Auxiliar
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Redução de Dimensionalidade Principal (t-SNE / Projeção Espacial)"),
                    dbc.CardBody([dcc.Graph(id='tsne-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Análise Auxiliar de Dimensionalidade (Dispersão / Variação Temporal)"),
                    dbc.CardBody([dcc.Graph(id='dim-reduction-aux-graph')])
                ], className="shadow-sm mb-4")
            ], width=12, lg=6),
        ]),

        # Linha 3: Sankey Dinâmico
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fluxos Temporais de Transições de Classe (Sankey Reativo XAI)"),
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
     Output('dim-reduction-aux-graph', 'figure'),
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
    df_d3 = carregar_csv_seguro("explicacoes_instancias", dataset_sel)

    # 1. Gráfico Combinado: Acurácia, Loss e Estabilidade XAI
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
        fig_heat.add_annotation(text="Colunas XAI/feature/epoca ausentes.", showarrow=False)
    else:
        df_h_sub = df_d4[df_d4['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_h_sub.empty:
            val_col = 'importancia_media' if 'importancia_media' in df_h_sub.columns else df_h_sub.columns[-1]
            piv = df_h_sub.pivot_table(index='feature', columns='epoca', values=val_col, fill_value=0)
            fig_heat = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale='Viridis'))
            fig_heat.update_layout(title=f"Heatmap de Importância ({metodo_sel})", margin=dict(t=30, b=30, l=50, r=20))

    # 3. Projeção t-SNE Principal
    fig_tsne = go.Figure()
    if df_d6.empty:
        fig_tsne.add_annotation(text="Arquivo CSV de projecao_tsne não encontrado.", showarrow=False)
    elif 'metodo_xai' not in df_d6.columns or 'tsne_dim_1' not in df_d6.columns:
        fig_tsne.add_annotation(text="Colunas t-SNE ausentes.", showarrow=False)
    else:
        df_t_sub = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if epocas_sel and 'epoca' in df_t_sub.columns:
            df_t_sub = df_t_sub[(df_t_sub['epoca'] >= epocas_sel[0]) & (df_t_sub['epoca'] <= epocas_sel[1])]
        if not df_t_sub.empty:
            color_col = 'classe_predita' if 'classe_predita' in df_t_sub.columns else ('cluster_id' if 'cluster_id' in df_t_sub.columns else df_t_sub.columns[-1])
            fig_tsne = go.Figure(data=go.Scatter(x=df_t_sub['tsne_dim_1'], y=df_t_sub['tsne_dim_2'], mode='markers', marker=dict(color=df_t_sub[color_col], colorscale='Plasma', opacity=0.7)))
            fig_tsne.update_layout(title=f"Projeção t-SNE Espacial ({metodo_sel})", margin=dict(t=30, b=30, l=30, r=20))

    # 4. Visualização Auxiliar de Redução de Dimensionalidade
    fig_dim_aux = go.Figure()
    if df_d6.empty or 'tsne_dim_1' not in df_d6.columns:
        fig_dim_aux.add_annotation(text="Dados de projeção de dimensionalidade não disponíveis.", showarrow=False)
    else:
        df_aux_sub = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()]
        if not df_aux_sub.empty and 'epoca' in df_aux_sub.columns:
            fig_dim_aux = go.Figure(data=go.Box(
                x=df_aux_sub['epoca'], y=df_aux_sub['tsne_dim_1'],
                name="Dimensão 1 (t-SNE)", marker_color='#9467bd'
            ))
            fig_dim_aux.update_layout(title=f"Dispersão da Componente Principal por Época ({metodo_sel})", margin=dict(t=30, b=30, l=40, r=20))

    # 5. Fluxos Temporais (SANKEY REATIVO COM CORES E OPACIDADE)
    fig_sankey = go.Figure()
    if df_d3.empty or df_d6.empty:
        fig_sankey.add_annotation(text="Dados de explicações/projeções não encontrados.", showarrow=False)
    else:
        id_col = next((c for c in ['instancia_id', 'instance_id', 'id', 'sample_id'] if c in df_d3.columns), None)
        id_col_d6 = next((c for c in ['instancia_id', 'instance_id', 'id', 'sample_id'] if c in df_d6.columns), None)
        
        # Filtra pelo Método XAI Selecionado (Garante reatividade completa)
        df_d3_met = df_d3[df_d3['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()].copy()
        df_d6_met = df_d6[df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()].copy()

        if not df_d3_met.empty and not df_d6_met.empty and id_col and id_col_d6:
            # Agrupa a intensidade atributiva total da instância por época para dar o perfil do Método XAI
            df_xai_profile = df_d3_met.groupby(['epoca', id_col])['valor_atribuicao'].sum().reset_index()
            
            # Merge com D6 para obter a classe predita e real
            cols_d6 = ['epoca', id_col_d6, 'classe_predita']
            if 'classe_real' in df_d6_met.columns:
                cols_d6.append('classe_real')
                
            df_sankey_base = pd.merge(
                df_xai_profile,
                df_d6_met[cols_d6].drop_duplicates(),
                left_on=['epoca', id_col],
                right_on=['epoca', id_col_d6],
                how='inner'
            )

            if classe_sel != 'TODAS' and 'classe_real' in df_sankey_base.columns:
                try:
                    df_sankey_base = df_sankey_base[df_sankey_base['classe_real'] == int(classe_sel)]
                except (ValueError, TypeError):
                    pass

            if epocas_sel and 'epoca' in df_sankey_base.columns:
                df_sankey_base = df_sankey_base[(df_sankey_base['epoca'] >= epocas_sel[0]) & (df_sankey_base['epoca'] <= epocas_sel[1])]

            epocas_disponiveis = sorted(df_sankey_base['epoca'].unique()) if 'epoca' in df_sankey_base.columns else []

            if len(epocas_disponiveis) < 2:
                fig_sankey.add_annotation(text="Selecione um intervalo com pelo menos 2 épocas para visualizar os fluxos.", showarrow=False)
            else:
                nodes = []
                node_colors = []
                sources, targets, values, link_colors = [], [], [], []
                node_map = {}
                map_cls = Mapeamento_classes_dataset(dataset_sel)

                # Cores distintas com transparência por classe (Estilo elegante da imagem)
                class_color_map = {
                    0: {'node': '#1f77b4', 'link': 'rgba(31, 119, 180, 0.45)'},    # Azul Benigno/Classe 0
                    1: {'node': '#ff7f0e', 'link': 'rgba(255, 127, 14, 0.45)'},    # Laranja Maligno/Classe 1
                    2: {'node': '#2ca02c', 'link': 'rgba(44, 160, 44, 0.45)'},     # Verde
                    3: {'node': '#d62728', 'link': 'rgba(214, 39, 40, 0.45)'}      # Vermelho
                }

                target_col = 'classe_predita'

                for ep in epocas_disponiveis:
                    grupos = sorted(df_sankey_base[df_sankey_base['epoca'] == ep][target_col].unique())
                    for g in grupos:
                        try:
                            g_int = int(g)
                            label_g = map_cls.get(g_int, f"Classe {g}")
                        except (ValueError, TypeError):
                            g_int = 0
                            label_g = f"Grupo {g}"

                        node_name = f"Ép {ep}: {label_g}"
                        node_map[(ep, g)] = len(nodes)
                        nodes.append(node_name)
                        
                        # Atribui a cor do nó
                        c_info = class_color_map.get(g_int, {'node': '#7f7f7f', 'link': 'rgba(127, 127, 127, 0.4)'})
                        node_colors.append(c_info['node'])

                for i in range(len(epocas_disponiveis) - 1):
                    ep1, ep2 = epocas_disponiveis[i], epocas_disponiveis[i+1]
                    df_e1 = df_sankey_base[df_sankey_base['epoca'] == ep1][[id_col, target_col]]
                    df_e2 = df_sankey_base[df_sankey_base['epoca'] == ep2][[id_col, target_col]]

                    merged = pd.merge(df_e1, df_e2, on=id_col, suffixes=('_src', '_tgt'))
                    trans = merged.groupby([f'{target_col}_src', f'{target_col}_tgt']).size().reset_index(name='count')

                    for _, row in trans.iterrows():
                        src_grp = row[f'{target_col}_src']
                        tgt_grp = row[f'{target_col}_tgt']
                        src_key = (ep1, src_grp)
                        tgt_key = (ep2, tgt_grp)

                        if src_key in node_map and tgt_key in node_map:
                            sources.append(node_map[src_key])
                            targets.append(node_map[tgt_key])
                            values.append(row['count'])
                            
                            # Define a cor do fluxo/link com transparência basada na classe de origem
                            try:
                                src_int = int(src_grp)
                            except (ValueError, TypeError):
                                src_int = 0
                            
                            l_color = class_color_map.get(src_int, {'link': 'rgba(160, 160, 160, 0.4)'})['link']
                            link_colors.append(l_color)

                if sources and targets:
                    fig_sankey = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=20,
                            thickness=18,
                            line=dict(color="black", width=0.5),
                            label=nodes,
                            color=node_colors
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color=link_colors
                        )
                    )])
                    fig_sankey.update_layout(
                        title=f"Fluxos Temporais de Transição de Classe ({metodo_sel.upper()})",
                        font=dict(size=12, color="black"),
                        margin=dict(t=40, b=30, l=30, r=30)
                    )
                else:
                    fig_sankey.add_annotation(text="Sem transições registradas para o filtro selecionado.", showarrow=False)

    return fig_combined, fig_heat, fig_tsne, fig_dim_aux, fig_sankey

if __name__ == '__main__':
    app.run(debug=False)
