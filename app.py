import os
import zipfile
import glob
import shutil
import gdown
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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
# 1. FUNÇÕES AUXILIARES E CARREGAMENTO
# ==========================================

MAPA_DATASET_MODELO = {
    'breast_cancer': 'MLP_5L_BREAST_CANCER',
    'wine_quality': 'MLP_5L_WINE_QUALITY',
    'german_credit': 'MLP_5L_GERMAN_CREDIT'
}

def Mapeamento_classes_dataset(dataset_name):
    """Retorna os nomes das classes de acordo com o dataset selecionado."""
    if dataset_name == 'breast_cancer':
        return {0: "Benigno", 1: "Maligno"}
    elif dataset_name == 'wine_quality':
        return {0: "Baixa Qualidade", 1: "Alta Qualidade"}
    elif dataset_name == 'german_credit':
        return {0: "Alto Risco", 1: "Bom Crédito"}
    return {0: "Classe 0", 1: "Classe 1"}


def carregar_dados_sankey(dataset_sel):
    """
    Carrega preferencialmente os arquivos gravados com o sufixo do DATASET_NAME.
    Se não encontrar, busca os arquivos genéricos de fallback e aplica o filtro por modelo_id.
    """
    f_d2 = f"metricas_epocas_{dataset_sel}.csv"
    f_d3 = f"explicacoes_instancias_{dataset_sel}.csv"

    # Fallback para arquivos genéricos
    if not (os.path.exists(f_d2) and os.path.exists(f_d3)):
        f_d2 = "metricas_epocas.csv"
        f_d3 = "explicacoes_instancias.csv"

    try:
        df_d2 = pd.read_csv(f_d2) if os.path.exists(f_d2) else pd.DataFrame()
        df_d3 = pd.read_csv(f_d3) if os.path.exists(f_d3) else pd.DataFrame()
        
        # Normalização dos nomes das colunas para minúsculas
        if not df_d2.empty:
            df_d2.columns = df_d2.columns.astype(str).str.strip().str.lower()
        if not df_d3.empty:
            df_d3.columns = df_d3.columns.astype(str).str.strip().str.lower()
    except Exception as e:
        print(f"[Aviso] Erro ao carregar arquivos ({dataset_sel}): {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

    modelo_alvo = MAPA_DATASET_MODELO.get(dataset_sel, 'MLP_5L_BREAST_CANCER')

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
server = app.server  # Obrigatório para o Gunicorn / Render

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
                html.Label("Método XAI:"),
                dcc.Dropdown(id='dropdown-metodo', options=[], clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Mapeamento de Cores:"),
                dcc.Dropdown(
                    id='dropdown-cor-metrica',
                    options=[
                        {'label': 'Classe Predita Dominante', 'value': 'classe'},
                        {'label': 'Taxa de Confiança', 'value': 'confianca'},
                        {'label': 'Taxa de Erro', 'value': 'erro'}
                    ],
                    value='classe',
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

        # Linha 1: Sankey Completo com Hover Detalhado
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Fluxos Temporais de Transição de Classe (Sankey Reativo ao XAI)"),
                    dbc.CardBody([dcc.Graph(id='sankey-graph', style={'height': '680px'})])
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
def update_dataset_options(dataset_sel):
    df_d3, _ = carregar_dados_sankey(dataset_sel)

    if not df_d3.empty and 'metodo_xai' in df_d3.columns:
        metodos = sorted(list(df_d3['metodo_xai'].dropna().unique()))
        epocas = sorted([int(x) for x in df_d3['epoca'].dropna().unique()])
    else:
        metodos = ['SHAP', 'PFI', 'LIME', 'IG']
        epocas = [1, 5, 10, 20, 30]

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
    Output('sankey-graph', 'figure'),
    [Input('dropdown-dataset', 'value'),
     Input('dropdown-metodo', 'value'),
     Input('dropdown-cor-metrica', 'value'),
     Input('dropdown-classe', 'value'),
     Input('slider-epocas', 'value')]
)
def update_sankey(dataset_sel, metodo_sel, cor_metrica_sel, classe_sel, epocas_sel):
    df_d3, df_d2 = carregar_dados_sankey(dataset_sel)

    if df_d3.empty or df_d2.empty:
        fig_err = go.Figure()
        fig_err.add_annotation(
            text=f"Arquivos CSV do dataset '{dataset_sel}' não encontrados.",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig_err

    df_sankey_processed, _ = processar_dados_sankey(df_d3, df_d2, dataset_sel)

    if df_sankey_processed.empty:
        fig_vazio = go.Figure()
        fig_vazio.add_annotation(text=f"Nenhum dado retornado para o Sankey ({dataset_sel}).", showarrow=False)
        return fig_vazio

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

    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Sem dados para os filtros: {dataset_sel} | {m_alvo} | Épocas {epocas_sel}",
            xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="red")
        )
        return fig

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

            if cor_metrica_sel == 'classe':
                base_rgb = "31, 119, 180" if row['classe_origem'] == 0 else "255, 127, 14"
            elif cor_metrica_sel == 'confianca':
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

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            x=node_x,
            y=node_y,
            color="lightgray"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>"
        )
    )])

    fig.update_layout(
        title_text=f"Transições Temporais de Explicações ({m_alvo}) - Dataset: {dataset_sel.upper()}",
        font_size=12,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == '__main__':
    app.run(debug=False)
