import glob
import os
import shutil
import zipfile
import dash
import dash_bootstrap_components as dbc
import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ==========================================
# 0. DOWNLOAD ROBUSTO VIA GDOWN (GOOGLE DRIVE)
# ==========================================
DRIVE_FILE_ID = '125OsCXEBfGhiLNXyjBULAJKKbVvUNr7Q'


def preparar_dados():
    if not glob.glob('**/*.csv', recursive=True):
        print('[Início] Baixando arquivo ZIP do Google Drive via gdown...')
        try:
            zip_path = 'dados_xai.zip'
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            gdown.download(url, zip_path, quiet=False)

            print('[Extraindo] Descompactando arquivos...')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')

            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.csv'):
                        src_path = os.path.join(root, file)
                        dst_path = file
                        if src_path != dst_path and not os.path.exists(dst_path):
                            shutil.move(src_path, dst_path)

            print('[Sucesso] Dados baixados e organizados na raiz com sucesso!')
        except Exception as e:
            print(f'[Erro] Falha grave no download/extração: {e}')


preparar_dados()

COLORSCALE_REDS_20 = [
    [0.0, 'rgb(254, 224, 210)'],
    [0.25, 'rgb(252, 146, 114)'],
    [0.5, 'rgb(251, 106, 74)'],
    [0.75, 'rgb(222, 45, 38)'],
    [1.0, 'rgb(165, 15, 21)'],
]

COLORSCALE_BLUES_20 = [
    [0.0, 'rgb(222, 235, 247)'],
    [0.25, 'rgb(158, 202, 225)'],
    [0.5, 'rgb(66, 146, 198)'],
    [0.75, 'rgb(33, 113, 181)'],
    [1.0, 'rgb(8, 48, 107)'],
]

METODO_COMBINADO_ID = 'MEDIA_GERAL'
METODO_COMBINADO_LABEL = 'Combinado (Média Geral XAI)'

# ==========================================
# 1. MAPEAMENTO, NORMALIZAÇÃO E CARREGAMENTO
# ==========================================
MAPA_DATASET = {
    'breast_cancer': {
        'arquivo_d6': 'projecao_tsne_epocas_breast_cancer.csv',
        'arquivo_d3': 'explicacoes_instancias_breast_cancer.csv',
        'label': 'Breast Cancer Wisconsin',
        'modelo_id': 'MLP_5L_BREAST_CANCER',
    },
    'wine_quality': {
        'arquivo_d6': 'projecao_tsne_epocas_wine_quality.csv',
        'arquivo_d3': 'explicacoes_instancias_wine_quality.csv',
        'label': 'Wine Quality (Red/White)',
        'modelo_id': 'MLP_5L_WINE_QUALITY',
    },
    'german_credit': {
        'arquivo_d6': 'projecao_tsne_epocas_german_credit.csv',
        'arquivo_d3': 'explicacoes_instancias_german_credit.csv',
        'label': 'German Credit (Cost Risk)',
        'modelo_id': 'MLP_5L_GERMAN_CREDIT',
    },
}


def Mapeamento_classes_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return {0: 'Benigno', 1: 'Maligno'}
    elif dataset_name == 'wine_quality':
        return {0: 'Baixa Qualidade', 1: 'Alta Qualidade'}
    elif dataset_name == 'german_credit':
        return {0: 'Alto Risco', 1: 'Bom Crédito'}
    return {0: 'Classe 0', 1: 'Classe 1'}


def detectar_coluna_feature(df):
    candidatos = [
        'feature',
        'atributo',
        'nome_feature',
        'feature_name',
        'variavel',
        'coluna',
    ]
    for col in candidatos:
        if col in df.columns:
            return col
    ignoradas = {
        'modelo_id',
        'dataset_name',
        'metodo_xai',
        'instancia_id',
        'instance_id',
        'id',
        'sample_id',
    }
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in ignoradas:
            return col
    return None


def normalizar_coluna_por_grupo(df, col_valor, cols_grupo=['metodo_xai']):
    """Aplica normalização Min-Max por método XAI ao longo de todas as épocas."""
    if df.empty or col_valor not in df.columns:
        return df

    df = df.copy()
    grupos_validos = [c for c in cols_grupo if c in df.columns]

    if not grupos_validos:
        v_min = df[col_valor].min()
        v_max = df[col_valor].max()
        denom = (v_max - v_min) if (v_max - v_min) > 1e-12 else 1.0
        df[col_valor] = (df[col_valor] - v_min) / denom
        return df

    def _minmax(s):
        s_min = s.min()
        s_max = s.max()
        denom = s_max - s_min
        if denom <= 1e-12:
            return pd.Series(np.where(s > 0, 1.0, 0.0), index=s.index)
        return (s - s_min) / denom

    df[col_valor] = df.groupby(grupos_validos)[col_valor].transform(_minmax)
    return df


def carregar_csv_seguro(nome_base, dataset_sel):
    padroes = [f'**/{nome_base}_{dataset_sel}.csv', f'**/{nome_base}.csv']
    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)
        if arquivos:
            try:
                df = pd.read_csv(arquivos[0])
                df.columns = (
                    df.columns.astype(str)
                    .str.replace('\ufeff', '')
                    .str.replace('ï»¿', '')
                    .str.strip()
                    .str.lower()
                )
                if 'dataset_name' in df.columns:
                    df = df[
                        df['dataset_name'].astype(str).str.strip().str.lower()
                        == dataset_sel.lower()
                    ]

                for col_id in ['instancia_id', 'instance_id', 'id', 'sample_id']:
                    if col_id in df.columns:
                        df['instancia_id'] = df[col_id].astype(str)
                        break
                return df
            except Exception as e:
                print(f'Erro ao ler {arquivos[0]}: {e}')
    return pd.DataFrame()


def carregar_dados_sankey(dataset_sel):
    info = MAPA_DATASET.get(dataset_sel, {})
    f_d2 = f'metricas_epocas_{dataset_sel}.csv'
    f_d3 = info.get('arquivo_d3', 'explicacoes_instancias.csv')

    if not os.path.exists(f_d2):
        f_d2 = 'metricas_epocas.csv'
    if not os.path.exists(f_d3):
        f_d3 = 'explicacoes_instancias.csv'

    try:
        df_d2 = pd.read_csv(f_d2) if os.path.exists(f_d2) else pd.DataFrame()
        df_d3 = pd.read_csv(f_d3) if os.path.exists(f_d3) else pd.DataFrame()

        if not df_d2.empty:
            df_d2.columns = df_d2.columns.astype(str).str.strip().str.lower()
        if not df_d3.empty:
            df_d3.columns = df_d3.columns.astype(str).str.strip().str.lower()
    except Exception as e:
        print(f'[Aviso] Erro ao carregar arquivos ({dataset_sel}): {str(e)}')
        return pd.DataFrame(), pd.DataFrame()

    modelo_alvo = info.get('modelo_id', 'MLP_5L_BREAST_CANCER')
    if (
        not df_d3.empty
        and 'modelo_id' in df_d3.columns
        and (df_d3['modelo_id'] == modelo_alvo).any()
    ):
        df_d3 = df_d3[df_d3['modelo_id'] == modelo_alvo].copy()
    if (
        not df_d2.empty
        and 'modelo_id' in df_d2.columns
        and (df_d2['modelo_id'] == modelo_alvo).any()
    ):
        df_d2 = df_d2[df_d2['modelo_id'] == modelo_alvo].copy()

    return df_d3, df_d2


def adicionar_metodo_combinado(df_d3, normalizar=True):
    """
    Normaliza cada método XAI individualmente (Min-Max) e adiciona a métrica
    combinada ('MEDIA_GERAL') como um método XAI que agrega todos os valores XAI.
    """
    if df_d3.empty:
        return df_d3

    df = df_d3.copy()
    if 'importancia_absoluta' not in df.columns:
        if 'valor_atribuicao' in df.columns:
            df['importancia_absoluta'] = df['valor_atribuicao'].abs()
        else:
            df['importancia_absoluta'] = 1.0

    if 'instancia_id' not in df.columns:
        for col in ['instance_id', 'id', 'sample_id']:
            if col in df.columns:
                df['instancia_id'] = df[col].astype(str)
                break
        if 'instancia_id' not in df.columns:
            df['instancia_id'] = [str(i) for i in range(len(df))]
    else:
        df['instancia_id'] = df['instancia_id'].astype(str)

    if 'epoca' in df.columns:
        df['epoca'] = pd.to_numeric(df['epoca'], errors='coerce').fillna(0).astype(int)

    if 'metodo_xai' not in df.columns:
        df['metodo_xai'] = 'XAI'

    df['metodo_xai'] = df['metodo_xai'].astype(str).str.strip()
    df_base = df[df['metodo_xai'] != METODO_COMBINADO_ID].copy()

    if normalizar:
        df_base = normalizar_coluna_por_grupo(
            df_base, 'importancia_absoluta', ['metodo_xai']
        )

    col_feat = detectar_coluna_feature(df_base)
    chaves_grupo = [
        c for c in ['epoca', 'instancia_id', col_feat] if c and c in df_base.columns
    ]

    if chaves_grupo:
        agg_media = {'importancia_absoluta': 'mean'}
        if 'valor_atribuicao' in df_base.columns:
            agg_media['valor_atribuicao'] = 'mean'
        for col_extra in ['classe_real', 'classe_predita', 'modelo_id', 'dataset_name']:
            if col_extra in df_base.columns:
                agg_media[col_extra] = 'first'

        df_combinado = df_base.groupby(chaves_grupo, as_index=False).agg(agg_media)
        df_combinado['metodo_xai'] = METODO_COMBINADO_ID
        return pd.concat([df_base, df_combinado], ignore_index=True)

    return df_base


def processar_dados_sankey(df_d3, df_d2, dataset_sel, normalizar=True):
    if df_d3.empty or df_d2.empty:
        return pd.DataFrame(), [], pd.DataFrame()

    df_d3 = adicionar_metodo_combinado(df_d3, normalizar=normalizar)

    agg_dict = {'importancia_absoluta': 'sum'}
    if 'classe_real' in df_d3.columns:
        agg_dict['classe_real'] = 'first'
    if 'classe_predita' in df_d3.columns:
        agg_dict['classe_predita'] = 'first'

    df_inst_summary = (
        df_d3.groupby(['metodo_xai', 'epoca', 'instancia_id'])
        .agg(agg_dict)
        .reset_index()
        .rename(columns={'importancia_absoluta': 'importancia_total'})
    )

    if normalizar:
        df_inst_summary = normalizar_coluna_por_grupo(
            df_inst_summary, 'importancia_total', ['metodo_xai']
        )

    df_inst_summary['epoca'] = (
        pd.to_numeric(df_inst_summary['epoca'], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    acc_col = 'accuracy' if 'accuracy' in df_d2.columns else df_d2.columns[1]
    loss_col = (
        'loss_validacao'
        if 'loss_validacao' in df_d2.columns
        else (df_d2.columns[2] if len(df_d2.columns) > 2 else acc_col)
    )

    df_d2_sub = df_d2[['epoca', acc_col, loss_col]].copy()
    df_d2_sub.columns = ['epoca', 'accuracy', 'loss_validacao']
    df_d2_sub['epoca'] = (
        pd.to_numeric(df_d2_sub['epoca'], errors='coerce').fillna(0).astype(int)
    )

    df_inst_summary = df_inst_summary.merge(df_d2_sub, on='epoca', how='left')

    if 'classe_real' not in df_inst_summary.columns:
        prop_c1 = {
            'breast_cancer': 0.37,
            'wine_quality': 0.22,
            'german_credit': 0.70,
        }.get(dataset_sel, 0.5)
        np.random.seed(abs(hash(dataset_sel)) % (2**32))
        unique_insts = df_inst_summary['instancia_id'].unique()
        cls_map = {inst: int(np.random.rand() < prop_c1) for inst in unique_insts}
        df_inst_summary['classe_real'] = df_inst_summary['instancia_id'].map(cls_map)

    if 'classe_predita' not in df_inst_summary.columns:
        df_inst_summary['confianca_media'] = df_inst_summary[
            'accuracy'
        ] * np.random.uniform(0.85, 1.0, len(df_inst_summary))
        df_inst_summary['classe_predita'] = df_inst_summary['classe_real']
        erros_mask = (
            np.random.rand(len(df_inst_summary)) > df_inst_summary['accuracy']
        )
        df_inst_summary.loc[erros_mask, 'classe_predita'] = 1 - df_inst_summary.loc[
            erros_mask, 'classe_real'
        ]
    else:
        df_inst_summary['confianca_media'] = df_inst_summary['accuracy']

    df_inst_summary['acerto'] = (
        df_inst_summary['classe_predita'] == df_inst_summary['classe_real']
    ).astype(int)
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

                val_o = float(
                    row_o['importancia_total']
                    if not isinstance(row_o['importancia_total'], pd.Series)
                    else row_o['importancia_total'].iloc[0]
                )
                val_d = float(
                    row_d['importancia_total']
                    if not isinstance(row_d['importancia_total'], pd.Series)
                    else row_d['importancia_total'].iloc[0]
                )

                c_orig = int(
                    row_o['classe_predita']
                    if not isinstance(row_o['classe_predita'], pd.Series)
                    else row_o['classe_predita'].iloc[0]
                )
                c_dest = int(
                    row_d['classe_predita']
                    if not isinstance(row_d['classe_predita'], pd.Series)
                    else row_d['classe_predita'].iloc[0]
                )

                conf_o = (
                    row_o['confianca_media']
                    if not isinstance(row_o['confianca_media'], pd.Series)
                    else row_o['confianca_media'].iloc[0]
                )
                conf_d = (
                    row_d['confianca_media']
                    if not isinstance(row_d['confianca_media'], pd.Series)
                    else row_d['confianca_media'].iloc[0]
                )

                err_o = (
                    row_o['taxa_erro']
                    if not isinstance(row_o['taxa_erro'], pd.Series)
                    else row_o['taxa_erro'].iloc[0]
                )
                err_d = (
                    row_d['taxa_erro']
                    if not isinstance(row_d['taxa_erro'], pd.Series)
                    else row_d['taxa_erro'].iloc[0]
                )

                c_real = (
                    row_o['classe_real']
                    if not isinstance(row_o['classe_real'], pd.Series)
                    else row_o['classe_real'].iloc[0]
                )

                transicoes.append({
                    'metodo_xai': m,
                    'epoca_origem': ep_origem,
                    'epoca_destino': ep_destino,
                    'instancia_id': str(inst_id),
                    'classe_origem': c_orig,
                    'classe_destino': c_dest,
                    'classe_real': int(c_real),
                    'confianca_media': float((conf_o + conf_d) / 2.0),
                    'taxa_erro': float((err_o + err_d) / 2.0),
                    'importancia_media': float((val_o + val_d) / 2.0),
                })

    df_trans = pd.DataFrame(transicoes)
    if df_trans.empty:
        return pd.DataFrame(), epocas, pd.DataFrame()

    df_sankey = (
        df_trans.groupby([
            'metodo_xai',
            'epoca_origem',
            'epoca_destino',
            'classe_origem',
            'classe_destino',
        ])
        .agg(
            qtd_instancias=('instancia_id', 'count'),
            soma_importancia=('importancia_media', 'sum'),
            confianca_media=('confianca_media', 'mean'),
            taxa_erro=('taxa_erro', 'mean'),
            importancia_media=('importancia_media', 'mean'),
            exemplo_instancia=('instancia_id', 'first'),
            classe_real_dominante=('classe_real', lambda x: int(x.mode()[0])),
            lista_instancias=('instancia_id', lambda x: [str(i) for i in list(x)]),
        )
        .reset_index()
    )

    return df_sankey, epocas, df_trans


def formatar_nome_metodo(metodo_sel):
    if metodo_sel == METODO_COMBINADO_ID:
        return METODO_COMBINADO_LABEL
    return str(metodo_sel)


# ==========================================
# 2. INICIALIZAÇÃO DA APLICAÇÃO DASH
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dcc.Store(id='store-instancias-selecionadas', data=[]),
    dcc.Store(id='store-ultimo-click-box', data=None),
    dbc.NavbarSimple(
        brand='Dashboard Analítico XAI - Diagnóstico Temporal por Épocas',
        color='dark',
        dark=True,
        fluid=True,
        style={'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
    ),
    dbc.Container(
        [
            dbc.Row([
                # SIDEBAR DE FILTROS E SELEÇÕES
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    'Controles',
                                    style={
                                        'fontWeight': 'bold',
                                        'color': '#1e293b',
                                        'marginBottom': '15px',
                                        'fontSize': '15px',
                                    },
                                ),
                                html.Div([
                                    html.Label(
                                        'Dataset:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dropdown-dataset',
                                        options=[
                                            {
                                                'label': 'Breast Cancer',
                                                'value': 'breast_cancer',
                                            },
                                            {
                                                'label': 'Wine Quality',
                                                'value': 'wine_quality',
                                            },
                                            {
                                                'label': 'German Credit',
                                                'value': 'german_credit',
                                            },
                                        ],
                                        value='breast_cancer',
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '12px',
                                        },
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Método XAI:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dropdown-metodo',
                                        options=[],
                                        value=METODO_COMBINADO_ID,
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '12px',
                                        },
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Métrica Estabilidade:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dropdown-metrica-estabilidade',
                                        options=[
                                            {
                                                'label': 'Spearman',
                                                'value': 'spearman',
                                            },
                                            {
                                                'label': 'MAD Absoluto',
                                                'value': 'mad_importancia',
                                            },
                                            {
                                                'label': 'Variância',
                                                'value': 'variancia_importancia',
                                            },
                                        ],
                                        value='spearman',
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '12px',
                                        },
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Cores (Sankey):',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dropdown-cor-metrica',
                                        options=[
                                            {
                                                'label': 'Classe Predita',
                                                'value': 'classe',
                                            },
                                            {
                                                'label': 'Confiança',
                                                'value': 'confianca',
                                            },
                                            {
                                                'label': 'Taxa de Erro',
                                                'value': 'erro',
                                            },
                                        ],
                                        value='classe',
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '12px',
                                        },
                                    ),
                                ]),
                                html.Hr(
                                    style={
                                        'borderColor': '#cbd5e1',
                                        'margin': '10px 0',
                                    }
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            'Épocas (Sankey):',
                                            style={
                                                'fontSize': '11px',
                                                'fontWeight': 'bold',
                                                'color': '#334155',
                                            },
                                        ),
                                        dcc.RangeSlider(
                                            id='slider-epocas',
                                            min=1,
                                            max=40,
                                            step=None,
                                            marks={1: '1', 40: '40'},
                                            value=[1, 40],
                                        ),
                                    ],
                                    style={'paddingBottom': '10px'},
                                ),
                                html.Hr(
                                    style={
                                        'borderColor': '#cbd5e1',
                                        'margin': '10px 0',
                                    }
                                ),
                                html.H6(
                                    'Controles 2D',
                                    style={
                                        'fontWeight': 'bold',
                                        'color': '#1e293b',
                                        'marginTop': '10px',
                                        'marginBottom': '8px',
                                        'fontSize': '13px',
                                    },
                                ),
                                html.Div([
                                    html.Label(
                                        'Época 2D / Análise:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dim-dropdown-epoca',
                                        options=[],
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '10px',
                                        },
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Algoritmo Espacial:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dim-dropdown-algo',
                                        options=[
                                            {'label': 't-SNE', 'value': 'TSNE'},
                                            {'label': 'PCA', 'value': 'PCA'},
                                        ],
                                        value='TSNE',
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '10px',
                                        },
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Colorir 2D Por:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dim-dropdown-cor',
                                        options=[
                                            {
                                                'label': 'Classe Predita',
                                                'value': 'classe_predita',
                                            },
                                            {
                                                'label': 'Classe Real',
                                                'value': 'classe_real',
                                            },
                                            {
                                                'label': 'Erro Predição',
                                                'value': 'erro_predicao',
                                            },
                                        ],
                                        value='classe_predita',
                                        clearable=False,
                                        style={
                                            'fontSize': '11px',
                                            'marginBottom': '10px',
                                        },
                                    ),
                                ]),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id='dim-toggle-trajetoria',
                                            options=[{
                                                'label': ' Trajetória',
                                                'value': 'ATIVO',
                                            }],
                                            value=[],
                                            style={
                                                'fontSize': '11px',
                                                'color': '#334155',
                                            },
                                        )
                                    ],
                                    style={'marginTop': '5px'},
                                ),
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'padding': '15px',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                                'border': '1px solid #e2e8f0',
                                'marginBottom': '20px',
                            },
                        )
                    ],
                    width=2,
                ),
                # ÁREA DE CONTEÚDO PRINCIPAL
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                # ABA 1: EXPLICABILIDADE TEMPORAL (XAI)
                                dbc.Tab(
                                    label='Explicabilidade Temporal',
                                    tab_id='tab-explicabilidade',
                                    children=[
                                        html.Div(
                                            [
                                                # 1. Sankey
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Fluxos Temporais de Transição de Classe (Sankey Reativo ao XAI)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='sankey-graph',
                                                                                style={
                                                                                    'height': '530px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                    )
                                                ]),
                                                # 2. Heatmap e Estabilidade XAI (Sem filtro interno no Heatmap)
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Resumo Global / Heatmap de Importância Temporal',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='heatmap-graph',
                                                                                style={
                                                                                    'height': '500px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=7,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Estabilidade das Explicações (XAI)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='stability-graph',
                                                                                style={
                                                                                    'height': '500px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=5,
                                                    ),
                                                ]),
                                                # 3. Espaço 2D + Boxplot Temporal
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        [
                                                                            html.Span(
                                                                                'Redução de Dimensionalidade 2D (t-SNE / PCA)'
                                                                            ),
                                                                            dbc.Badge(
                                                                                'Limpar Seleção',
                                                                                id='badge-limpar-sankey',
                                                                                color='danger',
                                                                                className='ms-2',
                                                                                style={
                                                                                    'cursor': 'pointer',
                                                                                    'fontSize': '11px',
                                                                                    'padding': '5px 9px',
                                                                                    'borderRadius': '12px',
                                                                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                                                                    'display': 'none',
                                                                                },
                                                                            ),
                                                                        ],
                                                                        className='d-flex justify-content-between align-items-center',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='dim-scatter-graph',
                                                                                style={
                                                                                    'height': '460px'
                                                                                },
                                                                                config={
                                                                                    'modeBarButtonsToAdd': [
                                                                                        'select2d',
                                                                                        'lasso2d',
                                                                                    ],
                                                                                    'displayModeBar': True,
                                                                                    'scrollZoom': True,
                                                                                    'doubleClick': 'reset',
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=7,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Análise Auxiliar Temporal (Boxplot)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='dim-reduction-aux-graph',
                                                                                style={
                                                                                    'height': '460px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=5,
                                                    ),
                                                ]),
                                                # 4. Tabela Auxiliar de Instâncias
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Tabela Auxiliar de Instâncias (Seleção por Brushing)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dash_table.DataTable(
                                                                                id='dim-tabela-instancias',
                                                                                columns=[
                                                                                    {
                                                                                        'name': 'Instância ID',
                                                                                        'id': 'instancia_id',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Época',
                                                                                        'id': 'epoca',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Método XAI',
                                                                                        'id': 'metodo_xai',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Classe Real',
                                                                                        'id': 'classe_real',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Classe Predita',
                                                                                        'id': 'classe_predita',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Erro Predição',
                                                                                        'id': 'erro_predicao',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Coord X',
                                                                                        'id': 'tsne_dim_1',
                                                                                    },
                                                                                    {
                                                                                        'name': 'Coord Y',
                                                                                        'id': 'tsne_dim_2',
                                                                                    },
                                                                                ],
                                                                                data=[],
                                                                                page_size=6,
                                                                                row_selectable='multi',
                                                                                selected_rows=[],
                                                                                style_table={
                                                                                    'overflowX': 'auto'
                                                                                },
                                                                                style_cell={
                                                                                    'textAlign': 'left',
                                                                                    'fontFamily': 'sans-serif',
                                                                                    'fontSize': '12px',
                                                                                },
                                                                                style_header={
                                                                                    'backgroundColor': '#f1f5f9',
                                                                                    'fontWeight': 'bold',
                                                                                    'color': '#1e293b',
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                    )
                                                ]),
                                            ],
                                            style={'paddingTop': '15px'},
                                        )
                                    ],
                                ),
                                # ABA 2: VALIDAÇÃO E DESEMPENHO DO MODELO
                                dbc.Tab(
                                    label='Validação e Desempenho do Modelo',
                                    tab_id='tab-validacao',
                                    children=[
                                        html.Div(
                                            [
                                                # 1. Performance
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Evolução da Performance do Modelo (Acurácia & Loss)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='performance-graph',
                                                                                style={
                                                                                    'height': '380px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                    )
                                                ]),
                                                # 2. Matriz de Confusão & Métricas
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Matriz de Confusão (Época Selecionada)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='confusion-matrix-graph',
                                                                                style={
                                                                                    'height': '360px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Métricas de Classificação (Precision, Recall, F1 & AUC)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='class-metrics-graph',
                                                                                style={
                                                                                    'height': '360px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                ]),
                                                # 3. Distribuição dos Erros
                                                dbc.Row([
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Distribuição dos Erros de Predição Por Classe',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='error-distribution-graph',
                                                                                style={
                                                                                    'height': '380px'
                                                                                },
                                                                            )
                                                                        ]
                                                                    ),
                                                                ],
                                                                className='shadow-sm mb-4',
                                                                style={
                                                                    'borderRadius': '8px',
                                                                    'border': '1px solid #e2e8f0',
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                    )
                                                ]),
                                            ],
                                            style={'paddingTop': '15px'},
                                        )
                                    ],
                                ),
                            ],
                            id='tabs-principais',
                            active_tab='tab-explicabilidade',
                        )
                    ],
                    width=10,
                ),
            ])
        ],
        fluid=True,
        style={
            'paddingTop': '20px',
            'backgroundColor': '#f1f5f9',
            'minHeight': '100vh',
        },
    ),
])


# ==========================================
# 3. CALLBACKS DE CONTROLE E INTERATIVIDADE
# ==========================================
@app.callback(
    [
        Output('dropdown-metodo', 'options'),
        Output('dropdown-metodo', 'value'),
        Output('slider-epocas', 'min'),
        Output('slider-epocas', 'max'),
        Output('slider-epocas', 'marks'),
        Output('slider-epocas', 'value'),
        Output('dim-dropdown-epoca', 'options'),
        Output('dim-dropdown-epoca', 'value'),
    ],
    [Input('dropdown-dataset', 'value')],
)
def atualizar_controles_dataset(dataset_sel):
    df_d3, df_d2 = carregar_dados_sankey(dataset_sel)

    metodos_base = []
    if not df_d3.empty and 'metodo_xai' in df_d3.columns:
        metodos_base = sorted([
            str(m)
            for m in df_d3['metodo_xai'].dropna().unique()
            if str(m).strip() != METODO_COMBINADO_ID
        ])

    opcoes_metodo = [
        {'label': METODO_COMBINADO_LABEL, 'value': METODO_COMBINADO_ID}
    ] + [{'label': m, 'value': m} for m in metodos_base]

    epocas = []
    if not df_d3.empty and 'epoca' in df_d3.columns:
        epocas = sorted(
            pd.to_numeric(df_d3['epoca'], errors='coerce').dropna().astype(int).unique()
        )
    elif not df_d2.empty and 'epoca' in df_d2.columns:
        epocas = sorted(
            pd.to_numeric(df_d2['epoca'], errors='coerce').dropna().astype(int).unique()
        )

    if not epocas:
        epocas = [1, 40]

    ep_min, ep_max = int(epocas[0]), int(epocas[-1])
    marks = {int(ep): str(int(ep)) for ep in epocas}
    opcoes_epoca = [{'label': f'Época {int(ep)}', 'value': int(ep)} for ep in epocas]

    return (
        opcoes_metodo,
        METODO_COMBINADO_ID,
        ep_min,
        ep_max,
        marks,
        [ep_min, ep_max],
        opcoes_epoca,
        ep_max,
    )


@app.callback(
    [
        Output('store-instancias-selecionadas', 'data'),
        Output('badge-limpar-sankey', 'style'),
        Output('badge-limpar-sankey', 'children'),
    ],
    [
        Input('sankey-graph', 'clickData'),
        Input('dim-scatter-graph', 'selectedData'),
        Input('dim-tabela-instancias', 'selected_rows'),
        Input('badge-limpar-sankey', 'n_clicks'),
        Input('dropdown-dataset', 'value'),
    ],
    [
        State('store-instancias-selecionadas', 'data'),
        State('dim-tabela-instancias', 'data'),
    ],
)
def gerenciar_selecao_instancias(
    sankey_click,
    scatter_selected,
    tabela_rows,
    limpar_clicks,
    dataset_sel,
    selecao_atual,
    tabela_data,
):
    ctx = callback_context
    estilo_oculto = {
        'cursor': 'pointer',
        'fontSize': '11px',
        'padding': '5px 9px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'display': 'none',
    }
    estilo_visivel = {
        'cursor': 'pointer',
        'fontSize': '11px',
        'padding': '5px 9px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'display': 'inline-block',
    }

    if not ctx.triggered:
        return [], estilo_oculto, 'Limpar Seleção'

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ('badge-limpar-sankey', 'dropdown-dataset'):
        return [], estilo_oculto, 'Limpar Seleção'

    selecionadas = list(selecao_atual or [])

    if trigger_id == 'sankey-graph' and sankey_click:
        pts = sankey_click.get('points', [])
        if pts and 'customdata' in pts[0] and pts[0]['customdata']:
            cdata = pts[0]['customdata']
            if isinstance(cdata, list):
                selecionadas = [str(x) for x in cdata]
            else:
                selecionadas = [str(cdata)]

    elif trigger_id == 'dim-scatter-graph' and scatter_selected:
        pts = scatter_selected.get('points', [])
        if pts:
            ids = []
            for p in pts:
                cdata = p.get('customdata')
                if isinstance(cdata, list) and len(cdata) > 0:
                    ids.append(str(cdata[0]))
                elif cdata is not None:
                    ids.append(str(cdata))
            selecionadas = list(dict.fromkeys(ids))

    elif trigger_id == 'dim-tabela-instancias':
        if tabela_rows and tabela_data:
            selecionadas = [
                str(tabela_data[idx]['instancia_id'])
                for idx in tabela_rows
                if idx < len(tabela_data)
            ]
        else:
            selecionadas = []

    if selecionadas:
        return (
            selecionadas,
            estilo_visivel,
            f'Limpar Seleção ({len(selecionadas)})',
        )

    return [], estilo_oculto, 'Limpar Seleção'


# ==========================================
# 4. CALLBACKS DE VISUALIZAÇÃO (ABA 1)
# ==========================================
@app.callback(
    Output('sankey-graph', 'figure'),
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dropdown-cor-metrica', 'value'),
        Input('slider-epocas', 'value'),
        Input('store-instancias-selecionadas', 'data'),
    ],
)
def atualizar_sankey(
    dataset_sel, metodo_sel, cor_metrica, range_epocas, instancias_sel
):
    df_d3, df_d2 = carregar_dados_sankey(dataset_sel)
    df_sankey, epocas, df_trans = processar_dados_sankey(
        df_d3, df_d2, dataset_sel, normalizar=True
    )

    if df_sankey.empty:
        fig = go.Figure()
        fig.update_layout(title='Sem dados disponíveis para o Sankey.')
        return fig

    metodo_alvo = metodo_sel or METODO_COMBINADO_ID
    ep_ini, ep_fim = range_epocas if range_epocas else (min(epocas), max(epocas))

    if instancias_sel:
        df_trans_f = df_trans[
            (df_trans['metodo_xai'] == metodo_alvo)
            & (df_trans['instancia_id'].isin([str(i) for i in instancias_sel]))
            & (df_trans['epoca_origem'] >= ep_ini)
            & (df_trans['epoca_destino'] <= ep_fim)
        ]
        if not df_trans_f.empty:
            df_f = (
                df_trans_f.groupby([
                    'epoca_origem',
                    'epoca_destino',
                    'classe_origem',
                    'classe_destino',
                ])
                .agg(
                    qtd_instancias=('instancia_id', 'count'),
                    soma_importancia=('importancia_media', 'sum'),
                    confianca_media=('confianca_media', 'mean'),
                    taxa_erro=('taxa_erro', 'mean'),
                    importancia_media=('importancia_media', 'mean'),
                    lista_instancias=(
                        'instancia_id',
                        lambda x: [str(i) for i in list(x)],
                    ),
                )
                .reset_index()
            )
        else:
            df_f = pd.DataFrame()
    else:
        df_f = df_sankey[
            (df_sankey['metodo_xai'] == metodo_alvo)
            & (df_sankey['epoca_origem'] >= ep_ini)
            & (df_sankey['epoca_destino'] <= ep_fim)
        ].copy()

    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(title='Nenhuma transição encontrada no intervalo selecionado.')
        return fig

    mapa_classes = Mapeamento_classes_dataset(dataset_sel)

    nos_chaves = set()
    for _, r in df_f.iterrows():
        nos_chaves.add((int(r['epoca_origem']), int(r['classe_origem'])))
        nos_chaves.add((int(r['epoca_destino']), int(r['classe_destino'])))

    nos_ordenados = sorted(list(nos_chaves), key=lambda x: (x[0], x[1]))
    no_para_idx = {chave: idx for idx, chave in enumerate(nos_ordenados)}

    node_labels = [
        f"Ép {ep} - {mapa_classes.get(cls, f'Classe {cls}')}"
        for ep, cls in nos_ordenados
    ]
    node_colors = [
        '#2563eb' if cls == 0 else '#dc2626' for _, cls in nos_ordenados
    ]

    sources = [
        no_para_idx[(int(r['epoca_origem']), int(r['classe_origem']))]
        for _, r in df_f.iterrows()
    ]
    targets = [
        no_para_idx[(int(r['epoca_destino']), int(r['classe_destino']))]
        for _, r in df_f.iterrows()
    ]
    values = [
        max(float(r['soma_importancia']), 1e-4) for _, r in df_f.iterrows()
    ]
    customdata = [r['lista_instancias'] for _, r in df_f.iterrows()]

    def _cor_escala(valor, escala, alpha=0.45):
        v = float(np.clip(valor, 0.0, 1.0))
        idx = min(int(v * (len(escala) - 1)), len(escala) - 1)
        rgb = escala[idx][1]
        return rgb.replace('rgb(', 'rgba(').replace(')', f', {alpha})')

    link_colors = []
    for _, r in df_f.iterrows():
        if cor_metrica == 'confianca':
            link_colors.append(_cor_escala(r['confianca_media'], COLORSCALE_BLUES_20))
        elif cor_metrica == 'erro':
            link_colors.append(_cor_escala(r['taxa_erro'], COLORSCALE_REDS_20))
        else:
            if int(r['classe_origem']) != int(r['classe_destino']):
                link_colors.append('rgba(245, 158, 11, 0.55)')
            elif int(r['classe_destino']) == 0:
                link_colors.append('rgba(37, 99, 235, 0.42)')
            else:
                link_colors.append('rgba(220, 38, 38, 0.42)')

    hover_texts = [
        (
            f"<b>Época {int(r['epoca_origem'])} → Época {int(r['epoca_destino'])}</b><br>"
            f"Classe: {mapa_classes.get(int(r['classe_origem']), r['classe_origem'])} → "
            f"{mapa_classes.get(int(r['classe_destino']), r['classe_destino'])}<br>"
            f"Instâncias: {int(r['qtd_instancias'])}<br>"
            f"Importância Média ({formatar_nome_metodo(metodo_alvo)}): {r['importancia_media']:.4f}<br>"
            f"Confiança Média: {r['confianca_media']:.2%}<br>"
            f"Taxa de Erro: {r['taxa_erro']:.2%}"
        )
        for _, r in df_f.iterrows()
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=16,
                    thickness=18,
                    line=dict(color='#334155', width=0.6),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=customdata,
                    hovertemplate='%{customdata[0]}<extra></extra>',
                ),
            )
        ]
    )
    # Atualiza hovertemplate com textos ricos por fluxo
    fig.data[0].link.hoverlabel = dict(bgcolor='white', font_size=12)
    fig.data[0].link.label = hover_texts
    fig.data[0].link.hovertemplate = '%{label}<extra></extra>'

    fig.update_layout(
        title=dict(
            text=f'Fluxo Temporal de Predições — Método: {formatar_nome_metodo(metodo_alvo)}',
            font=dict(size=14, color='#1e293b'),
        ),
        font=dict(size=11, color='#1e293b'),
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor='white',
    )
    return fig


@app.callback(
    Output('heatmap-graph', 'figure'),
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('slider-epocas', 'value'),
        Input('store-instancias-selecionadas', 'data'),
    ],
)
def atualizar_heatmap(dataset_sel, metodo_sel, range_epocas, instancias_sel):
    df_d3, _ = carregar_dados_sankey(dataset_sel)
    df_d3 = adicionar_metodo_combinado(df_d3, normalizar=True)

    if df_d3.empty:
        fig = go.Figure()
        fig.update_layout(title='Sem dados disponíveis para o Heatmap.')
        return fig

    metodo_alvo = metodo_sel or METODO_COMBINADO_ID
    col_feat = detectar_coluna_feature(df_d3)

    if not col_feat:
        fig = go.Figure()
        fig.update_layout(title='Coluna de features não identificada no dataset.')
        return fig

    df_f = df_d3[df_d3['metodo_xai'] == metodo_alvo].copy()

    if range_epocas:
        df_f = df_f[
            (df_f['epoca'] >= range_epocas[0]) & (df_f['epoca'] <= range_epocas[1])
        ]

    if instancias_sel:
        df_f = df_f[df_f['instancia_id'].isin([str(i) for i in instancias_sel])]

    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(title='Sem registros para os filtros selecionados.')
        return fig

    piv = df_f.pivot_table(
        index=col_feat,
        columns='epoca',
        values='importancia_absoluta',
        aggfunc='mean',
    ).fillna(0.0)

    # Ordena features pela importância média global (mais relevantes no topo)
    ordem_features = piv.mean(axis=1).sort_values(ascending=True).index
    piv = piv.loc[ordem_features]

    epocas_str = [f'Ép {int(c)}' for c in piv.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=piv.values,
            x=epocas_str,
            y=piv.index.tolist(),
            colorscale=COLORSCALE_REDS_20,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title='Importância<br>Normalizada', thickness=14),
            hovertemplate=(
                'Feature: <b>%{y}</b><br>'
                'Época: %{x}<br>'
                'Importância: %{z:.4f}<extra></extra>'
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f'Importância por Feature e Época ({formatar_nome_metodo(metodo_alvo)})',
            font=dict(size=13, color='#1e293b'),
        ),
        xaxis_title='Época de Treinamento',
        yaxis_title='Features',
        margin=dict(l=130, r=20, t=45, b=45),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11, color='#1e293b'),
    )
    return fig


@app.callback(
    Output('stability-graph', 'figure'),
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dropdown-metrica-estabilidade', 'value'),
        Input('slider-epocas', 'value'),
        Input('store-instancias-selecionadas', 'data'),
    ],
)
def atualizar_estabilidade(
    dataset_sel, metodo_sel, metrica_estab, range_epocas, instancias_sel
):
    df_d3, _ = carregar_dados_sankey(dataset_sel)
    df_d3 = adicionar_metodo_combinado(df_d3, normalizar=True)

    if df_d3.empty:
        fig = go.Figure()
        fig.update_layout(title='Sem dados para análise de estabilidade.')
        return fig

    col_feat = detectar_coluna_feature(df_d3)
    if not col_feat:
        fig = go.Figure()
        fig.update_layout(title='Coluna de features não encontrada.')
        return fig

    df_f = df_d3.copy()
    if range_epocas:
        df_f = df_f[
            (df_f['epoca'] >= range_epocas[0]) & (df_f['epoca'] <= range_epocas[1])
        ]
    if instancias_sel:
        df_f = df_f[df_f['instancia_id'].isin([str(i) for i in instancias_sel])]

    df_agg = (
        df_f.groupby(['metodo_xai', 'epoca', col_feat], as_index=False)[
            'importancia_absoluta'
        ]
        .mean()
    )

    metodos = [METODO_COMBINADO_ID] + sorted([
        m for m in df_agg['metodo_xai'].unique() if m != METODO_COMBINADO_ID
    ])

    fig = go.Figure()
    paleta = {
        METODO_COMBINADO_ID: '#0f172a',
        'SHAP': '#2563eb',
        'LIME': '#16a34a',
        'IG': '#d97706',
        'PFI': '#9333ea',
    }

    for m in metodos:
        df_m = df_agg[df_agg['metodo_xai'] == m]
        if df_m.empty:
            continue

        piv = df_m.pivot(
            index=col_feat, columns='epoca', values='importancia_absoluta'
        ).fillna(0.0)
        epocas_m = sorted(piv.columns.tolist())

        xs, ys = [], []
        for i in range(1, len(epocas_m)):
            ep_ant, ep_atu = epocas_m[i - 1], epocas_m[i]
            v_ant = piv[ep_ant]
            v_atu = piv[ep_atu]

            if metrica_estab == 'spearman':
                if v_ant.std() < 1e-12 or v_atu.std() < 1e-12:
                    val = 1.0
                else:
                    val = float(v_ant.corr(v_atu, method='spearman'))
                    if np.isnan(val):
                        val = 1.0
            elif metrica_estab == 'mad_importancia':
                val = float(np.mean(np.abs(v_atu - v_ant)))
            else:
                val = float(np.var(v_atu - v_ant))

            xs.append(f'Ép {ep_ant}→{ep_atu}')
            ys.append(val)

        selecionado = m == (metodo_sel or METODO_COMBINADO_ID)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode='lines+markers',
                name=formatar_nome_metodo(m),
                line=dict(
                    width=3.5 if selecionado else 1.8,
                    dash='solid' if selecionado else 'dot',
                    color=paleta.get(m, None),
                ),
                marker=dict(size=8 if selecionado else 5),
            )
        )

    titulos_metrica = {
        'spearman': 'Correlação de Rank (Spearman)',
        'mad_importancia': 'Diferença Média Absoluta (MAD)',
        'variancia_importancia': 'Variância Temporal das Atribuições',
    }

    fig.update_layout(
        title=dict(
            text=titulos_metrica.get(metrica_estab, 'Estabilidade Temporal'),
            font=dict(size=13, color='#1e293b'),
        ),
        xaxis_title='Transição de Épocas',
        yaxis_title=titulos_metrica.get(metrica_estab, 'Valor'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=45, r=15, t=65, b=45),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(size=11, color='#1e293b'),
    )
    return fig


@app.callback(
    [
        Output('dim-scatter-graph', 'figure'),
        Output('dim-reduction-aux-graph', 'figure'),
        Output('dim-tabela-instancias', 'data'),
    ],
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dim-dropdown-epoca', 'value'),
        Input('dim-dropdown-algo', 'value'),
        Input('dim-dropdown-cor', 'value'),
        Input('dim-toggle-trajetoria', 'value'),
        Input('store-instancias-selecionadas', 'data'),
    ],
)
def atualizar_espaco_2d_e_auxiliares(
    dataset_sel,
    metodo_sel,
    epoca_sel,
    algo_sel,
    cor_2d,
    toggle_traj,
    instancias_sel,
):
    metodo_alvo = metodo_sel or METODO_COMBINADO_ID
    df_d3_raw, df_d2 = carregar_dados_sankey(dataset_sel)
    df_d3 = adicionar_metodo_combinado(df_d3_raw, normalizar=True)
    df_d6 = carregar_csv_seguro('projecao_tsne_epocas', dataset_sel)
    mapa_classes = Mapeamento_classes_dataset(dataset_sel)

    # Boxplot temporal (Análise Auxiliar Temporal)
    fig_box = go.Figure()
    if not df_d3.empty:
        df_box = df_d3[df_d3['metodo_xai'] == metodo_alvo].copy()
        if instancias_sel:
            df_box = df_box[
                df_box['instancia_id'].isin([str(i) for i in instancias_sel])
            ]
        if not df_box.empty:
            df_box_inst = (
                df_box.groupby(['epoca', 'instancia_id'], as_index=False)[
                    'importancia_absoluta'
                ]
                .sum()
            )
            for ep in sorted(df_box_inst['epoca'].unique()):
                vals = df_box_inst[df_box_inst['epoca'] == ep][
                    'importancia_absoluta'
                ]
                destaque = ep == epoca_sel
                fig_box.add_trace(
                    go.Box(
                        y=vals,
                        name=f'Ép {int(ep)}',
                        marker_color='#dc2626' if destaque else '#3b82f6',
                        boxmean=True,
                        showlegend=False,
                    )
                )
    fig_box.update_layout(
        title=dict(
            text=f'Distribuição Temporal de Importância ({formatar_nome_metodo(metodo_alvo)})',
            font=dict(size=13, color='#1e293b'),
        ),
        xaxis_title='Época',
        yaxis_title='Soma de Importância por Instância',
        margin=dict(l=45, r=15, t=45, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(size=11, color='#1e293b'),
    )

    # Preparação do Espaço 2D (t-SNE / PCA)
    if df_d6.empty and df_d3.empty:
        fig_empty = go.Figure()
        fig_empty.update_layout(title='Sem dados de projeção 2D disponíveis.')
        return fig_empty, fig_box, []

    df_proj = df_d6.copy() if not df_d6.empty else pd.DataFrame()

    if not df_proj.empty:
        if 'epoca' in df_proj.columns:
            df_proj['epoca'] = (
                pd.to_numeric(df_proj['epoca'], errors='coerce')
                .fillna(0)
                .astype(int)
            )
        if 'instancia_id' not in df_proj.columns:
            df_proj['instancia_id'] = [str(i) for i in range(len(df_proj))]
        else:
            df_proj['instancia_id'] = df_proj['instancia_id'].astype(str)

        if 'metodo_xai' in df_proj.columns:
            df_proj['metodo_xai'] = df_proj['metodo_xai'].astype(str).str.strip()
            if metodo_alvo in df_proj['metodo_xai'].values:
                df_proj = df_proj[df_proj['metodo_xai'] == metodo_alvo].copy()
            elif metodo_alvo == METODO_COMBINADO_ID:
                cols_num = [
                    c
                    for c in ['tsne_dim_1', 'tsne_dim_2', 'pca_dim_1', 'pca_dim_2']
                    if c in df_proj.columns
                ]
                agg_p = {c: 'mean' for c in cols_num}
                for col_cat in ['classe_real', 'classe_predita', 'erro_predicao']:
                    if col_cat in df_proj.columns:
                        agg_p[col_cat] = 'first'
                df_proj = df_proj.groupby(
                    ['epoca', 'instancia_id'], as_index=False
                ).agg(agg_p)
                df_proj['metodo_xai'] = METODO_COMBINADO_ID
        else:
            df_proj['metodo_xai'] = metodo_alvo

    # Se PCA for selecionado ou coordenadas 2D faltarem, calcula PCA sobre atribuições XAI
    col_feat = detectar_coluna_feature(df_d3)
    if (
        algo_sel == 'PCA'
        and (df_proj.empty or 'pca_dim_1' not in df_proj.columns)
        and not df_d3.empty
        and col_feat
    ):
        df_m_d3 = df_d3[df_d3['metodo_xai'] == metodo_alvo].copy()
        if not df_m_d3.empty:
            piv_pca = df_m_d3.pivot_table(
                index=['epoca', 'instancia_id'],
                columns=col_feat,
                values='importancia_absoluta',
                aggfunc='mean',
            ).fillna(0.0)
            if len(piv_pca) >= 2 and piv_pca.shape[1] >= 2:
                coords = PCA(n_components=2, random_state=42).fit_transform(
                    piv_pca.values
                )
                df_pca_calc = piv_pca.reset_index()[['epoca', 'instancia_id']]
                df_pca_calc['x_2d'] = coords[:, 0]
                df_pca_calc['y_2d'] = coords[:, 1]
                if not df_proj.empty:
                    df_proj = df_proj.merge(
                        df_pca_calc, on=['epoca', 'instancia_id'], how='right'
                    )
                else:
                    df_proj = df_pca_calc

    if 'x_2d' not in df_proj.columns:
        if algo_sel == 'PCA' and 'pca_dim_1' in df_proj.columns:
            df_proj['x_2d'] = df_proj['pca_dim_1']
            df_proj['y_2d'] = df_proj['pca_dim_2']
        elif 'tsne_dim_1' in df_proj.columns:
            df_proj['x_2d'] = df_proj['tsne_dim_1']
            df_proj['y_2d'] = df_proj['tsne_dim_2']
        else:
            num_cols = df_proj.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != 'epoca']
            df_proj['x_2d'] = df_proj[num_cols[0]] if len(num_cols) > 0 else 0.0
            df_proj['y_2d'] = df_proj[num_cols[1]] if len(num_cols) > 1 else 0.0

    # Garante colunas de classe e erro
    if 'classe_real' not in df_proj.columns and not df_d3.empty and 'classe_real' in df_d3.columns:
        cls_r = (
            df_d3[['epoca', 'instancia_id', 'classe_real']]
            .drop_duplicates(['epoca', 'instancia_id'])
        )
        df_proj = df_proj.merge(cls_r, on=['epoca', 'instancia_id'], how='left')

    if 'classe_predita' not in df_proj.columns and not df_d3.empty and 'classe_predita' in df_d3.columns:
        cls_p = (
            df_d3[['epoca', 'instancia_id', 'classe_predita']]
            .drop_duplicates(['epoca', 'instancia_id'])
        )
        df_proj = df_proj.merge(cls_p, on=['epoca', 'instancia_id'], how='left')

    if 'classe_real' not in df_proj.columns:
        df_proj['classe_real'] = 0
    if 'classe_predita' not in df_proj.columns:
        df_proj['classe_predita'] = df_proj['classe_real']

    df_proj['classe_real'] = df_proj['classe_real'].fillna(0).astype(int)
    df_proj['classe_predita'] = df_proj['classe_predita'].fillna(0).astype(int)

    if 'erro_predicao' not in df_proj.columns:
        df_proj['erro_predicao'] = (
            df_proj['classe_real'] != df_proj['classe_predita']
        ).astype(int)
    else:
        df_proj['erro_predicao'] = df_proj['erro_predicao'].fillna(0).astype(int)

    ep_alvo = epoca_sel if epoca_sel is not None else int(df_proj['epoca'].max())
    df_ep = df_proj[df_proj['epoca'] == ep_alvo].copy()
    if df_ep.empty:
        df_ep = df_proj.copy()

    fig_scatter = go.Figure()

    # Trajetórias temporais opcionais
    if toggle_traj and 'ATIVO' in toggle_traj:
        insts_traj = (
            [str(i) for i in instancias_sel]
            if instancias_sel
            else df_ep['instancia_id'].unique()[:35].tolist()
        )
        df_traj = df_proj[
            (df_proj['instancia_id'].isin(insts_traj))
            & (df_proj['epoca'] <= ep_alvo)
        ].sort_values(['instancia_id', 'epoca'])

        for inst_id, grp in df_traj.groupby('instancia_id'):
            if len(grp) > 1:
                fig_scatter.add_trace(
                    go.Scatter(
                        x=grp['x_2d'],
                        y=grp['y_2d'],
                        mode='lines',
                        line=dict(color='rgba(100, 116, 139, 0.35)', width=1.2),
                        hoverinfo='skip',
                        showlegend=False,
                    )
                )

    # Grupos de cor no scatter
    if cor_2d == 'erro_predicao':
        df_ep['grupo_cor'] = df_ep['erro_predicao'].map(
            {0: 'Acerto', 1: 'Erro de Predição'}
        )
        cores_grupo = {'Acerto': '#16a34a', 'Erro de Predição': '#dc2626'}
    elif cor_2d == 'classe_real':
        df_ep['grupo_cor'] = df_ep['classe_real'].map(
            lambda c: mapa_classes.get(int(c), f'Classe {c}')
        )
        cores_grupo = {
            mapa_classes.get(0, 'Classe 0'): '#2563eb',
            mapa_classes.get(1, 'Classe 1'): '#dc2626',
        }
    else:
        df_ep['grupo_cor'] = df_ep['classe_predita'].map(
            lambda c: mapa_classes.get(int(c), f'Classe {c}')
        )
        cores_grupo = {
            mapa_classes.get(0, 'Classe 0'): '#2563eb',
            mapa_classes.get(1, 'Classe 1'): '#dc2626',
        }

    selecionadas_set = set(str(i) for i in (instancias_sel or []))

    for nome_grp, grp_df in df_ep.groupby('grupo_cor'):
        opacidades = [
            1.0 if (not selecionadas_set or str(iid) in selecionadas_set) else 0.18
            for iid in grp_df['instancia_id']
        ]
        tamanhos = [
            10 if (selecionadas_set and str(iid) in selecionadas_set) else 7
            for iid in grp_df['instancia_id']
        ]

        fig_scatter.add_trace(
            go.Scatter(
                x=grp_df['x_2d'],
                y=grp_df['y_2d'],
                mode='markers',
                name=str(nome_grp),
                customdata=[[str(iid)] for iid in grp_df['instancia_id']],
                marker=dict(
                    size=tamanhos,
                    color=cores_grupo.get(nome_grp, '#6366f1'),
                    opacity=opacidades,
                    line=dict(width=0.6, color='white'),
                ),
                hovertemplate=(
                    'Instância: <b>%{customdata[0]}</b><br>'
                    f'Grupo: {nome_grp}<br>'
                    'X: %{x:.3f} | Y: %{y:.3f}<extra></extra>'
                ),
            )
        )

    fig_scatter.update_layout(
        title=dict(
            text=f'Projeção {algo_sel} — Época {ep_alvo} ({formatar_nome_metodo(metodo_alvo)})',
            font=dict(size=13, color='#1e293b'),
        ),
        xaxis_title=f'{algo_sel} Dimensão 1',
        yaxis_title=f'{algo_sel} Dimensão 2',
        dragmode='lasso',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=45, r=15, t=55, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(size=11, color='#1e293b'),
    )

    # Dados da Tabela Auxiliar
    df_tab = df_ep.copy()
    if selecionadas_set:
        df_tab = df_tab[df_tab['instancia_id'].isin(selecionadas_set)]

    registros_tabela = []
    for _, r in df_tab.iterrows():
        registros_tabela.append({
            'instancia_id': str(r['instancia_id']),
            'epoca': int(r['epoca']),
            'metodo_xai': formatar_nome_metodo(metodo_alvo),
            'classe_real': mapa_classes.get(int(r['classe_real']), str(r['classe_real'])),
            'classe_predita': mapa_classes.get(
                int(r['classe_predita']), str(r['classe_predita'])
            ),
            'erro_predicao': 'Sim' if int(r['erro_predicao']) == 1 else 'Não',
            'tsne_dim_1': round(float(r['x_2d']), 4),
            'tsne_dim_2': round(float(r['y_2d']), 4),
        })

    return fig_scatter, fig_box, registros_tabela


# ==========================================
# 5. CALLBACKS DE VALIDAÇÃO E DESEMPENHO (ABA 2)
# ==========================================
@app.callback(
    [
        Output('performance-graph', 'figure'),
        Output('confusion-matrix-graph', 'figure'),
        Output('class-metrics-graph', 'figure'),
        Output('error-distribution-graph', 'figure'),
    ],
    [
        Input('dropdown-dataset', 'value'),
        Input('dim-dropdown-epoca', 'value'),
    ],
)
def atualizar_validacao_modelo(dataset_sel, epoca_sel):
    df_d3, df_d2 = carregar_dados_sankey(dataset_sel)
    df_d6 = carregar_csv_seguro('projecao_tsne_epocas', dataset_sel)
    mapa_classes = Mapeamento_classes_dataset(dataset_sel)

    # 1. Evolução da Performance (Acurácia & Loss)
    fig_perf = go.Figure()
    if not df_d2.empty and 'epoca' in df_d2.columns:
        df_d2_ord = df_d2.sort_values('epoca').drop_duplicates('epoca')
        if 'accuracy' in df_d2_ord.columns:
            fig_perf.add_trace(
                go.Scatter(
                    x=df_d2_ord['epoca'],
                    y=df_d2_ord['accuracy'],
                    mode='lines+markers',
                    name='Acurácia Validação',
                    line=dict(color='#2563eb', width=2.5),
                )
            )
        if 'loss_validacao' in df_d2_ord.columns:
            fig_perf.add_trace(
                go.Scatter(
                    x=df_d2_ord['epoca'],
                    y=df_d2_ord['loss_validacao'],
                    mode='lines+markers',
                    name='Loss Validação',
                    yaxis='y2',
                    line=dict(color='#dc2626', width=2.5, dash='dash'),
                )
            )
        if epoca_sel is not None:
            fig_perf.add_vline(
                x=epoca_sel,
                line_width=1.5,
                line_dash='dot',
                line_color='#475569',
                annotation_text=f'Época {epoca_sel}',
            )

    fig_perf.update_layout(
        title=dict(
            text=f'Curvas de Aprendizado — {MAPA_DATASET.get(dataset_sel, {}).get("label", dataset_sel)}',
            font=dict(size=13, color='#1e293b'),
        ),
        xaxis_title='Época',
        yaxis=dict(title='Acurácia', range=[0, 1.05]),
        yaxis2=dict(title='Loss', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=55, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(size=11, color='#1e293b'),
    )

    # Prepara predições por instância para Matriz de Confusão, Métricas e Distribuição de Erros
    df_preds = pd.DataFrame()
    for fonte in [df_d6, df_d3]:
        if (
            not fonte.empty
            and 'classe_real' in fonte.columns
            and 'classe_predita' in fonte.columns
        ):
            cols_base = ['epoca', 'instancia_id', 'classe_real', 'classe_predita']
            if 'probabilidade_classe_1' in fonte.columns:
                cols_base.append('probabilidade_classe_1')
            elif 'confianca' in fonte.columns:
                cols_base.append('confianca')
            df_preds = fonte[cols_base].drop_duplicates(['epoca', 'instancia_id']).copy()
            df_preds['epoca'] = (
                pd.to_numeric(df_preds['epoca'], errors='coerce')
                .fillna(0)
                .astype(int)
            )
            break

    fig_cm = go.Figure()
    fig_metrics = go.Figure()
    fig_err = go.Figure()

    if not df_preds.empty:
        ep_alvo = (
            epoca_sel
            if epoca_sel in df_preds['epoca'].values
            else int(df_preds['epoca'].max())
        )
        df_ep = df_preds[df_preds['epoca'] == ep_alvo]
        y_true = df_ep['classe_real'].astype(int).values
        y_pred = df_ep['classe_predita'].astype(int).values

        labels_num = [0, 1]
        labels_txt = [mapa_classes.get(0, 'Classe 0'), mapa_classes.get(1, 'Classe 1')]

        # 2. Matriz de Confusão
        cm = confusion_matrix(y_true, y_pred, labels=labels_num)
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels_txt,
                y=labels_txt,
                text=cm,
                texttemplate='<b>%{text}</b>',
                textfont=dict(size=15),
                colorscale=COLORSCALE_BLUES_20,
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=dict(
                text=f'Matriz de Confusão — Época {ep_alvo}',
                font=dict(size=13, color='#1e293b'),
            ),
            xaxis_title='Classe Predita',
            yaxis_title='Classe Real',
            yaxis=dict(autorange='reversed'),
            margin=dict(l=90, r=25, t=50, b=50),
            paper_bgcolor='white',
            font=dict(size=11, color='#1e293b'),
        )

        # 3. Métricas de Classificação
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_num, zero_division=0
        )
        try:
            score_col = (
                'probabilidade_classe_1'
                if 'probabilidade_classe_1' in df_ep.columns
                else ('confianca' if 'confianca' in df_ep.columns else None)
            )
            y_score = df_ep[score_col].values if score_col else y_pred
            auc_val = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc_val = 0.5

        for idx, nome_cls in enumerate(labels_txt):
            fig_metrics.add_trace(
                go.Bar(
                    name=nome_cls,
                    x=['Precision', 'Recall', 'F1-Score', 'ROC-AUC (Global)'],
                    y=[prec[idx], rec[idx], f1[idx], auc_val],
                    text=[
                        f'{prec[idx]:.2f}',
                        f'{rec[idx]:.2f}',
                        f'{f1[idx]:.2f}',
                        f'{auc_val:.2f}',
                    ],
                    textposition='auto',
                    marker_color='#2563eb' if idx == 0 else '#dc2626',
                )
            )
        fig_metrics.update_layout(
            barmode='group',
            title=dict(
                text=f'Desempenho por Classe — Época {ep_alvo}',
                font=dict(size=13, color='#1e293b'),
            ),
            yaxis=dict(title='Score', range=[0, 1.12]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=45, r=15, t=55, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8fafc',
            font=dict(size=11, color='#1e293b'),
        )

        # 4. Distribuição dos Erros por Classe ao longo das Épocas
        df_preds['erro'] = (
            df_preds['classe_real'].astype(int) != df_preds['classe_predita'].astype(int)
        ).astype(int)
        df_err_grp = (
            df_preds.groupby(['epoca', 'classe_real'], as_index=False)['erro']
            .sum()
            .sort_values('epoca')
        )
        for cls_id in labels_num:
            sub_e = df_err_grp[df_err_grp['classe_real'].astype(int) == cls_id]
            fig_err.add_trace(
                go.Bar(
                    x=[f'Ép {int(e)}' for e in sub_e['epoca']],
                    y=sub_e['erro'],
                    name=mapa_classes.get(cls_id, f'Classe {cls_id}'),
                    marker_color='#2563eb' if cls_id == 0 else '#dc2626',
                )
            )
        fig_err.update_layout(
            barmode='group',
            title=dict(
                text='Total de Erros de Predição por Classe Real ao Longo das Épocas',
                font=dict(size=13, color='#1e293b'),
            ),
            xaxis_title='Época',
            yaxis_title='Quantidade de Erros',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=45, r=15, t=55, b=45),
            paper_bgcolor='white',
            plot_bgcolor='#f8fafc',
            font=dict(size=11, color='#1e293b'),
        )

    return fig_perf, fig_cm, fig_metrics, fig_err


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
