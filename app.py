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

# ==========================================
# CONSTANTES E MAPEAMENTO GLOBAL DE MÉTRICAS XAI
# ==========================================
CODIGO_MEDIA_GERAL = 'MEDIA_GERAL'
LABEL_MEDIA_GERAL = 'Média Geral (4 Métodos)'

METODOS_BASE_XAI = ['SHAP', 'LIME', 'IG', 'PFI']

MAPA_METODOS_XAI = {
    'SHAP': 'SHAP',
    'LIME': 'LIME',
    'IG': 'Integrated Gradients (IG)',
    'PFI': 'Permutation Importance (PFI)',
    CODIGO_MEDIA_GERAL: LABEL_MEDIA_GERAL,
}

CORES_METODOS_XAI = {
    'SHAP': '#2563eb',
    'LIME': '#16a34a',
    'IG': '#9333ea',
    'PFI': '#ea580c',
    CODIGO_MEDIA_GERAL: '#0f172a',
}


def formatar_label_metodo(codigo_metodo):
  codigo_str = str(codigo_metodo).strip()
  return MAPA_METODOS_XAI.get(
      codigo_str, MAPA_METODOS_XAI.get(codigo_str.upper(), codigo_str)
  )


def detectar_coluna_feature(df):
  for col in [
      'feature',
      'atributo',
      'variavel',
      'nome_feature',
      'feature_name',
  ]:
    if col in df.columns:
      return col
  return None


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


def normalizar_coluna_por_grupo(df, col_valor, cols_grupo=['metodo_xai']):
  """Aplica normalização Min-Max [0, 1] globalmente por método XAI através de todas as épocas."""
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


def adicionar_metrica_media_geral(df_d3):
  """Calcula e anexa a métrica sintética MEDIA_GERAL (média normalizada [0,1]

  das 4 métricas XAI: SHAP, LIME, IG, PFI) por época, instância e feature.
  """
  if df_d3.empty or 'metodo_xai' not in df_d3.columns:
    return df_d3

  df_base = df_d3[df_d3['metodo_xai'] != CODIGO_MEDIA_GERAL].copy()
  if df_base.empty:
    return df_d3

  if 'importancia_absoluta' not in df_base.columns:
    if 'valor_atribuicao' in df_base.columns:
      df_base['importancia_absoluta'] = pd.to_numeric(
          df_base['valor_atribuicao'], errors='coerce'
      ).abs()
    elif 'importancia_media' in df_base.columns:
      df_base['importancia_absoluta'] = pd.to_numeric(
          df_base['importancia_media'], errors='coerce'
      ).abs()
    else:
      df_base['importancia_absoluta'] = 1.0
  else:
    df_base['importancia_absoluta'] = pd.to_numeric(
        df_base['importancia_absoluta'], errors='coerce'
    ).fillna(0.0)

  # Normaliza cada um dos 4 métodos em [0, 1] antes de tirar a média geral
  df_norm = normalizar_coluna_por_grupo(
      df_base, 'importancia_absoluta', ['metodo_xai']
  )

  col_feat = detectar_coluna_feature(df_norm)
  chaves_grupo = [
      c for c in ['epoca', 'instancia_id', col_feat] if c and c in df_norm.columns
  ]
  if not chaves_grupo:
    return df_base

  agg_dict = {'importancia_absoluta': 'mean'}
  if 'valor_atribuicao' in df_norm.columns:
    df_norm['valor_atribuicao'] = pd.to_numeric(
        df_norm['valor_atribuicao'], errors='coerce'
    ).fillna(0.0)
    agg_dict['valor_atribuicao'] = 'mean'
  for col_extra in ['classe_real', 'classe_predita', 'modelo_id', 'dataset_name']:
    if col_extra in df_norm.columns:
      agg_dict[col_extra] = 'first'

  df_media = df_norm.groupby(chaves_grupo, as_index=False).agg(agg_dict)
  df_media['metodo_xai'] = CODIGO_MEDIA_GERAL

  return pd.concat([df_base, df_media], ignore_index=True)


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

  if not df_d3.empty:
    if 'instancia_id' not in df_d3.columns:
      for col in ['instance_id', 'id', 'sample_id']:
        if col in df_d3.columns:
          df_d3['instancia_id'] = df_d3[col].astype(str)
          break
      if 'instancia_id' not in df_d3.columns:
        df_d3['instancia_id'] = [str(i) for i in range(len(df_d3))]
    else:
      df_d3['instancia_id'] = df_d3['instancia_id'].astype(str)

    # Incrementa a métrica de Média Geral das 4 métricas XAI
    df_d3 = adicionar_metrica_media_geral(df_d3)

  return df_d3, df_d2


def processar_dados_sankey(df_d3, df_d2, dataset_sel, normalizar=True):
  if df_d3.empty or df_d2.empty:
    return pd.DataFrame(), [], pd.DataFrame()

  df_d3 = df_d3.copy()
  if 'importancia_absoluta' not in df_d3.columns:
    if 'valor_atribuicao' in df_d3.columns:
      df_d3['importancia_absoluta'] = df_d3['valor_atribuicao'].abs()
    else:
      df_d3['importancia_absoluta'] = 1.0

  if normalizar:
    df_d3 = normalizar_coluna_por_grupo(
        df_d3, 'importancia_absoluta', ['metodo_xai']
    )

  if 'instancia_id' not in df_d3.columns:
    for col in ['instance_id', 'id', 'sample_id']:
      if col in df_d3.columns:
        df_d3['instancia_id'] = df_d3[col].astype(str)
        break
    if 'instancia_id' not in df_d3.columns:
      df_d3['instancia_id'] = [str(i) for i in range(len(df_d3))]
  else:
    df_d3['instancia_id'] = df_d3['instancia_id'].astype(str)

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

        transicoes.append({
            'metodo_xai': m,
            'epoca_origem': ep_origem,
            'epoca_destino': ep_destino,
            'instancia_id': str(inst_id),
            'classe_origem': c_orig,
            'classe_destino': c_dest,
            'classe_real': int(
                row_o['classe_real']
                if not isinstance(row_o['classe_real'], pd.Series)
                else row_o['classe_real'].iloc[0]
            ),
            'confianca_media': float(
                (row_o['confianca_media'] + row_d['confianca_media']) / 2.0
                if not isinstance(row_o['confianca_media'], pd.Series)
                else (
                    row_o['confianca_media'].iloc[0]
                    + row_d['confianca_media'].iloc[0]
                )
                / 2.0
            ),
            'taxa_erro': float(
                (row_o['taxa_erro'] + row_d['taxa_erro']) / 2.0
                if not isinstance(row_o['taxa_erro'], pd.Series)
                else (
                    row_o['taxa_erro'].iloc[0] + row_d['taxa_erro'].iloc[0]
                )
                / 2.0
            ),
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
                                        'Método / Métrica XAI:',
                                        style={
                                            'fontSize': '11px',
                                            'fontWeight': 'bold',
                                            'color': '#334155',
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id='dropdown-metodo',
                                        options=[
                                            {
                                                'label': LABEL_MEDIA_GERAL,
                                                'value': CODIGO_MEDIA_GERAL,
                                            }
                                        ],
                                        value=CODIGO_MEDIA_GERAL,
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
                                                'value': (
                                                    'variancia_importancia'
                                                ),
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
                                'boxShadow': (
                                    '0 4px 6px -1px rgba(0, 0, 0, 0.05)'
                                ),
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
                                                # 2. Heatmap e Estabilidade XAI
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
                                                                        'Tabela Auxiliar de Instâncias (Seleção por Brushing) ',
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
                                                        lg=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardHeader(
                                                                        'Métricas de Classificação por Época (Precision, Recall, F1, AUC)',
                                                                        style={
                                                                            'backgroundColor': '#f8fafc',
                                                                            'fontWeight': 'bold',
                                                                            'color': '#1e293b',
                                                                        },
                                                                    ),
                                                                    dbc.CardBody(
                                                                        [
                                                                            dcc.Graph(
                                                                                id='validation-metrics-graph',
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
                                                        lg=6,
                                                    ),
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
            'backgroundColor': '#f1f5f9',
            'minHeight': '100vh',
            'paddingTop': '20px',
        },
    ),
])


# ==========================================
# 3. CALLBACKS REATIVOS DO DASHBOARD
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
    [State('dropdown-metodo', 'value')],
)
def atualizar_controles_dataset(dataset_sel, metodo_atual):
  df_d3, df_d2 = carregar_dados_sankey(dataset_sel)

  metodos_disponiveis = []
  if not df_d3.empty and 'metodo_xai' in df_d3.columns:
    unicos = [
        str(m)
        for m in df_d3['metodo_xai'].dropna().unique()
        if str(m) != CODIGO_MEDIA_GERAL
    ]
    ordenados = [m for m in METODOS_BASE_XAI if m in unicos] + [
        m for m in unicos if m not in METODOS_BASE_XAI
    ]
    metodos_disponiveis = ordenados

  opcoes_metodo = [
      {'label': LABEL_MEDIA_GERAL, 'value': CODIGO_MEDIA_GERAL}
  ] + [
      {'label': formatar_label_metodo(m), 'value': m}
      for m in metodos_disponiveis
  ]

  valores_validos = [o['value'] for o in opcoes_metodo]
  metodo_sel = (
      metodo_atual if metodo_atual in valores_validos else CODIGO_MEDIA_GERAL
  )

  epocas = []
  if not df_d3.empty and 'epoca' in df_d3.columns:
    epocas = sorted(
        pd.to_numeric(df_d3['epoca'], errors='coerce')
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
  elif not df_d2.empty and 'epoca' in df_d2.columns:
    epocas = sorted(
        pd.to_numeric(df_d2['epoca'], errors='coerce')
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

  if not epocas:
    epocas = [1, 40]

  ep_min, ep_max = epocas[0], epocas[-1]
  marks = {int(e): str(e) for e in epocas}
  opcoes_epocas = [{'label': f'Época {e}', 'value': int(e)} for e in epocas]

  return (
      opcoes_metodo,
      metodo_sel,
      ep_min,
      ep_max,
      marks,
      [ep_min, ep_max],
      opcoes_epocas,
      ep_max,
  )


@app.callback(
    [
        Output('store-instancias-selecionadas', 'data'),
        Output('badge-limpar-sankey', 'children'),
        Output('badge-limpar-sankey', 'style'),
    ],
    [
        Input('dim-scatter-graph', 'selectedData'),
        Input('sankey-graph', 'clickData'),
        Input('badge-limpar-sankey', 'n_clicks'),
        Input('dropdown-dataset', 'value'),
    ],
    [State('store-instancias-selecionadas', 'data')],
)
def sincronizar_selecao_instancias(
    selected_data, sankey_click, n_clicks_limpar, dataset_sel, selecao_atual
):
  ctx = callback_context
  trigger_id = (
      ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
  )

  estilo_oculto = {
      'cursor': 'pointer',
      'fontSize': '11px',
      'padding': '5px 9px',
      'borderRadius': '12px',
      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
      'display': 'none',
  }
  estilo_visivel = {**estilo_oculto, 'display': 'inline-block'}

  if trigger_id in ['badge-limpar-sankey', 'dropdown-dataset']:
    return [], '', estilo_oculto

  novas_insts = list(selecao_atual or [])

  if trigger_id == 'dim-scatter-graph':
    if selected_data and selected_data.get('points'):
      novas_insts = sorted(
          list({
              str(p['customdata'][0])
              for p in selected_data['points']
              if p.get('customdata')
          })
      )
    else:
      novas_insts = []

  elif trigger_id == 'sankey-graph' and sankey_click and sankey_click.get('points'):
    pt = sankey_click['points'][0]
    cd = pt.get('customdata')
    if isinstance(cd, list):
      novas_insts = sorted(list({str(x) for x in cd}))
    elif cd is not None:
      novas_insts = [str(cd)]

  if novas_insts:
    return (
        novas_insts,
        f'Limpar Filtro ({len(novas_insts)} instâncias)',
        estilo_visivel,
    )
  return [], '', estilo_oculto


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
    dataset_sel, metodo_sel, cor_metrica, intervalo_epocas, insts_selecionadas
):
  df_d3, df_d2 = carregar_dados_sankey(dataset_sel)
  if df_d3.empty or df_d2.empty:
    return go.Figure().update_layout(
        title='Dados indisponíveis para o Sankey'
    )

  if insts_selecionadas:
    df_d3_filtrado = df_d3[
        df_d3['instancia_id'].astype(str).isin(
            [str(i) for i in insts_selecionadas]
        )
    ]
    if not df_d3_filtrado.empty:
      df_d3 = df_d3_filtrado

  df_sankey, _, _ = processar_dados_sankey(
      df_d3, df_d2, dataset_sel, normalizar=True
  )
  if df_sankey.empty:
    return go.Figure().update_layout(
        title='Sem transições suficientes para exibir o Sankey'
    )

  metodo_alvo = metodo_sel or CODIGO_MEDIA_GERAL
  df_m = df_sankey[df_sankey['metodo_xai'] == metodo_alvo].copy()
  if df_m.empty:
    df_m = df_sankey.copy()

  if intervalo_epocas and len(intervalo_epocas) == 2:
    ep_ini, ep_fim = intervalo_epocas
    df_m = df_m[
        (df_m['epoca_origem'] >= ep_ini) & (df_m['epoca_destino'] <= ep_fim)
    ]

  if df_m.empty:
    return go.Figure().update_layout(
        title='Nenhuma transição no intervalo de épocas selecionado'
    )

  mapa_classes = Mapeamento_classes_dataset(dataset_sel)

  nos_unicos = []
  for _, row in df_m.iterrows():
    n_orig = f"Ép {int(row['epoca_origem'])} | {mapa_classes.get(int(row['classe_origem']), row['classe_origem'])}"
    n_dest = f"Ép {int(row['epoca_destino'])} | {mapa_classes.get(int(row['classe_destino']), row['classe_destino'])}"
    if n_orig not in nos_unicos:
      nos_unicos.append(n_orig)
    if n_dest not in nos_unicos:
      nos_unicos.append(n_dest)

  idx_nos = {nome: i for i, nome in enumerate(nos_unicos)}
  cores_nos = [
      '#3b82f6' if mapa_classes.get(0, 'Classe 0') in nome else '#ef4444'
      for nome in nos_unicos
  ]

  sources, targets, values, link_colors, customdata, hovertexts = (
      [],
      [],
      [],
      [],
      [],
      [],
  )
  for _, row in df_m.iterrows():
    n_orig = f"Ép {int(row['epoca_origem'])} | {mapa_classes.get(int(row['classe_origem']), row['classe_origem'])}"
    n_dest = f"Ép {int(row['epoca_destino'])} | {mapa_classes.get(int(row['classe_destino']), row['classe_destino'])}"
    sources.append(idx_nos[n_orig])
    targets.append(idx_nos[n_dest])
    peso = max(float(row['soma_importancia']), 1e-4)
    values.append(peso)
    customdata.append(row['lista_instancias'])

    if cor_metrica == 'confianca':
      alpha = np.clip(float(row['confianca_media']), 0.25, 0.85)
      link_colors.append(f'rgba(16, 185, 129, {alpha:.2f})')
    elif cor_metrica == 'erro':
      err = float(row['taxa_erro'])
      link_colors.append(
          f'rgba(239, 68, 68, {np.clip(0.25 + err * 0.6, 0.25, 0.85):.2f})'
      )
    else:
      c_orig = int(row['classe_origem'])
      link_colors.append(
          'rgba(59, 130, 246, 0.45)'
          if c_orig == 0
          else 'rgba(239, 68, 68, 0.45)'
      )

    hovertexts.append(
        f'<b>{formatar_label_metodo(metodo_alvo)}</b><br>'
        f'{n_orig} → {n_dest}<br>'
        f"Instâncias: {int(row['qtd_instancias'])}<br>"
        f"Importância Média: {row['importancia_media']:.4f}<br>"
        f"Confiança Média: {row['confianca_media']:.2%}<br>"
        f"Taxa de Erro: {row['taxa_erro']:.2%}"
    )

  fig = go.Figure(
      data=[
          go.Sankey(
              node=dict(
                  pad=15,
                  thickness=18,
                  line=dict(color='#1e293b', width=0.5),
                  label=nos_unicos,
                  color=cores_nos,
              ),
              link=dict(
                  source=sources,
                  target=targets,
                  value=values,
                  color=link_colors,
                  customdata=customdata,
                  hovertemplate='%{customdata_hover}<extra></extra>',
              ),
          )
      ]
  )
  fig.data[0].link.hovertemplate = '%{label}<extra></extra>'
  fig.data[0].link.label = hovertexts

  fig.update_layout(
      title=dict(
          text=f'Fluxo Temporal de Decisões — {formatar_label_metodo(metodo_alvo)}',
          font=dict(size=14, color='#1e293b'),
      ),
      font=dict(size=11),
      margin=dict(l=15, r=15, t=45, b=15),
  )
  return fig


@app.callback(
    Output('heatmap-graph', 'figure'),
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('store-instancias-selecionadas', 'data'),
    ],
)
def atualizar_heatmap(dataset_sel, metodo_sel, insts_selecionadas):
  df_d3, _ = carregar_dados_sankey(dataset_sel)
  if df_d3.empty:
    return go.Figure().update_layout(
        title='Sem dados de explicação para o Heatmap'
    )

  col_feat = detectar_coluna_feature(df_d3)
  if not col_feat:
    return go.Figure().update_layout(
        title='Coluna de features não identificada no dataset'
    )

  if insts_selecionadas:
    df_sub = df_d3[
        df_d3['instancia_id'].astype(str).isin(
            [str(i) for i in insts_selecionadas]
        )
    ]
    if not df_sub.empty:
      df_d3 = df_sub

  df_d3 = normalizar_coluna_por_grupo(
      df_d3, 'importancia_absoluta', ['metodo_xai']
  )
  df_d3['epoca'] = (
      pd.to_numeric(df_d3['epoca'], errors='coerce').fillna(0).astype(int)
  )

  metodo_alvo = metodo_sel or CODIGO_MEDIA_GERAL
  df_m = df_d3[df_d3['metodo_xai'] == metodo_alvo].copy()
  if df_m.empty:
    df_m = df_d3.copy()

  # Agrega importância média por feature e época
  df_agg = (
      df_m.groupby([col_feat, 'epoca'], as_index=False)['importancia_absoluta']
      .mean()
  )

  # Ordena features pela importância global (mais relevantes no topo)
  ordem_features = (
      df_agg.groupby(col_feat)['importancia_absoluta']
      .mean()
      .sort_values(ascending=True)
      .index.tolist()
  )
  epocas_ord = sorted(df_agg['epoca'].unique().tolist())

  piv = df_agg.pivot(
      index=col_feat, columns='epoca', values='importancia_absoluta'
  ).reindex(index=ordem_features, columns=epocas_ord)

  # Constrói hover detalhado: quando MEDIA_GERAL está selecionado, exibe também
  # o valor individual das 4 métricas base (SHAP, LIME, IG, PFI)
  detalhes_por_metodo = {}
  if metodo_alvo == CODIGO_MEDIA_GERAL:
    for m_base in METODOS_BASE_XAI:
      df_mb = df_d3[df_d3['metodo_xai'] == m_base]
      if not df_mb.empty:
        piv_mb = (
            df_mb.groupby([col_feat, 'epoca'])['importancia_absoluta']
            .mean()
            .unstack()
            .reindex(index=ordem_features, columns=epocas_ord)
        )
        detalhes_por_metodo[m_base] = piv_mb

  hover_matrix = []
  for feat in ordem_features:
    linha_hover = []
    for ep in epocas_ord:
      val = piv.loc[feat, ep]
      val_str = f'{val:.4f}' if pd.notna(val) else 'N/A'
      txt = (
          f'<b>Feature:</b> {feat}<br>'
          f'<b>Época:</b> {ep}<br>'
          f'<b>{formatar_label_metodo(metodo_alvo)}:</b> {val_str}'
      )
      if metodo_alvo == CODIGO_MEDIA_GERAL and detalhes_por_metodo:
        txt += '<br>--- Detalhe por Métrica ---'
        for m_base, piv_mb in detalhes_por_metodo.items():
          v_mb = piv_mb.loc[feat, ep]
          if pd.notna(v_mb):
            txt += f'<br>• {m_base}: {v_mb:.4f}'
      linha_hover.append(txt)
    hover_matrix.append(linha_hover)

  fig = go.Figure(
      data=go.Heatmap(
          z=piv.values,
          x=[str(e) for e in epocas_ord],
          y=ordem_features,
          colorscale=COLORSCALE_REDS_20,
          zmin=0.0,
          zmax=1.0,
          text=hover_matrix,
          hovertemplate='%{text}<extra></extra>',
          colorbar=dict(title='Importância<br>(Norm.)'),
      )
  )

  fig.update_layout(
      title=dict(
          text=f'Evolução Temporal das Features — {formatar_label_metodo(metodo_alvo)}',
          font=dict(size=14, color='#1e293b'),
      ),
      xaxis_title='Época de Treinamento',
      yaxis_title='Atributo (Feature)',
      margin=dict(l=140, r=20, t=45, b=45),
  )
  return fig


@app.callback(
    Output('stability-graph', 'figure'),
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dropdown-metrica-estabilidade', 'value'),
    ],
)
def atualizar_estabilidade(dataset_sel, metodo_sel, metrica_estab):
  df_d3, _ = carregar_dados_sankey(dataset_sel)
  if df_d3.empty:
    return go.Figure().update_layout(
        title='Sem dados para calcular estabilidade'
    )

  col_feat = detectar_coluna_feature(df_d3)
  if not col_feat:
    return go.Figure().update_layout(title='Coluna de features ausente')

  df_d3 = normalizar_coluna_por_grupo(
      df_d3, 'importancia_absoluta', ['metodo_xai']
  )
  df_d3['epoca'] = (
      pd.to_numeric(df_d3['epoca'], errors='coerce').fillna(0).astype(int)
  )

  # Inclui os 4 métodos base + MEDIA_GERAL para comparação direta
  metodos_todos = [
      m
      for m in (METODOS_BASE_XAI + [CODIGO_MEDIA_GERAL])
      if m in df_d3['metodo_xai'].unique()
  ]

  fig = go.Figure()
  for m in metodos_todos:
    df_m = df_d3[df_d3['metodo_xai'] == m]
    piv = (
        df_m.groupby([col_feat, 'epoca'])['importancia_absoluta']
        .mean()
        .unstack()
    )
    epocas = sorted(piv.columns.tolist())
    if len(epocas) < 2:
      continue

    eixos_x, valores_y = [], []
    for i in range(1, len(epocas)):
      ep_ant, ep_atual = epocas[i - 1], epocas[i]
      s1 = piv[ep_ant].fillna(0.0)
      s2 = piv[ep_atual].fillna(0.0)

      if metrica_estab == 'spearman':
        val = s1.corr(s2, method='spearman')
        val = 1.0 if pd.isna(val) else float(val)
      elif metrica_estab == 'mad_importancia':
        val = float(np.mean(np.abs(s2 - s1)))
      else:
        val = float(np.var(s2 - s1))

      eixos_x.append(f'{ep_ant}→{ep_atual}')
      valores_y.append(val)

    eh_selecionado = m == metodo_sel
    eh_media_geral = m == CODIGO_MEDIA_GERAL

    fig.add_trace(
        go.Scatter(
            x=eixos_x,
            y=valores_y,
            mode='lines+markers',
            name=formatar_label_metodo(m),
            line=dict(
                color=CORES_METODOS_XAI.get(m, '#475569'),
                width=3.5 if (eh_selecionado or eh_media_geral) else 1.8,
                dash='dash' if eh_media_geral else 'solid',
            ),
            marker=dict(size=7 if eh_selecionado else 5),
            opacity=1.0 if (eh_selecionado or eh_media_geral) else 0.65,
        )
    )

  nomes_metrica = {
      'spearman': 'Correlação de Spearman (Maior = Mais Estável)',
      'mad_importancia': 'MAD Absoluto entre Épocas (Menor = Mais Estável)',
      'variancia_importancia': 'Variância entre Épocas (Menor = Mais Estável)',
  }

  fig.update_layout(
      title=dict(
          text=nomes_metrica.get(metrica_estab, 'Estabilidade Temporal'),
          font=dict(size=13, color='#1e293b'),
      ),
      xaxis_title='Transição de Épocas',
      yaxis_title='Valor da Métrica',
      legend=dict(
          orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
      ),
      margin=dict(l=50, r=20, t=65, b=45),
      hovermode='x unified',
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
def atualizar_visualizacao_2d(
    dataset_sel,
    metodo_sel,
    epoca_sel,
    algo_sel,
    cor_por,
    toggle_trajetoria,
    insts_selecionadas,
):
  df_d3, df_d2 = carregar_dados_sankey(dataset_sel)
  df_proj = carregar_csv_seguro('projecao_tsne_epocas', dataset_sel)

  metodo_alvo = metodo_sel or CODIGO_MEDIA_GERAL
  mapa_classes = Mapeamento_classes_dataset(dataset_sel)

  if not df_d3.empty:
    df_d3 = normalizar_coluna_por_grupo(
        df_d3, 'importancia_absoluta', ['metodo_xai']
    )
    df_d3['epoca'] = (
        pd.to_numeric(df_d3['epoca'], errors='coerce').fillna(0).astype(int)
    )

  # Constrói projeção 2D (t-SNE ou PCA) se não houver CSV dedicado
  if df_proj.empty and not df_d3.empty:
    col_feat = detectar_coluna_feature(df_d3)
    df_base_proj = df_d3[df_d3['metodo_xai'] == metodo_alvo].copy()
    if df_base_proj.empty:
      df_base_proj = df_d3.copy()

    if col_feat:
      mat = (
          df_base_proj.pivot_table(
              index=['epoca', 'instancia_id'],
              columns=col_feat,
              values='importancia_absoluta',
              fill_value=0.0,
          )
          .reset_index()
      )
      coords = PCA(n_components=2, random_state=42).fit_transform(
          mat.drop(columns=['epoca', 'instancia_id']).values
      )
      df_proj = mat[['epoca', 'instancia_id']].copy()
      df_proj['tsne_dim_1'] = coords[:, 0]
      df_proj['tsne_dim_2'] = coords[:, 1]
      df_proj['pca_dim_1'] = coords[:, 0]
      df_proj['pca_dim_2'] = coords[:, 1]

  if df_proj.empty:
    vazio = go.Figure().update_layout(title='Dados espaciais 2D indisponíveis')
    return vazio, vazio, []

  df_proj['epoca'] = (
      pd.to_numeric(df_proj['epoca'], errors='coerce').fillna(0).astype(int)
  )
  df_proj['instancia_id'] = df_proj['instancia_id'].astype(str)

  if 'metodo_xai' in df_proj.columns:
    if metodo_alvo in df_proj['metodo_xai'].unique():
      df_proj = df_proj[df_proj['metodo_xai'] == metodo_alvo].copy()
    elif metodo_alvo == CODIGO_MEDIA_GERAL:
      cols_num = [
          c
          for c in ['tsne_dim_1', 'tsne_dim_2', 'pca_dim_1', 'pca_dim_2']
          if c in df_proj.columns
      ]
      agg_p = {c: 'mean' for c in cols_num}
      for c_extra in ['classe_real', 'classe_predita']:
        if c_extra in df_proj.columns:
          agg_p[c_extra] = 'first'
      df_proj = df_proj.groupby(['epoca', 'instancia_id'], as_index=False).agg(
          agg_p
      )

  if algo_sel == 'PCA' and {'pca_dim_1', 'pca_dim_2'}.issubset(df_proj.columns):
    col_x, col_y = 'pca_dim_1', 'pca_dim_2'
  elif {'tsne_dim_1', 'tsne_dim_2'}.issubset(df_proj.columns):
    col_x, col_y = 'tsne_dim_1', 'tsne_dim_2'
  else:
    num_cols = df_proj.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['epoca', 'classe_real', 'classe_predita']]
    col_x, col_y = num_cols[0], num_cols[1]

  # Enriquece com importância e classes vindas de df_d3
  if not df_d3.empty:
    df_imp = (
        df_d3[df_d3['metodo_xai'] == metodo_alvo]
        .groupby(['epoca', 'instancia_id'], as_index=False)
        .agg(
            importancia_total=('importancia_absoluta', 'sum'),
            **{
                c: (c, 'first')
                for c in ['classe_real', 'classe_predita']
                if c in df_d3.columns
            },
        )
    )
    cols_drop = [
        c
        for c in ['classe_real', 'classe_predita', 'importancia_total']
        if c in df_proj.columns and c in df_imp.columns
    ]
    df_proj = df_proj.drop(columns=cols_drop).merge(
        df_imp, on=['epoca', 'instancia_id'], how='left'
    )

  if 'classe_real' not in df_proj.columns:
    df_proj['classe_real'] = 0
  if 'classe_predita' not in df_proj.columns:
    df_proj['classe_predita'] = df_proj['classe_real']
  if 'importancia_total' not in df_proj.columns:
    df_proj['importancia_total'] = 0.5

  df_proj['classe_real'] = df_proj['classe_real'].fillna(0).astype(int)
  df_proj['classe_predita'] = df_proj['classe_predita'].fillna(0).astype(int)
  df_proj['erro_predicao'] = np.where(
      df_proj['classe_real'] != df_proj['classe_predita'], 'Erro', 'Acerto'
  )

  ep_alvo = epoca_sel if epoca_sel is not None else int(df_proj['epoca'].max())
  df_ep = df_proj[df_proj['epoca'] == ep_alvo].copy()
  if df_ep.empty:
    df_ep = df_proj.copy()

  if cor_por == 'erro_predicao':
    df_ep['grupo_cor'] = df_ep['erro_predicao']
    mapa_cores = {'Acerto': '#10b981', 'Erro': '#ef4444'}
  elif cor_por == 'classe_real':
    df_ep['grupo_cor'] = df_ep['classe_real'].map(mapa_classes)
    mapa_cores = {
        mapa_classes.get(0, 'Classe 0'): '#3b82f6',
        mapa_classes.get(1, 'Classe 1'): '#f59e0b',
    }
  else:
    df_ep['grupo_cor'] = df_ep['classe_predita'].map(mapa_classes)
    mapa_cores = {
        mapa_classes.get(0, 'Classe 0'): '#3b82f6',
        mapa_classes.get(1, 'Classe 1'): '#ef4444',
    }

  fig_scatter = px.scatter(
      df_ep,
      x=col_x,
      y=col_y,
      color='grupo_cor',
      color_discrete_map=mapa_cores,
      custom_data=['instancia_id', 'importancia_total'],
      hover_data={
          'instancia_id': True,
          'importancia_total': ':.4f',
          'grupo_cor': True,
          col_x: ':.2f',
          col_y: ':.2f',
      },
      title=f'Projeção {algo_sel} (Época {ep_alvo}) — {formatar_label_metodo(metodo_alvo)}',
  )
  fig_scatter.update_traces(marker=dict(size=8, opacity=0.85))

  # Trajetória temporal das instâncias selecionadas
  if toggle_trajetoria and 'ATIVO' in toggle_trajetoria and insts_selecionadas:
    df_traj = df_proj[
        df_proj['instancia_id'].isin([str(i) for i in insts_selecionadas])
    ].sort_values('epoca')
    for inst_id, grp in df_traj.groupby('instancia_id'):
      fig_scatter.add_trace(
          go.Scatter(
              x=grp[col_x],
              y=grp[col_y],
              mode='lines+markers',
              name=f'Traj. {inst_id}',
              line=dict(width=1.5, dash='dot'),
              marker=dict(size=5),
              showlegend=False,
          )
      )

  fig_scatter.update_layout(
      margin=dict(l=40, r=20, t=45, b=40),
      legend_title_text='',
  )

  # Boxplot Auxiliar Temporal por Época
  df_box = df_proj.copy()
  if insts_selecionadas:
    df_box_sub = df_box[
        df_box['instancia_id'].isin([str(i) for i in insts_selecionadas])
    ]
    if not df_box_sub.empty:
      df_box = df_box_sub

  fig_box = px.box(
      df_box,
      x='epoca',
      y='importancia_total',
      color_discrete_sequence=[CORES_METODOS_XAI.get(metodo_alvo, '#2563eb')],
      title=f'Distribuição Temporal da Importância — {formatar_label_metodo(metodo_alvo)}',
  )
  fig_box.update_layout(
      xaxis_title='Época',
      yaxis_title='Importância Agregada (Norm.)',
      margin=dict(l=45, r=20, t=45, b=40),
  )

  # Dados para a tabela auxiliar
  df_tab = df_ep.copy()
  if insts_selecionadas:
    df_tab_sub = df_tab[
        df_tab['instancia_id'].isin([str(i) for i in insts_selecionadas])
    ]
    if not df_tab_sub.empty:
      df_tab = df_tab_sub

  df_tab['metodo_xai'] = formatar_label_metodo(metodo_alvo)
  df_tab['classe_real'] = df_tab['classe_real'].map(mapa_classes)
  df_tab['classe_predita'] = df_tab['classe_predita'].map(mapa_classes)
  df_tab['tsne_dim_1'] = df_tab[col_x].round(4)
  df_tab['tsne_dim_2'] = df_tab[col_y].round(4)

  colunas_tabela = [
      'instancia_id',
      'epoca',
      'metodo_xai',
      'classe_real',
      'classe_predita',
      'erro_predicao',
      'tsne_dim_1',
      'tsne_dim_2',
  ]
  dados_tabela = df_tab[colunas_tabela].head(200).to_dict('records')

  return fig_scatter, fig_box, dados_tabela


@app.callback(
    [
        Output('performance-graph', 'figure'),
        Output('confusion-matrix-graph', 'figure'),
        Output('validation-metrics-graph', 'figure'),
    ],
    [
        Input('dropdown-dataset', 'value'),
        Input('dim-dropdown-epoca', 'value'),
    ],
)
def atualizar_validacao_modelo(dataset_sel, epoca_sel):
  df_d3, df_d2 = carregar_dados_sankey(dataset_sel)
  mapa_classes = Mapeamento_classes_dataset(dataset_sel)

  if df_d2.empty:
    vazio = go.Figure().update_layout(
        title='Métricas de validação indisponíveis'
    )
    return vazio, vazio, vazio

  df_d2 = df_d2.copy()
  df_d2['epoca'] = (
      pd.to_numeric(df_d2['epoca'], errors='coerce').fillna(0).astype(int)
  )
  df_d2 = df_d2.sort_values('epoca')

  ep_alvo = epoca_sel if epoca_sel is not None else int(df_d2['epoca'].max())

  # 1. Gráfico de Performance (Acurácia & Loss)
  fig_perf = go.Figure()
  if 'accuracy' in df_d2.columns:
    fig_perf.add_trace(
        go.Scatter(
            x=df_d2['epoca'],
            y=df_d2['accuracy'],
            mode='lines+markers',
            name='Acurácia',
            line=dict(color='#16a34a', width=2.5),
        )
    )
  if 'loss_validacao' in df_d2.columns:
    fig_perf.add_trace(
        go.Scatter(
            x=df_d2['epoca'],
            y=df_d2['loss_validacao'],
            mode='lines+markers',
            name='Loss Validação',
            yaxis='y2',
            line=dict(color='#dc2626', width=2.5, dash='dash'),
        )
    )

  fig_perf.add_vline(
      x=ep_alvo,
      line_width=1.5,
      line_dash='dot',
      line_color='#334155',
      annotation_text=f'Época {ep_alvo}',
  )
  fig_perf.update_layout(
      xaxis_title='Época',
      yaxis=dict(title='Acurácia', range=[0, 1.05]),
      yaxis2=dict(title='Loss', overlaying='y', side='right'),
      legend=dict(orientation='h', y=1.12, x=0),
      margin=dict(l=50, r=50, t=45, b=40),
      hovermode='x unified',
  )

  # 2. Matriz de Confusão na Época Selecionada
  cm = np.array([[0, 0], [0, 0]])
  if (
      not df_d3.empty
      and {'classe_real', 'classe_predita', 'epoca'}.issubset(df_d3.columns)
  ):
    df_ep_inst = (
        df_d3[df_d3['epoca'] == ep_alvo]
        .drop_duplicates(subset=['instancia_id'])
    )
    if not df_ep_inst.empty:
      cm = confusion_matrix(
          df_ep_inst['classe_real'].astype(int),
          df_ep_inst['classe_predita'].astype(int),
          labels=[0, 1],
      )

  labels_cls = [
      mapa_classes.get(0, 'Classe 0'),
      mapa_classes.get(1, 'Classe 1'),
  ]
  fig_cm = go.Figure(
      data=go.Heatmap(
          z=cm,
          x=labels_cls,
          y=labels_cls,
          text=cm,
          texttemplate='%{text}',
          textfont=dict(size=18),
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
      margin=dict(l=80, r=30, t=45, b=50),
  )

  # 3. Métricas Secundárias (Precision, Recall, F1, ROC-AUC)
  fig_met = go.Figure()
  metricas_candidatas = [
      ('precision', 'Precision', '#2563eb'),
      ('recall', 'Recall', '#9333ea'),
      ('f1_score', 'F1-Score', '#ea580c'),
      ('roc_auc', 'ROC-AUC', '#0d9488'),
  ]
  for col_m, label_m, cor_m in metricas_candidatas:
    if col_m in df_d2.columns:
      fig_met.add_trace(
          go.Scatter(
              x=df_d2['epoca'],
              y=df_d2[col_m],
              mode='lines+markers',
              name=label_m,
              line=dict(color=cor_m, width=2),
          )
      )

  if not fig_met.data and 'accuracy' in df_d2.columns:
    fig_met.add_trace(
        go.Scatter(
            x=df_d2['epoca'],
            y=df_d2['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#2563eb', width=2),
        )
    )

  fig_met.add_vline(
      x=ep_alvo, line_width=1.5, line_dash='dot', line_color='#334155'
  )
  fig_met.update_layout(
      xaxis_title='Época',
      yaxis=dict(title='Score', range=[0, 1.05]),
      legend=dict(orientation='h', y=1.12, x=0),
      margin=dict(l=50, r=20, t=45, b=40),
      hovermode='x unified',
  )

  return fig_perf, fig_cm, fig_met


if __name__ == '__main__':
  app.run(debug=True)
