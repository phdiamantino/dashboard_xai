import glob
import os
import shutil
import zipfile
from functools import lru_cache

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

# Método XAI sintético que combina todos os métodos XAI
METODO_COMBINADO = 'MEDIA_GERAL'
LABEL_COMBINADO = 'MÉDIA GERAL (todos os XAI)'


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


def combinar_metodos_xai(df, chaves, cols_media):
  """Cria o método XAI combinado (METODO_COMBINADO).

  Agrupa as linhas de todos os métodos pelas `chaves` (ex.: época, instância,
  feature) e tira a média das colunas numéricas em `cols_media`. As demais
  colunas recebem o primeiro valor do grupo. Deve ser chamada DEPOIS da
  normalização por método, para que nenhum método domine a média.
  """
  if df.empty or 'metodo_xai' not in df.columns:
    return df

  df = df.copy()
  df['metodo_xai'] = df['metodo_xai'].astype(str)
  df = df[df['metodo_xai'] != METODO_COMBINADO]
  chaves_validas = [c for c in chaves if c in df.columns]
  if df.empty or not chaves_validas:
    return df

  agg = {}
  for c in df.columns:
    if c in chaves_validas or c == 'metodo_xai':
      continue
    if c in cols_media and pd.api.types.is_numeric_dtype(df[c]):
      agg[c] = 'mean'
    else:
      agg[c] = 'first'

  comb = df.groupby(chaves_validas, as_index=False).agg(agg)
  comb['metodo_xai'] = METODO_COMBINADO
  return pd.concat([df, comb[df.columns]], ignore_index=True)


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


def _d6_combinado(df_d6, df_d3):
  """Projeção 2D do método combinado.

  As coordenadas t-SNE de métodos diferentes vivem em espaços diferentes, então
  não faz sentido tirar a média delas. O método combinado usa um PCA conjunto
  (todas as épocas) sobre as atribuições combinadas. Se não for possível,
  usa a média das coordenadas como fallback.
  """
  req = ['metodo_xai', 'epoca', 'instancia_id']
  if df_d6.empty or any(c not in df_d6.columns for c in req):
    return df_d6

  base = df_d6[df_d6['metodo_xai'].astype(str) != METODO_COMBINADO].copy()
  if base.empty:
    return df_d6
  base['instancia_id'] = base['instancia_id'].astype(str)
  base['epoca'] = pd.to_numeric(base['epoca'], errors='coerce')
  base = base.dropna(subset=['epoca'])

  coords = [c for c in ['tsne_dim_1', 'tsne_dim_2'] if c in base.columns]
  comb = base.drop_duplicates(['epoca', 'instancia_id']).copy()
  comb['metodo_xai'] = METODO_COMBINADO

  # Fallback: média das coordenadas entre os métodos
  if len(coords) == 2:
    medias = base.groupby(['epoca', 'instancia_id'], as_index=False)[coords].mean()
    comb = comb.drop(columns=coords).merge(
        medias, on=['epoca', 'instancia_id'], how='left'
    )

    cols_d3 = {'metodo_xai', 'epoca', 'instancia_id', 'feature', 'valor_atribuicao'}
    if not df_d3.empty and cols_d3.issubset(df_d3.columns):
      d3c = df_d3[df_d3['metodo_xai'] == METODO_COMBINADO].copy()
      d3c['epoca'] = pd.to_numeric(d3c['epoca'], errors='coerce')
      d3c['instancia_id'] = d3c['instancia_id'].astype(str)
      piv = d3c.pivot_table(
          index=['epoca', 'instancia_id'],
          columns='feature',
          values='valor_atribuicao',
          aggfunc='mean',
      ).fillna(0)
      if len(piv) >= 2 and piv.shape[1] >= 2:
        res = PCA(n_components=2).fit_transform(piv.values)
        pca_df = pd.DataFrame(res, index=piv.index, columns=coords).reset_index()
        comb = comb.drop(columns=coords).merge(
            pca_df, on=['epoca', 'instancia_id'], how='left'
        )
    comb = comb.dropna(subset=coords)

  return pd.concat([base, comb[base.columns]], ignore_index=True)


@lru_cache(maxsize=8)
def carregar_bundle(dataset_sel):
  """Carrega e prepara todos os dados do dataset (com o método combinado).

  Os DataFrames retornados são compartilhados via cache: não modificar
  in-place (use .copy()).
  """
  df_d2 = carregar_csv_seguro('metricas_epocas', dataset_sel)
  df_d3 = carregar_csv_seguro('explicacoes_instancias', dataset_sel)
  df_d4 = carregar_csv_seguro('resumo_explicacoes', dataset_sel)
  df_d5 = carregar_csv_seguro('metricas_estabilidade_concordancia', dataset_sel)
  df_d6 = carregar_csv_seguro('projecao_tsne_epocas', dataset_sel)

  # D3: normalização por método (preservando evolução temporal) + combinado
  if not df_d3.empty:
    df_d3 = df_d3.copy()
    if 'valor_atribuicao' in df_d3.columns:
      df_d3['valor_atribuicao_abs'] = df_d3['valor_atribuicao'].abs()
      df_d3 = normalizar_coluna_por_grupo(
          df_d3, 'valor_atribuicao_abs', ['metodo_xai']
      )
      df_d3['valor_atribuicao'] = (
          np.sign(df_d3['valor_atribuicao']) * df_d3['valor_atribuicao_abs']
      )
      df_d3 = df_d3.drop(columns=['valor_atribuicao_abs'])
    if 'importancia_absoluta' in df_d3.columns:
      df_d3 = normalizar_coluna_por_grupo(
          df_d3, 'importancia_absoluta', ['metodo_xai']
      )
    df_d3 = combinar_metodos_xai(
        df_d3,
        ['epoca', 'instancia_id', 'feature'],
        ['valor_atribuicao', 'importancia_absoluta'],
    )

  # D4 (heatmap): |importância| normalizada por método + combinado
  if (
      not df_d4.empty
      and {'metodo_xai', 'feature', 'epoca'}.issubset(df_d4.columns)
  ):
    val_col = (
        'importancia_media'
        if 'importancia_media' in df_d4.columns
        else df_d4.columns[-1]
    )
    df_h = df_d4.copy()
    if val_col != 'importancia_media':
      df_h = df_h.rename(columns={val_col: 'importancia_media'})
    df_h['importancia_media'] = pd.to_numeric(
        df_h['importancia_media'], errors='coerce'
    ).abs()
    df_h['epoca'] = pd.to_numeric(df_h['epoca'], errors='coerce')
    df_h = df_h.dropna(
        subset=['importancia_media', 'epoca', 'metodo_xai', 'feature']
    )
    df_h['metodo_xai'] = df_h['metodo_xai'].astype(str).str.upper().str.strip()
    df_h = normalizar_coluna_por_grupo(df_h, 'importancia_media', ['metodo_xai'])
    df_h = combinar_metodos_xai(
        df_h, ['feature', 'epoca'], ['importancia_media']
    )
    df_d4 = df_h

  # D6: projeção do método combinado
  df_d6 = _d6_combinado(df_d6, df_d3)

  return {'d2': df_d2, 'd3': df_d3, 'd4': df_d4, 'd5': df_d5, 'd6': df_d6}


def serie_estabilidade(df_d5, metodo_sel, metrica_sel, metodos_xai):
  """Série de estabilidade; para o método combinado, média entre os métodos
  (cada um normalizado individualmente quando a métrica não é Spearman)."""
  if df_d5.empty or 'tipo_metrica' not in df_d5.columns:
    return pd.DataFrame()

  if 'metrica' in df_d5.columns:
    mask_metrica = (
        df_d5['metrica'].astype(str).str.lower() == str(metrica_sel).lower()
    )
  else:
    mask_metrica = pd.Series(True, index=df_d5.index)

  def _prep(df_m):
    df_m = df_m.copy()
    if metrica_sel != 'spearman' and 'valor_metrica' in df_m.columns:
      s_min = df_m['valor_metrica'].min()
      s_max = df_m['valor_metrica'].max()
      if (s_max - s_min) > 1e-12:
        df_m['valor_metrica'] = (df_m['valor_metrica'] - s_min) / (s_max - s_min)
    return df_m

  tipo_upper = df_d5['tipo_metrica'].astype(str).str.upper()

  if metodo_sel == METODO_COMBINADO:
    alvo = {str(m).upper() for m in metodos_xai}
    df_m = df_d5[mask_metrica & tipo_upper.isin(alvo)]
    if df_m.empty or not {'epoca_destino', 'valor_metrica'}.issubset(df_m.columns):
      return pd.DataFrame()
    partes = [_prep(g) for _, g in df_m.groupby('tipo_metrica')]
    df_all = pd.concat(partes, ignore_index=True)
    return df_all.groupby('epoca_destino', as_index=False)['valor_metrica'].mean()

  df_m = df_d5[mask_metrica & (tipo_upper == str(metodo_sel).upper())]
  if df_m.empty:
    return pd.DataFrame()
  return _prep(df_m)


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

  # Método XAI combinado (após a normalização por método)
  df_d3 = combinar_metodos_xai(
      df_d3,
      ['epoca', 'instancia_id', 'feature'],
      ['importancia_absoluta', 'valor_atribuicao'],
  )

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


@lru_cache(maxsize=8)
def obter_sankey(dataset_sel):
  """Sankey processado (cacheado por dataset). Não modificar in-place."""
  df_d3_s, df_d2_s = carregar_dados_sankey(dataset_sel)
  return processar_dados_sankey(df_d3_s, df_d2_s, dataset_sel, normalizar=True)


# ==========================================
# 2. INICIALIZAÇÃO DA APLICAÇÃO DASH (LAYOUT)
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

ESTILO_HEADER = {
    'backgroundColor': '#f8fafc',
    'fontWeight': 'bold',
    'color': '#1e293b',
}
ESTILO_CARD = {'borderRadius': '8px', 'border': '1px solid #e2e8f0'}
ESTILO_LABEL = {'fontSize': '11px', 'fontWeight': 'bold', 'color': '#334155'}


def campo_dropdown(label, dd_id, options, value=None, mb='12px'):
  return html.Div([
      html.Label(label, style=ESTILO_LABEL),
      dcc.Dropdown(
          id=dd_id,
          options=options,
          value=value,
          clearable=False,
          style={'fontSize': '11px', 'marginBottom': mb},
      ),
  ])


def card_grafico(titulo, graph_id, altura, config=None, header_extra=None):
  if header_extra is None:
    header = dbc.CardHeader(titulo, style=ESTILO_HEADER)
  else:
    header = dbc.CardHeader(
        [html.Span(titulo), header_extra],
        className='d-flex justify-content-between align-items-center',
        style=ESTILO_HEADER,
    )
  kwargs = {'id': graph_id, 'style': {'height': altura}}
  if config:
    kwargs['config'] = config
  return dbc.Card(
      [header, dbc.CardBody([dcc.Graph(**kwargs)])],
      className='shadow-sm mb-4',
      style=ESTILO_CARD,
  )


def coluna(conteudo, lg=None):
  if lg:
    return dbc.Col([conteudo], width=12, lg=lg)
  return dbc.Col([conteudo], width=12)


CONFIG_SCATTER = {
    'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
    'displayModeBar': True,
    'scrollZoom': True,
    'doubleClick': 'reset',
}

BADGE_LIMPAR = dbc.Badge(
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
)

TABELA_INSTANCIAS = dash_table.DataTable(
    id='dim-tabela-instancias',
    columns=[
        {'name': 'Instância ID', 'id': 'instancia_id'},
        {'name': 'Época', 'id': 'epoca'},
        {'name': 'Método XAI', 'id': 'metodo_xai'},
        {'name': 'Classe Real', 'id': 'classe_real'},
        {'name': 'Classe Predita', 'id': 'classe_predita'},
        {'name': 'Erro Predição', 'id': 'erro_predicao'},
        {'name': 'Coord X', 'id': 'tsne_dim_1'},
        {'name': 'Coord Y', 'id': 'tsne_dim_2'},
    ],
    data=[],
    page_size=6,
    row_selectable='multi',
    selected_rows=[],
    style_table={'overflowX': 'auto'},
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

SIDEBAR = dbc.Col(
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
                campo_dropdown(
                    'Dataset:',
                    'dropdown-dataset',
                    [
                        {'label': 'Breast Cancer', 'value': 'breast_cancer'},
                        {'label': 'Wine Quality', 'value': 'wine_quality'},
                        {'label': 'German Credit', 'value': 'german_credit'},
                    ],
                    'breast_cancer',
                ),
                campo_dropdown('Método XAI:', 'dropdown-metodo', [], None),
                campo_dropdown(
                    'Métrica Estabilidade:',
                    'dropdown-metrica-estabilidade',
                    [
                        {'label': 'Spearman', 'value': 'spearman'},
                        {'label': 'MAD Absoluto', 'value': 'mad_importancia'},
                        {
                            'label': 'Variância',
                            'value': 'variancia_importancia',
                        },
                    ],
                    'spearman',
                ),
                campo_dropdown(
                    'Cores (Sankey):',
                    'dropdown-cor-metrica',
                    [
                        {'label': 'Classe Predita', 'value': 'classe'},
                        {'label': 'Confiança', 'value': 'confianca'},
                        {'label': 'Taxa de Erro', 'value': 'erro'},
                    ],
                    'classe',
                ),
                html.Hr(style={'borderColor': '#cbd5e1', 'margin': '10px 0'}),
                html.Div(
                    [
                        html.Label('Épocas (Sankey):', style=ESTILO_LABEL),
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
                html.Hr(style={'borderColor': '#cbd5e1', 'margin': '10px 0'}),
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
                campo_dropdown(
                    'Época 2D / Análise:', 'dim-dropdown-epoca', [], None, '10px'
                ),
                campo_dropdown(
                    'Algoritmo Espacial:',
                    'dim-dropdown-algo',
                    [
                        {'label': 't-SNE', 'value': 'TSNE'},
                        {'label': 'PCA', 'value': 'PCA'},
                    ],
                    'TSNE',
                    '10px',
                ),
                campo_dropdown(
                    'Colorir 2D Por:',
                    'dim-dropdown-cor',
                    [
                        {'label': 'Classe Predita', 'value': 'classe_predita'},
                        {'label': 'Classe Real', 'value': 'classe_real'},
                        {'label': 'Erro Predição', 'value': 'erro_predicao'},
                    ],
                    'classe_predita',
                    '10px',
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id='dim-toggle-trajetoria',
                            options=[{'label': ' Trajetória', 'value': 'ATIVO'}],
                            value=[],
                            style={'fontSize': '11px', 'color': '#334155'},
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
)

ABA_XAI = dbc.Tab(
    label='Explicabilidade Temporal',
    tab_id='tab-explicabilidade',
    children=[
        html.Div(
            [
                # 1. Sankey
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Fluxos Temporais de Transição de Classe (Sankey'
                            ' Reativo ao XAI)',
                            'sankey-graph',
                            '530px',
                        )
                    )
                ]),
                # 2. Heatmap (segue o "Método XAI" principal) e Estabilidade
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Resumo Global / Heatmap de Importância Temporal',
                            'heatmap-graph',
                            '500px',
                        ),
                        lg=7,
                    ),
                    coluna(
                        card_grafico(
                            'Estabilidade das Explicações (XAI)',
                            'stability-graph',
                            '500px',
                        ),
                        lg=5,
                    ),
                ]),
                # 3. Espaço 2D + Boxplot Temporal
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Redução de Dimensionalidade 2D (t-SNE / PCA)',
                            'dim-scatter-graph',
                            '460px',
                            config=CONFIG_SCATTER,
                            header_extra=BADGE_LIMPAR,
                        ),
                        lg=7,
                    ),
                    coluna(
                        card_grafico(
                            'Análise Auxiliar Temporal (Boxplot)',
                            'dim-reduction-aux-graph',
                            '460px',
                        ),
                        lg=5,
                    ),
                ]),
                # 4. Tabela Auxiliar de Instâncias
                dbc.Row([
                    coluna(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    'Tabela Auxiliar de Instâncias (Seleção por'
                                    ' Brushing) ',
                                    style=ESTILO_HEADER,
                                ),
                                dbc.CardBody([TABELA_INSTANCIAS]),
                            ],
                            className='shadow-sm mb-4',
                            style=ESTILO_CARD,
                        )
                    )
                ]),
            ],
            style={'paddingTop': '15px'},
        )
    ],
)

ABA_VALIDACAO = dbc.Tab(
    label='Validação e Desempenho do Modelo',
    tab_id='tab-validacao',
    children=[
        html.Div(
            [
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Evolução da Performance do Modelo (Acurácia &'
                            ' Loss)',
                            'performance-graph',
                            '380px',
                        )
                    )
                ]),
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Matriz de Confusão (Época Selecionada)',
                            'confusion-matrix-graph',
                            '360px',
                        ),
                        lg=6,
                    ),
                    coluna(
                        card_grafico(
                            'Métricas de Classificação (Precision, Recall, F1 &'
                            ' AUC)',
                            'class-metrics-graph',
                            '360px',
                        ),
                        lg=6,
                    ),
                ]),
                dbc.Row([
                    coluna(
                        card_grafico(
                            'Distribuição dos Erros de Predição Por Classe',
                            'error-distribution-graph',
                            '380px',
                        )
                    )
                ]),
            ],
            style={'paddingTop': '15px'},
        )
    ],
)

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
                SIDEBAR,
                dbc.Col(
                    [
                        dbc.Tabs(
                            [ABA_XAI, ABA_VALIDACAO],
                            id='tabs-main',
                            active_tab='tab-explicabilidade',
                        )
                    ],
                    width=10,
                ),
            ])
        ],
        fluid=True,
        style={
            'backgroundColor': '#f4f7f9',
            'padding': '20px',
            'minHeight': '100vh',
        },
    ),
])


# ==========================================
# 3. CALLBACKS DE ATUALIZAÇÃO
# ==========================================


@app.callback(
    [
        Output('dropdown-metodo', 'options'),
        Output('dropdown-metodo', 'value'),
        Output('dim-dropdown-epoca', 'options'),
        Output('dim-dropdown-epoca', 'value'),
        Output('slider-epocas', 'min'),
        Output('slider-epocas', 'max'),
        Output('slider-epocas', 'marks'),
        Output('slider-epocas', 'value'),
    ],
    [Input('dropdown-dataset', 'value')],
)
def update_options(dataset_sel):
  bundle = carregar_bundle(dataset_sel)
  df_d3 = bundle['d3']
  df_d6 = bundle['d6']

  metodos = ['SHAP', 'PFI', 'LIME', 'IG']
  if not df_d3.empty and 'metodo_xai' in df_d3.columns:
    metodos = sorted(
        str(m)
        for m in df_d3['metodo_xai'].dropna().unique()
        if str(m) != METODO_COMBINADO
    )

  epocas = [1, 10, 20, 30, 40]
  if not df_d6.empty and 'epoca' in df_d6.columns:
    epocas = sorted([int(x) for x in df_d6['epoca'].dropna().unique()])
  elif not df_d3.empty and 'epoca' in df_d3.columns:
    epocas = sorted([int(x) for x in df_d3['epoca'].dropna().unique()])

  options_metodo = [{'label': str(m).upper(), 'value': m} for m in metodos]
  options_metodo.append({'label': LABEL_COMBINADO, 'value': METODO_COMBINADO})
  val_metodo = metodos[0] if metodos else 'SHAP'

  opt_epocas = [{'label': f'Época {ep}', 'value': ep} for ep in epocas]
  val_epoca = epocas[0] if epocas else 1

  min_ep = min(epocas) if epocas else 1
  max_ep = max(epocas) if epocas else 40
  marks_ep = {int(ep): f'{ep}' for ep in epocas} if epocas else {1: '1'}

  return (
      options_metodo,
      val_metodo,
      opt_epocas,
      val_epoca,
      min_ep,
      max_ep,
      marks_ep,
      [min_ep, max_ep],
  )


# CALLBACK CENTRAL DE ESTADO
@app.callback(
    [
        Output('store-instancias-selecionadas', 'data'),
        Output('store-ultimo-click-box', 'data'),
    ],
    [
        Input('dim-scatter-graph', 'selectedData'),
        Input('dim-scatter-graph', 'clickData'),
        Input('sankey-graph', 'clickData'),
        Input('dim-reduction-aux-graph', 'clickData'),
        Input('dim-tabela-instancias', 'selected_rows'),
        Input('badge-limpar-sankey', 'n_clicks'),
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dim-dropdown-epoca', 'value'),
        Input('slider-epocas', 'value'),
    ],
    [
        State('store-instancias-selecionadas', 'data'),
        State('store-ultimo-click-box', 'data'),
        State('dim-tabela-instancias', 'data'),
    ],
)
def gerenciar_estado_selecao(
    selectedDataScatter,
    clickDataScatter,
    clickDataSankey,
    clickDataBoxplot,
    selected_rows_table,
    btn_badge_clicks,
    dataset_sel,
    metodo_sel,
    epoca_proj_sel,
    epocas_sel,
    state_selected_ids,
    state_last_box,
    table_data,
):
  ctx = callback_context
  triggered_id = (
      ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
  )

  if triggered_id in [
      'badge-limpar-sankey',
      'dropdown-dataset',
      'dropdown-metodo',
      'dim-dropdown-epoca',
  ]:
    return [], None

  if (
      triggered_id == 'dim-scatter-graph'
      and selectedDataScatter is None
      and clickDataScatter is None
  ):
    return [], None

  selected_ids = []

  # 1. CLIQUE NO BOXPLOT
  if (
      triggered_id == 'dim-reduction-aux-graph'
      and clickDataBoxplot
      and 'points' in clickDataBoxplot
      and len(clickDataBoxplot['points']) > 0
  ):
    p_box = clickDataBoxplot['points'][0]
    if 'customdata' in p_box and len(p_box['customdata']) > 0:
      inst_id = str(p_box['customdata'][0])
      if state_last_box == inst_id:
        return [], None
      else:
        return [inst_id], inst_id

  # 2. SELEÇÃO VIA SANKEY
  if (
      triggered_id == 'sankey-graph'
      and clickDataSankey
      and 'points' in clickDataSankey
      and len(clickDataSankey['points']) > 0
  ):
    pt_s = clickDataSankey['points'][0]
    if 'pointNumber' in pt_s:
      df_sankey_proc, _, _ = obter_sankey(dataset_sel)
      if not df_sankey_proc.empty:
        m_alvo = (
            metodo_sel
            if metodo_sel in df_sankey_proc['metodo_xai'].unique()
            else df_sankey_proc['metodo_xai'].iloc[0]
        )
        df_fil = df_sankey_proc[df_sankey_proc['metodo_xai'] == m_alvo]
        if epocas_sel:
          df_fil = df_fil[
              (df_fil['epoca_origem'] >= epocas_sel[0])
              & (df_fil['epoca_destino'] <= epocas_sel[1])
          ]

        if pt_s['pointNumber'] < len(df_fil):
          row_flow = df_fil.iloc[pt_s['pointNumber']]
          ids_sankey = [str(x) for x in row_flow['lista_instancias']]

          df_d6 = carregar_bundle(dataset_sel)['d6']
          if not df_d6.empty and 'classe_predita' in df_d6.columns:
            epoca_check = row_flow['epoca_origem']
            classe_origem_alvo = row_flow['classe_origem']

            df_d6_sub = df_d6[
                (
                    df_d6['metodo_xai'].astype(str).str.upper()
                    == str(m_alvo).upper()
                )
                & (df_d6['epoca'] == int(epoca_check))
                & (df_d6['classe_predita'] == int(classe_origem_alvo))
            ]
            valid_ids_set = set(df_d6_sub['instancia_id'].astype(str))
            ids_sankey = [i for i in ids_sankey if i in valid_ids_set]

          return ids_sankey, None

  # 3. SELEÇÃO VIA BRUSHING NO SPATIAL SCATTER (2D)
  if triggered_id == 'dim-scatter-graph':
    if (
        selectedDataScatter is not None
        and 'points' in selectedDataScatter
        and len(selectedDataScatter['points']) > 0
    ):
      for pt in selectedDataScatter['points']:
        if (
            'customdata' in pt
            and isinstance(pt['customdata'], list)
            and len(pt['customdata']) > 0
        ):
          selected_ids.append(str(pt['customdata'][0]))
      selected_ids = list(dict.fromkeys(selected_ids))
      return selected_ids, None
    elif (
        clickDataScatter
        and 'points' in clickDataScatter
        and len(clickDataScatter['points']) > 0
    ):
      p_click = clickDataScatter['points'][0]
      if 'customdata' in p_click and len(p_click['customdata']) > 0:
        inst_id = str(p_click['customdata'][0])
        if state_selected_ids == [inst_id]:
          return [], None
        return [inst_id], None

  # 4. SELEÇÃO VIA TABELA
  if (
      triggered_id == 'dim-tabela-instancias'
      and selected_rows_table
      and table_data
  ):
    for r_idx in selected_rows_table:
      if r_idx < len(table_data):
        selected_ids.append(str(table_data[r_idx]['instancia_id']))
    return selected_ids, None

  return state_selected_ids or [], state_last_box


# CALLBACK PRINCIPAL DOS GRÁFICOS
@app.callback(
    [
        Output('performance-graph', 'figure'),
        Output('stability-graph', 'figure'),
        Output('heatmap-graph', 'figure'),
        Output('dim-scatter-graph', 'figure'),
        Output('dim-reduction-aux-graph', 'figure'),
        Output('sankey-graph', 'figure'),
        Output('confusion-matrix-graph', 'figure'),
        Output('class-metrics-graph', 'figure'),
        Output('error-distribution-graph', 'figure'),
        Output('badge-limpar-sankey', 'children'),
        Output('badge-limpar-sankey', 'style'),
    ],
    [
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dropdown-metrica-estabilidade', 'value'),
        Input('dropdown-cor-metrica', 'value'),
        Input('slider-epocas', 'value'),
        Input('dim-dropdown-epoca', 'value'),
        Input('dim-dropdown-algo', 'value'),
        Input('dim-dropdown-cor', 'value'),
        Input('dim-toggle-trajetoria', 'value'),
        Input('store-instancias-selecionadas', 'data'),
        Input('sankey-graph', 'clickData'),
        Input('badge-limpar-sankey', 'n_clicks'),
    ],
)
def update_all_graphs(
    dataset_sel,
    metodo_sel,
    metrica_est_sel,
    cor_sankey_sel,
    epocas_sel,
    epoca_proj_sel,
    algo_sel,
    cor_proj_sel,
    trajetoria_sel,
    instancias_selecionadas_store,
    clickDataSankey,
    btn_badge_clicks,
):
  bundle = carregar_bundle(dataset_sel)
  df_d2 = bundle['d2']
  df_d3 = bundle['d3']  # já normalizado por método + método combinado
  df_d4 = bundle['d4']  # já normalizado por método + método combinado
  df_d5 = bundle['d5']
  df_d6 = bundle['d6']  # inclui o método combinado
  # Versão sem o método combinado (evita duplicar contagens nas métricas)
  df_d6_orig = (
      df_d6[df_d6['metodo_xai'].astype(str) != METODO_COMBINADO]
      if (not df_d6.empty and 'metodo_xai' in df_d6.columns)
      else df_d6
  )

  label_ds = MAPA_DATASET.get(dataset_sel, {}).get('label', dataset_sel.upper())
  map_cls = Mapeamento_classes_dataset(dataset_sel)

  is_comb = metodo_sel == METODO_COMBINADO
  metodo_label = LABEL_COMBINADO if is_comb else str(metodo_sel)

  if not df_d3.empty and 'metodo_xai' in df_d3.columns:
    metodos_xai = [
        str(m) for m in df_d3['metodo_xai'].dropna().unique()
        if str(m) != METODO_COMBINADO
    ]
  else:
    metodos_xai = []

  ctx = callback_context
  triggered_id = (
      ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
  )

  # ----------------------------------------------------
  # 1. GRAPH 1: PERFORMANCE DO MODELO
  # ----------------------------------------------------
  fig_perf = go.Figure()
  if not df_d2.empty:
    acc_col = (
        'accuracy'
        if 'accuracy' in df_d2.columns
        else (df_d2.columns[1] if len(df_d2.columns) > 1 else None)
    )
    loss_col = 'loss_validacao' if 'loss_validacao' in df_d2.columns else None

    if acc_col and 'epoca' in df_d2.columns:
      fig_perf.add_trace(
          go.Scatter(
              x=df_d2['epoca'],
              y=df_d2[acc_col],
              mode='lines+markers',
              name='Acurácia Validação',
              line=dict(color='#059669', width=2.5),
          )
      )
    if loss_col and 'epoca' in df_d2.columns:
      fig_perf.add_trace(
          go.Scatter(
              x=df_d2['epoca'],
              y=df_d2[loss_col],
              mode='lines+markers',
              name='Loss Validação',
              yaxis='y2',
              line=dict(color='#dc2626', width=2.5),
          )
      )

  fig_perf.update_layout(
      title=f'Desempenho Preditivo do Modelo - {label_ds}',
      xaxis=dict(title='Época'),
      yaxis=dict(title='Acurácia'),
      yaxis2=dict(
          title='Loss (Perda)', overlaying='y', side='right', showgrid=False
      ),
      margin=dict(t=40, b=40, l=40, r=40),
      legend=dict(x=0.01, y=0.99, orientation='h'),
      template='plotly_white',
  )

  # ----------------------------------------------------
  # 2. GRAPH 2: ESTABILIDADE DAS EXPLICAÇÕES
  # ----------------------------------------------------
  fig_stab = go.Figure()
  df_s_sub = serie_estabilidade(df_d5, metodo_sel, metrica_est_sel, metodos_xai)
  if (
      not df_s_sub.empty
      and 'epoca_destino' in df_s_sub.columns
      and 'valor_metrica' in df_s_sub.columns
  ):
    fig_stab.add_trace(
        go.Scatter(
            x=df_s_sub['epoca_destino'],
            y=df_s_sub['valor_metrica'],
            mode='lines+markers',
            name=f'Estabilidade ({metrica_est_sel.upper()})',
            line=dict(color='#2563eb', width=2.5, dash='solid'),
        )
    )
  fig_stab.update_layout(
      title=(
          'Consistência Temporal'
          f' ({"MÉDIA GERAL" if is_comb else metodo_sel} -'
          f' {metrica_est_sel.upper()})'
      ),
      xaxis=dict(title='Época'),
      yaxis=dict(title=f'Índice ({metrica_est_sel})'),
      margin=dict(t=40, b=40, l=40, r=40),
      legend=dict(x=0.01, y=0.99, orientation='h'),
      template='plotly_white',
  )

  # ----------------------------------------------------
  # 3. HEATMAP DE IMPORTÂNCIA (segue o Método XAI principal)
  # ----------------------------------------------------
  fig_heat = go.Figure()
  heat_label = 'MÉDIA GERAL' if is_comb else str(metodo_sel).upper()

  if not df_d4.empty and 'importancia_media' in df_d4.columns:
    df_h_sub = df_d4[
        df_d4['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()
    ]

    if not df_h_sub.empty:
      piv = df_h_sub.pivot_table(
          index='feature',
          columns='epoca',
          values='importancia_media',
          aggfunc='mean',
          fill_value=0,
      )

      if not piv.empty:
        piv = piv.reindex(sorted(piv.columns), axis=1)
        ultima_coluna = piv.columns[-1]
        ordem_ranking = piv[ultima_coluna].sort_values(ascending=True).index
        piv = piv.reindex(ordem_ranking)

        df_ranks = piv.rank(ascending=False, method='min', axis=0)
        num_features = len(piv)

        custom_text = np.empty((piv.shape[0], piv.shape[1], 2), dtype=object)
        cols_list = list(piv.columns)
        for i_feat, feat_name in enumerate(piv.index):
          for j_ep, ep_val in enumerate(cols_list):
            rank_atual = int(df_ranks.loc[feat_name, ep_val])
            if j_ep > 0:
              ep_ant = cols_list[j_ep - 1]
              rank_ant = int(df_ranks.loc[feat_name, ep_ant])
              diff = rank_ant - rank_atual
              if diff > 0:
                var_str = f'▲ Subiu {diff} posição(ões)'
              elif diff < 0:
                var_str = f'▼ Caiu {abs(diff)} posição(ões)'
              else:
                var_str = '▶ Estável'
            else:
              var_str = '—'

            custom_text[i_feat, j_ep, 0] = f'{rank_atual}º de {num_features}'
            custom_text[i_feat, j_ep, 1] = var_str

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=piv.values,
                x=piv.columns,
                y=piv.index,
                zmin=0.0,
                zmax=1.0,
                customdata=custom_text,
                colorscale=COLORSCALE_REDS_20,
                colorbar=dict(title='Importância'),
                hovertemplate=(
                    '<b>Época:</b> %{x}<br><b>Feature:</b> %{y}<br>'
                    '<b>Importância XAI:</b> %{z:.4f}'
                    '<br>----------------------------------<br><b>Ranking:</b>'
                    ' %{customdata[0]}<br><b>Evolução:</b>'
                    ' %{customdata[1]}<extra></extra>'
                ),
            )
        )
        fig_heat.update_layout(
            title=f'Heatmap de Importância ({heat_label})',
            xaxis=dict(title='Época'),
            yaxis=dict(dtick=1, automargin=True, tickfont=dict(size=10)),
            margin=dict(t=40, b=30, l=150, r=20),
            template='plotly_white',
        )
  if not fig_heat.data:
    fig_heat.update_layout(
        title=f'Heatmap de Importância ({heat_label})',
        annotations=[dict(
            text='Não há dados suficientes para o método selecionado.',
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white',
    )

  instancias_selecionadas_tabela = [
      str(x) for x in (instancias_selecionadas_store or [])
  ]
  instancias_destaque_sankey = []

  df_sankey_processed, _, _ = obter_sankey(dataset_sel)

  if (
      triggered_id != 'badge-limpar-sankey'
      and not instancias_selecionadas_tabela
      and clickDataSankey
      and 'points' in clickDataSankey
      and len(clickDataSankey['points']) > 0
  ):
    pt_s = clickDataSankey['points'][0]
    if 'pointNumber' in pt_s and not df_sankey_processed.empty:
      flow_idx = pt_s['pointNumber']
      m_alvo = (
          metodo_sel
          if metodo_sel in df_sankey_processed['metodo_xai'].unique()
          else df_sankey_processed['metodo_xai'].iloc[0]
      )
      df_filtered_s = df_sankey_processed[
          df_sankey_processed['metodo_xai'] == m_alvo
      ].copy()
      if epocas_sel:
        df_filtered_s = df_filtered_s[
            (df_filtered_s['epoca_origem'] >= epocas_sel[0])
            & (df_filtered_s['epoca_destino'] <= epocas_sel[1])
        ]
      if flow_idx < len(df_filtered_s):
        row_flow = df_filtered_s.iloc[flow_idx]
        lista_raw = row_flow['lista_instancias']
        instancias_destaque_sankey = [str(x) for x in lista_raw]

        if not df_d6.empty and 'classe_predita' in df_d6.columns:
          df_d6_ep = df_d6[
              (
                  df_d6['metodo_xai'].astype(str).str.upper()
                  == str(m_alvo).upper()
              )
              & (df_d6['epoca'] == int(row_flow['epoca_origem']))
              & (df_d6['classe_predita'] == int(row_flow['classe_origem']))
          ]
          valid_ids_set = set(df_d6_ep['instancia_id'].astype(str))
          instancias_destaque_sankey = [
              i for i in instancias_destaque_sankey if i in valid_ids_set
          ]

  if instancias_destaque_sankey and triggered_id != 'badge-limpar-sankey':
    badge_children = [
        f'Instâncias do Fluxo Sankey ({len(instancias_destaque_sankey)}) ',
        html.Span('✕', style={'fontWeight': 'bold', 'marginLeft': '4px'}),
    ]
    badge_style = {
        'cursor': 'pointer',
        'fontSize': '11px',
        'padding': '5px 9px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'display': 'inline-block',
    }
  else:
    badge_children = []
    badge_style = {'display': 'none'}

  # ----------------------------------------------------
  # 4. REDUÇÃO DE DIMENSIONALIDADE (t-SNE / PCA)
  # ----------------------------------------------------
  fig_tsne = go.Figure()
  if not df_d6.empty:
    epoca_alvo = (
        epoca_proj_sel
        if epoca_proj_sel
        else (df_d6['epoca'].min() if 'epoca' in df_d6.columns else 1)
    )
    df_sub = df_d6[
        (df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper())
        & (df_d6['epoca'] == int(epoca_alvo))
    ].copy()

    # No método combinado, a projeção "t-SNE" é um PCA conjunto
    algo_label = 'PCA conjunto' if (is_comb and algo_sel != 'PCA') else algo_sel

    if not df_sub.empty:
      df_sub['instancia_id'] = df_sub['instancia_id'].astype(str)
      if algo_sel == 'PCA' and not df_d3.empty:
        df_exp_ep = df_d3[
            (
                df_d3['metodo_xai'].astype(str).str.upper()
                == str(metodo_sel).upper()
            )
            & (df_d3['epoca'] == int(epoca_alvo))
        ].copy()
        if not df_exp_ep.empty:
          df_exp_ep['instancia_id'] = df_exp_ep['instancia_id'].astype(str)
          pivot_pca = df_exp_ep.pivot_table(
              index='instancia_id', columns='feature', values='valor_atribuicao'
          ).fillna(0)
          pca = PCA(n_components=2)
          res_pca = pca.fit_transform(pivot_pca.values)
          pca_df = pd.DataFrame(
              res_pca, index=pivot_pca.index, columns=['dim_1', 'dim_2']
          ).reset_index()
          df_sub = df_sub.merge(pca_df, on='instancia_id', how='left')
          x_col, y_col = 'dim_1', 'dim_2'
        else:
          x_col, y_col = 'tsne_dim_1', 'tsne_dim_2'
      else:
        x_col, y_col = 'tsne_dim_1', 'tsne_dim_2'

      if cor_proj_sel == 'erro_predicao':
        df_sub['cor_label'] = df_sub['erro_predicao'].map(
            {0: 'Acerto', 1: 'Erro'}
        )
        fig_tsne = px.scatter(
            df_sub,
            x=x_col,
            y=y_col,
            color='cor_label',
            color_discrete_map={'Acerto': '#059669', 'Erro': '#dc2626'},
            custom_data=[
                'instancia_id',
                'classe_real',
                'classe_predita',
                'erro_predicao',
            ],
        )
      else:
        fig_tsne = px.scatter(
            df_sub,
            x=x_col,
            y=y_col,
            color=cor_proj_sel,
            color_continuous_scale=COLORSCALE_BLUES_20,
            custom_data=[
                'instancia_id',
                'classe_real',
                'classe_predita',
                'erro_predicao',
            ],
        )
        fig_tsne.update_coloraxes(
            colorbar=dict(tickvals=[0, 1], ticktext=['0', '1'], len=0.8)
        )

      fig_tsne.update_traces(
          marker=dict(
              size=8, opacity=0.85, line=dict(width=0.5, color='#334155')
          ),
          hovertemplate=(
              '<b>Instância ID:</b> #%{customdata[0]}<br><b>Classe Real:</b>'
              ' %{customdata[1]}<br><b>Classe Predita:</b>'
              ' %{customdata[2]}<br><b>Erro:</b> %{customdata[3]}<br><b>X:</b>'
              ' %{x:.3f} | <b>Y:</b> %{y:.3f}<extra></extra>'
          ),
      )

      if instancias_destaque_sankey and triggered_id != 'badge-limpar-sankey':
        df_sankey_pts = df_sub[
            df_sub['instancia_id'].isin(instancias_destaque_sankey)
        ]
        if not df_sankey_pts.empty:
          fig_tsne.add_trace(
              go.Scatter(
                  x=df_sankey_pts[x_col],
                  y=df_sankey_pts[y_col],
                  mode='markers',
                  marker=dict(
                      size=14,
                      color='rgba(0,0,0,0)',
                      line=dict(color='#ec4899', width=2.5),
                  ),
                  showlegend=False,
                  hoverinfo='skip',
              )
          )

      if (
          'ATIVO' in trajetoria_sel
          and len(instancias_selecionadas_tabela) == 1
      ):
        inst_id_traj = instancias_selecionadas_tabela[0]
        df_traj = df_d6[
            (
                df_d6['metodo_xai'].astype(str).str.upper()
                == str(metodo_sel).upper()
            )
            & (df_d6['instancia_id'].astype(str) == inst_id_traj)
        ].sort_values('epoca')
        fig_tsne.add_trace(
            go.Scatter(
                x=df_traj['tsne_dim_1'],
                y=df_traj['tsne_dim_2'],
                mode='lines+markers+text',
                text=[f'E{int(ep)}' for ep in df_traj['epoca']],
                textposition='top center',
                line=dict(color='#f59e0b', width=3, dash='dash'),
                marker=dict(size=12, color='#f59e0b', symbol='diamond'),
                name=f'Trajetória ID #{inst_id_traj}',
            )
        )

      if instancias_selecionadas_tabela:
        pontos_clicados = df_sub[
            df_sub['instancia_id'].isin(instancias_selecionadas_tabela)
        ]
        if not pontos_clicados.empty:
          fig_tsne.add_trace(
              go.Scatter(
                  x=pontos_clicados[x_col],
                  y=pontos_clicados[y_col],
                  mode='markers',
                  marker=dict(
                      size=18,
                      color='rgba(0,0,0,0)',
                      line=dict(color='#eab308', width=3),
                  ),
                  showlegend=False,
                  hoverinfo='skip',
              )
          )

      fig_tsne.update_layout(
          title=(
              f'Espaço 2D ({algo_label}) - Época {epoca_alvo}'
              f' ({"MÉDIA GERAL" if is_comb else metodo_sel}) - {label_ds}'
          ),
          xaxis=dict(showticklabels=False, title=''),
          yaxis=dict(showticklabels=False, title=''),
          margin=dict(l=20, r=20, t=40, b=20),
          template='plotly_white',
          dragmode='select',
          clickmode='event+select',
      )

  # ----------------------------------------------------
  # 5. BOXPLOT AUXILIAR TEMPORAL
  # ----------------------------------------------------
  fig_dim_aux = go.Figure()
  if not df_d6.empty and 'tsne_dim_1' in df_d6.columns:
    df_aux_sub = df_d6[
        df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper()
    ].copy()
    if not df_aux_sub.empty and 'epoca' in df_aux_sub.columns:
      df_aux_sub['instancia_id'] = df_aux_sub['instancia_id'].astype(str)
      custom_box_data = np.array(df_aux_sub['instancia_id']).reshape(-1, 1)
      nome_dim = 'PCA conjunto Dim 1' if is_comb else 't-SNE Dim 1'

      fig_dim_aux = go.Figure(
          data=go.Box(
              x=df_aux_sub['epoca'],
              y=df_aux_sub['tsne_dim_1'],
              customdata=custom_box_data,
              name=nome_dim,
              marker_color='#6366f1',
              boxpoints='outliers',
              hovertemplate=(
                  '<b>Instância ID:</b> #%{customdata[0]}<br><b>Época:</b>'
                  f' %{{x}}<br><b>{nome_dim}:</b>'
                  ' %{y:.4f}<extra></extra>'
              ),
          )
      )
      fig_dim_aux.update_layout(
          title=f'Dispersão Temporal ({nome_dim})',
          margin=dict(l=20, r=20, t=30, b=20),
          template='plotly_white',
      )

  # ----------------------------------------------------
  # 6. SANKEY TEMPORAL DE EXPLICAÇÕES
  # ----------------------------------------------------
  fig_sankey = go.Figure()
  if not df_sankey_processed.empty:
    metodos_validos = df_sankey_processed['metodo_xai'].unique()
    m_alvo = (
        metodo_sel
        if metodo_sel in metodos_validos
        else metodos_validos[0]
    )
    df_filtered = df_sankey_processed[
        df_sankey_processed['metodo_xai'] == m_alvo
    ].copy()

    if epocas_sel:
      df_filtered = df_filtered[
          (df_filtered['epoca_origem'] >= epocas_sel[0])
          & (df_filtered['epoca_destino'] <= epocas_sel[1])
      ]

    if not df_filtered.empty:
      epocas_unicas = sorted(
          list(
              set(df_filtered['epoca_origem']).union(
                  set(df_filtered['epoca_destino'])
              )
          )
      )
      classes_unicas = [0, 1]

      node_labels, node_x, node_y = [], [], []
      node_dict = {}

      for i_ep, ep in enumerate(epocas_unicas):
        for i_cl, cl in enumerate(classes_unicas):
          node_id = len(node_labels)
          node_dict[f'E{ep}_C{cl}'] = node_id
          nome_classe = map_cls.get(cl, f'Classe {cl}')
          node_labels.append(f'Época {ep}<br>{nome_classe}')
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
            base_rgb = (
                '37, 99, 235' if row['classe_origem'] == 0 else '249, 115, 22'
            )
          elif cor_sankey_sel == 'confianca':
            base_rgb = f'{int(255*(1-conf))}, {int(200*conf)}, 150'
          else:
            err = float(row['taxa_erro'])
            base_rgb = f'{int(255*err)}, 50, 50'

          link_colors.append(f'rgba({base_rgb}, {alpha:.2f})')

          hover_text = (
              f'<b>Dataset:</b> {dataset_sel.upper()}<br><b>Método XAI:</b>'
              f' {m_alvo}<br><b>Transição:</b> Época {row["epoca_origem"]}'
              f' &rarr; Época {row["epoca_destino"]}<br><b>Classe Predita:</b>'
              f' {map_cls.get(row["classe_origem"])} &rarr;'
              f' {map_cls.get(row["classe_destino"])}<br><b>Classe Real'
              ' Dominante:</b>'
              f' {map_cls.get(row["classe_real_dominante"])}<br><b>Qtd'
              f' Instâncias:</b> {row["qtd_instancias"]}<br><b>Exemplo ID:</b>'
              f' #{row["exemplo_instancia"]}<br><b>Confiança Média:</b>'
              f' {conf*100:.1f}%<br><b>Taxa Erro Média:</b>'
              f' {row["taxa_erro"]*100:.1f}%<br><b>Importância Média XAI'
              f':</b> {row["importancia_media"]:.4f}'
          )
          hover_texts.append(hover_text)

      fig_sankey = go.Figure(
          data=[
              go.Sankey(
                  arrangement='snap',
                  node=dict(
                      pad=30,
                      thickness=20,
                      line=dict(color='#334155', width=0.5),
                      label=node_labels,
                      x=node_x,
                      y=node_y,
                      color='#e2e8f0',
                  ),
                  link=dict(
                      source=sources,
                      target=targets,
                      value=values,
                      color=link_colors,
                      customdata=hover_texts,
                      hovertemplate='%{customdata}<extra></extra>',
                  ),
              )
          ]
      )
      fig_sankey.update_layout(
          title_text=(
              'Transições Temporais de Explicações'
              f' ({"MÉDIA GERAL" if m_alvo == METODO_COMBINADO else m_alvo})'
              f' - {label_ds}'
          ),
          font_size=12,
          margin=dict(l=20, r=20, t=50, b=20),
          template='plotly_white',
      )

  # ----------------------------------------------------
  # 7. MATRIZ DE CONFUSÃO
  # ----------------------------------------------------
  fig_cm = go.Figure()
  epoca_alvo_val = epoca_proj_sel if epoca_proj_sel else 1
  if not df_d6_orig.empty:
    df_ep_val = df_d6_orig[df_d6_orig['epoca'] == int(epoca_alvo_val)].copy()
    if (
        not df_ep_val.empty
        and 'classe_real' in df_ep_val.columns
        and 'classe_predita' in df_ep_val.columns
    ):
      y_true = df_ep_val['classe_real'].astype(int)
      y_pred = df_ep_val['classe_predita'].astype(int)
      cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

      labels_cls = [map_cls.get(0, 'Classe 0'), map_cls.get(1, 'Classe 1')]

      fig_cm = px.imshow(
          cm,
          text_auto=True,
          x=labels_cls,
          y=labels_cls,
          color_continuous_scale='Blues',
          labels=dict(x='Classe Predita', y='Classe Real', color='Frequência'),
      )
      fig_cm.update_layout(
          title=f'Matriz de Confusão (Época {epoca_alvo_val})',
          margin=dict(t=40, b=30, l=60, r=20),
          template='plotly_white',
      )

  # ----------------------------------------------------
  # 8. MÉTRICAS DE CLASSIFICAÇÃO
  # ----------------------------------------------------
  fig_class_metrics = go.Figure()
  if not df_d6_orig.empty:
    df_ep_val = df_d6_orig[df_d6_orig['epoca'] == int(epoca_alvo_val)].copy()
    if (
        not df_ep_val.empty
        and 'classe_real' in df_ep_val.columns
        and 'classe_predita' in df_ep_val.columns
    ):
      y_true = df_ep_val['classe_real'].astype(int)
      y_pred = df_ep_val['classe_predita'].astype(int)

      p, r, f1, _ = precision_recall_fscore_support(
          y_true, y_pred, average=None, labels=[0, 1], zero_division=0
      )
      try:
        auc = roc_auc_score(y_true, y_pred)
      except Exception:
        auc = 0.5

      categories = ['Precision', 'Recall', 'F1-Score']
      fig_class_metrics.add_trace(
          go.Bar(
              name=map_cls.get(0, 'Classe 0'),
              x=categories,
              y=[p[0], r[0], f1[0]],
              marker_color='#3b82f6',
          )
      )
      fig_class_metrics.add_trace(
          go.Bar(
              name=map_cls.get(1, 'Classe 1'),
              x=categories,
              y=[p[1], r[1], f1[1]],
              marker_color='#f97316',
          )
      )

      fig_class_metrics.update_layout(
          title=(
              f'Métricas por Classe (Época {epoca_alvo_val}) | AUC-ROC:'
              f' {auc:.3f}'
          ),
          barmode='group',
          yaxis=dict(range=[0, 1.05], title='Score'),
          margin=dict(t=40, b=30, l=40, r=20),
          template='plotly_white',
      )

  # ----------------------------------------------------
  # 9. DISTRIBUIÇÃO DOS ERROS DE PREDIÇÃO
  # ----------------------------------------------------
  fig_err_dist = go.Figure()
  if not df_d6_orig.empty and 'erro_predicao' in df_d6_orig.columns:
    df_d6_copy = df_d6_orig.copy()
    df_d6_copy['status_pred'] = df_d6_copy['erro_predicao'].map(
        {0: 'Acerto', 1: 'Erro'}
    )

    df_err_summary = (
        df_d6_copy.groupby(['epoca', 'classe_real', 'status_pred'])
        .size()
        .reset_index(name='contagem')
    )
    df_err_summary['nome_classe'] = df_err_summary['classe_real'].map(map_cls)

    fig_err_dist = px.bar(
        df_err_summary,
        x='epoca',
        y='contagem',
        color='status_pred',
        facet_col='nome_classe',
        color_discrete_map={'Acerto': '#059669', 'Erro': '#dc2626'},
        labels={
            'contagem': 'Qtd Instâncias',
            'epoca': 'Época',
            'status_pred': 'Resultado',
        },
    )
    fig_err_dist.update_layout(
        title='Evolução da Distribuição de Erros x Acertos por Classe Real',
        margin=dict(t=40, b=30, l=40, r=20),
        template='plotly_white',
    )

  return (
      fig_perf,
      fig_stab,
      fig_heat,
      fig_tsne,
      fig_dim_aux,
      fig_sankey,
      fig_cm,
      fig_class_metrics,
      fig_err_dist,
      badge_children,
      badge_style,
  )


# CALLBACK TABELA
@app.callback(
    Output('dim-tabela-instancias', 'data'),
    [
        Input('store-instancias-selecionadas', 'data'),
        Input('dropdown-dataset', 'value'),
        Input('dropdown-metodo', 'value'),
        Input('dim-dropdown-epoca', 'value'),
        Input('dim-dropdown-algo', 'value'),
    ],
)
def update_table_from_store(
    instancias_selecionadas, dataset_sel, metodo_sel, epoca_proj_sel, algo_sel
):
  bundle = carregar_bundle(dataset_sel)
  df_d6 = bundle['d6']
  df_d3 = bundle['d3']
  if df_d6.empty:
    return []

  epoca_alvo = (
      epoca_proj_sel
      if epoca_proj_sel
      else (df_d6['epoca'].min() if 'epoca' in df_d6.columns else 1)
  )
  df_sub = df_d6[
      (df_d6['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper())
      & (pd.to_numeric(df_d6['epoca'], errors='coerce') == int(epoca_alvo))
  ].copy()

  if df_sub.empty:
    return []

  df_sub['instancia_id'] = df_sub['instancia_id'].astype(str)

  if algo_sel == 'PCA' and not df_d3.empty:
    df_exp = df_d3[
        (df_d3['metodo_xai'].astype(str).str.upper() == str(metodo_sel).upper())
        & (pd.to_numeric(df_d3['epoca'], errors='coerce') == int(epoca_alvo))
    ].copy()
    if not df_exp.empty:
      df_exp['instancia_id'] = df_exp['instancia_id'].astype(str)
      pivot = df_exp.pivot_table(
          index='instancia_id',
          columns='feature',
          values='valor_atribuicao',
          fill_value=0,
      )
      if len(pivot) >= 2 and pivot.shape[1] >= 2:
        pca = PCA(n_components=2)
        resultado = pca.fit_transform(pivot.values)
        pca_df = pd.DataFrame(resultado, columns=['dim_1', 'dim_2'])
        pca_df['instancia_id'] = pivot.index.astype(str)
        df_sub = df_sub.merge(pca_df, on='instancia_id', how='inner')

  if instancias_selecionadas and len(instancias_selecionadas) > 0:
    inst_str_list = [str(x) for x in instancias_selecionadas]
    df_tabela = df_sub[df_sub['instancia_id'].isin(inst_str_list)]
  else:
    df_tabela = df_sub

  return df_tabela.to_dict('records')


if __name__ == '__main__':
  app.run(debug=False)
