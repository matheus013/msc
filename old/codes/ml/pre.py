import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np
import os

# Leitura dos dados
df = pd.read_parquet('data/clean/vendas_uf/uf=MA')
df = df[['produto_cod', 'revendedor_cod', 'venda_data', 'venda_qtd',
         'venda_vlr_receita_liquida', 'venda_vlr_receita_bruta',
         'venda_ciclo', 'genero', 'idade', 'segmento',
         'status', 'ciclos_inativos']]

# Nomes amigáveis para colunas
colunas_legiveis = {
    'venda_qtd': 'Quantidade Vendida',
    'venda_vlr_receita_liquida': 'Receita Líquida',
    'venda_vlr_receita_bruta': 'Receita Bruta',
    'idade': 'Idade',
    'ciclos_inativos': 'Ciclos Inativos',
    'genero': 'Gênero',
    'segmento': 'Segmento',
    'status': 'Status',
    'venda_ciclo': 'Ciclo de Venda'
}

# Conversões de tipos
df['ciclos_inativos_num'] = df['ciclos_inativos'].str.extract(r'(\d+)')
df['ciclos_inativos'] = pd.to_numeric(df['ciclos_inativos_num'], errors='coerce')
df['venda_qtd'] = pd.to_numeric(df['venda_qtd'], errors='coerce')
df['venda_vlr_receita_liquida'] = pd.to_numeric(df['venda_vlr_receita_liquida'], errors='coerce')
df['venda_vlr_receita_bruta'] = pd.to_numeric(df['venda_vlr_receita_bruta'], errors='coerce')
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')

# Identificar os top 6 ciclos com maior receita líquida
top_ciclos = df.groupby('venda_ciclo')['venda_vlr_receita_liquida'].sum().nlargest(6).index
df_top_ciclos = df[df['venda_ciclo'].isin(top_ciclos)]

# Criar pasta de saída
os.makedirs("plots", exist_ok=True)

def salvar(fig, nome):
    fig.write_image(f"plots/{nome}.png", width=1000, height=600, scale=2)

# Gráfico 1: Histograma de Quantidade Vendida
fig = px.histogram(df, x='venda_qtd', title=f'Distribuição de {colunas_legiveis["venda_qtd"]}', marginal="box")
fig.update_xaxes(title=colunas_legiveis["venda_qtd"])
salvar(fig, "distribuicao_quantidade_vendida")

# Gráfico 2: Boxplot de Receita Líquida
fig = px.box(df, y='venda_vlr_receita_liquida', title=f'Boxplot da {colunas_legiveis["venda_vlr_receita_liquida"]}')
fig.update_yaxes(title=colunas_legiveis["venda_vlr_receita_liquida"])
salvar(fig, "boxplot_receita_liquida")

# Gráfico 3: Receita por Segmento
agg = df.groupby('segmento')['venda_vlr_receita_liquida'].sum().reset_index()
fig = px.bar(agg, x='segmento', y='venda_vlr_receita_liquida',
             title=f'{colunas_legiveis["venda_vlr_receita_liquida"]} por {colunas_legiveis["segmento"]}')
fig.update_xaxes(title=colunas_legiveis["segmento"])
fig.update_yaxes(title=colunas_legiveis["venda_vlr_receita_liquida"])
salvar(fig, "receita_por_segmento")

# Gráfico 4: Distribuição de Idade
fig = px.histogram(df, x='idade', title=f'Distribuição de {colunas_legiveis["idade"]}', marginal="box")
fig.update_xaxes(title=colunas_legiveis["idade"])
salvar(fig, "distribuicao_idade")

# Gráfico 5: Pizza de Gênero
genero_counts = df['genero'].value_counts().reset_index()
genero_counts.columns = ['genero', 'count']
fig = px.pie(genero_counts, values='count', names='genero', title=f'Distribuição de {colunas_legiveis["genero"]}')
salvar(fig, "distribuicao_genero")

# Gráfico 6: Idade por Ciclo (Top 6)
fig = px.box(df_top_ciclos, x='venda_ciclo', y='idade',
             title=f'{colunas_legiveis["idade"]} por {colunas_legiveis["venda_ciclo"]} (Top 6)')
fig.update_xaxes(title=colunas_legiveis["venda_ciclo"])
fig.update_yaxes(title=colunas_legiveis["idade"])
salvar(fig, "idade_por_top_ciclos")

# 7. Distribuição por Segmento
fig = px.histogram(df, x='segmento', title=f'Distribuição por {colunas_legiveis["segmento"]}')
fig.update_xaxes(title=colunas_legiveis["segmento"])
salvar(fig, "distribuicao_por_segmento")

# 8. Distribuição por Status
fig = px.histogram(df, x='status', title=f'Distribuição por {colunas_legiveis["status"]}')
fig.update_xaxes(title=colunas_legiveis["status"])
salvar(fig, "distribuicao_por_status")

# 9. Média de Ciclos Inativos por Status
agg = df.groupby('status')['ciclos_inativos'].mean().reset_index()
fig = px.bar(agg, x='status', y='ciclos_inativos',
             title=f'Média de {colunas_legiveis["ciclos_inativos"]} por {colunas_legiveis["status"]}')
fig.update_xaxes(title=colunas_legiveis["status"])
fig.update_yaxes(title=colunas_legiveis["ciclos_inativos"])
salvar(fig, "media_ciclos_inativos_por_status")

# 10. Evolução por Ciclo (Top 6)
df_grouped = df[df['venda_ciclo'].isin(top_ciclos)].groupby('venda_ciclo')[
    ['venda_qtd', 'venda_vlr_receita_liquida']].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_grouped['venda_ciclo'], y=df_grouped['venda_qtd'],
                         mode='lines+markers', name=colunas_legiveis["venda_qtd"]))
fig.add_trace(go.Scatter(x=df_grouped['venda_ciclo'], y=df_grouped['venda_vlr_receita_liquida'],
                         mode='lines+markers', name=colunas_legiveis["venda_vlr_receita_liquida"]))
fig.update_layout(title=f'Evolução por {colunas_legiveis["venda_ciclo"]} (Top 6)',
                  xaxis_title=colunas_legiveis["venda_ciclo"], yaxis_title="Valor")
salvar(fig, "evolucao_top_ciclos")

# 11. Receita Média por Gênero
agg = df.groupby('genero')['venda_vlr_receita_liquida'].mean().reset_index()
fig = px.bar(agg, x='genero', y='venda_vlr_receita_liquida',
             title=f'{colunas_legiveis["venda_vlr_receita_liquida"]} Média por {colunas_legiveis["genero"]}')
fig.update_xaxes(title=colunas_legiveis["genero"])
fig.update_yaxes(title=colunas_legiveis["venda_vlr_receita_liquida"])
salvar(fig, "receita_media_por_genero")

# 12. Ciclos Inativos por Segmento
fig = px.box(df, x='segmento', y='ciclos_inativos',
             title=f'{colunas_legiveis["ciclos_inativos"]} por {colunas_legiveis["segmento"]}')
fig.update_xaxes(title=colunas_legiveis["segmento"])
fig.update_yaxes(title=colunas_legiveis["ciclos_inativos"])
salvar(fig, "ciclos_inativos_por_segmento")

# 13. Receita vs Idade
fig = px.scatter(df, x='idade', y='venda_vlr_receita_liquida', color='genero', opacity=0.5,
                 title=f'{colunas_legiveis["venda_vlr_receita_liquida"]} vs {colunas_legiveis["idade"]} por {colunas_legiveis["genero"]}')
fig.update_xaxes(title=colunas_legiveis["idade"])
fig.update_yaxes(title=colunas_legiveis["venda_vlr_receita_liquida"])
salvar(fig, "scatter_idade_receita")

# Gráfico 14: Heatmap Correlação
num_cols = ['venda_qtd', 'venda_vlr_receita_liquida', 'venda_vlr_receita_bruta', 'idade', 'ciclos_inativos']
corr = df[num_cols].corr()
z_values = np.round(corr.values, 4)
colunas_legiveis_corr = [colunas_legiveis[col] for col in num_cols]
fig = ff.create_annotated_heatmap(
    z=z_values,
    x=colunas_legiveis_corr,
    y=colunas_legiveis_corr,
    annotation_text=z_values.astype(str),
    colorscale='Viridis',
    showscale=True
)
fig.update_layout(title='Correlação entre Variáveis Numéricas (4 casas decimais)')
salvar(fig, "heatmap_correlacoes")


# 15. Receita por Gênero e Ciclo (Top 6, facetado)
fig = px.box(df_top_ciclos, x='genero', y='venda_vlr_receita_liquida', color='genero',
             facet_col='venda_ciclo', facet_col_wrap=3,
             title=f'{colunas_legiveis["venda_vlr_receita_liquida"]} por {colunas_legiveis["genero"]} nos Top Ciclos')
salvar(fig, "box_receita_genero_top_ciclos")

# 16. Inatividade por Faixa Etária
df['faixa_etaria'] = pd.cut(df['idade'], bins=[0, 20, 30, 40, 50, 60, 80, 100],
                            labels=['<20', '20-30', '30-40', '40-50', '50-60', '60-80', '80+'])
agg = df.groupby('faixa_etaria')['ciclos_inativos'].mean().reset_index()
fig = px.bar(agg, x='faixa_etaria', y='ciclos_inativos',
             title=f'Média de {colunas_legiveis["ciclos_inativos"]} por Faixa Etária')
fig.update_xaxes(title='Faixa Etária')
fig.update_yaxes(title=colunas_legiveis["ciclos_inativos"])
salvar(fig, "inatividade_faixa_etaria")

# 17. Painel resumo (matplotlib)
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs[0, 0].hist(df['idade'].dropna(), bins=30, color='skyblue')
axs[0, 0].set_title(f'Distribuição de {colunas_legiveis["idade"]}')
df.groupby('segmento')['venda_vlr_receita_liquida'].sum().plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title(f'{colunas_legiveis["venda_vlr_receita_liquida"]} por {colunas_legiveis["segmento"]}')
df['status'].value_counts().plot(kind='bar', ax=axs[1, 0], color='orange')
axs[1, 0].set_title(f'Distribuição por {colunas_legiveis["status"]}')
df.groupby('status')['ciclos_inativos'].mean().plot(kind='bar', ax=axs[1, 1], color='green')
axs[1, 1].set_title(f'Média de {colunas_legiveis["ciclos_inativos"]} por {colunas_legiveis["status"]}')
plt.tight_layout()
plt.savefig("plots/resumo_painel.png")
plt.clf()
