"""
transform_vendas_quick.py
Versão rápida que processa vendas parquet incrementalmente com feedback
Estrutura: UF (Estoque) → Revendedor (PDV) → Produto → Período
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*70)
    print("  Transform Vendas (Rede Multi-Nível)")
    print("  UF=Estoque → Revendedor=PDV → Produto → Período")
    print("="*70)
    
    vendas_dir = "data/source/vendas"
    output_file = "data/vendas_preparadas.csv"
    
    # Encontrar arquivos
    pattern = os.path.join(vendas_dir, "**", "*.parquet")
    parquet_files = sorted(glob.glob(pattern, recursive=True))
    
    print(f"\n[1] Encontrados {len(parquet_files)} arquivos")
    
    # Processar
    all_data = []
    for idx, pfile in enumerate(parquet_files, 1):
        # Extrair UF do caminho: data/source/vendas/uf=AC/*.parquet
        uf = os.path.basename(os.path.dirname(pfile)).replace("uf=", "").upper()
        
        # Pular partição vazia
        if uf == "__HIVE_DEFAULT_PARTITION__":
            print(f"  [{idx:>2}/{len(parquet_files)}] {uf:<4}... (skip)")
            continue
        
        print(f"  [{idx:>2}/{len(parquet_files)}] {uf}...", end=" ", flush=True)
        
        try:
            df = pd.read_parquet(pfile, engine='pyarrow')
            
            # Agrupar por dimensões: UF + Produto + Revendedor + Período
            grouped = df.groupby(['produto_cod', 'revendedor_cod', 'venda_ciclo']).agg({
                'venda_qtd': 'sum',
                'venda_vlr_receita_liquida': 'sum'
            }).reset_index()
            
            # Adicionar UF (estoque/estado)
            grouped['uf'] = uf
            
            all_data.append(grouped)
            print(f"✓ {len(grouped)} registros")
        
        except Exception as e:
            print(f"❌ Erro: {str(e)[:50]}")
    
    if not all_data:
        print("\n❌ Nenhum arquivo foi processado")
        return
    
    print(f"\n[2] Combinando...")
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Agregar novamente em caso de duplicatas
    df_final = df_final.groupby(['uf', 'produto_cod', 'revendedor_cod', 'venda_ciclo']).agg({
        'venda_qtd': 'sum',
        'venda_vlr_receita_liquida': 'sum'
    }).reset_index()
    
    print(f"  {len(df_final)} registros únicos")
    
    # Converter colunas
    df_final['date'] = df_final['venda_ciclo'].astype(str).apply(
        lambda x: f"{x[:4]}-{x[4:6]}-01"
    )
    df_final['item_id'] = df_final['produto_cod'].astype(str)
    df_final['store_id'] = df_final['revendedor_cod'].astype(str)
    # Converter para float (pode ter valores muito grandes que não cabem em int)
    df_final['sales'] = pd.to_numeric(df_final['venda_qtd'], errors='coerce').fillna(0).astype(float)
    df_final['revenue'] = pd.to_numeric(df_final['venda_vlr_receita_liquida'], errors='coerce').fillna(0).astype(float)
    df_final['avg_price'] = (df_final['revenue'] / df_final['sales'].replace(0, np.nan)).fillna(0)
    
    # Selecionar colunas: UF como coluna explícita (estoque/warehouse)
    output_cols = ['date', 'venda_ciclo', 'uf', 'item_id', 'store_id', 'sales', 'revenue', 'avg_price']
    df_output = df_final[output_cols].copy()
    
    # Salvar
    print(f"\n[3] Salvando em {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    print(f"  ✓ Salvo")
    
    # Estatísticas
    print(f"\n[4] Estatísticas (REDE MULTI-NÍVEL):")
    print(f"    Registros: {len(df_output):,}")
    
    unique_ufs = df_output['uf'].nunique()
    unique_produtos = df_output['item_id'].nunique()
    unique_revendedores = df_output['store_id'].nunique()
    unique_periodos = df_output['venda_ciclo'].nunique()
    
    print(f"\n    Dimensões:")
    print(f"      Estoques (UFs): {unique_ufs}")
    print(f"      Produtos: {unique_produtos:,}")
    print(f"      Revendedores (PDVs): {unique_revendedores:,}")
    print(f"      Períodos: {unique_periodos}")
    
    print(f"\n    Demanda:")
    print(f"      Total: {df_output['sales'].sum():,.0f} unidades")
    print(f"      Receita: R$ {df_output['revenue'].sum():,.2f}")
    
    # Por estoque
    print(f"\n    Por Estoque (UF):")
    for uf in sorted(df_output['uf'].unique())[:5]:
        uf_data = df_output[df_output['uf'] == uf]
        print(f"      {uf}: {uf_data['sales'].sum():>8,.0f} un | "
              f"R$ {uf_data['revenue'].sum():>12,.0f} | "
              f"{uf_data['store_id'].nunique():>3} revendedores")
    
    # Exemplos
    print(f"\n[5] Amostra dos dados:")
    for _, row in df_output.head(5).iterrows():
        print(f"    [{row['uf']}] P:{row['item_id']:<8} → R:{row['store_id']:<10} "
              f"({row['venda_ciclo']}) {row['sales']:>5} un")
    
    print(f"\n✅ Próximo:")
    print(f"   1. Ajuste config.yaml:")
    print(f"      file_path: \"data/vendas_preparadas.csv\"")
    print(f"      columns:")
    print(f"        warehouse: \"uf\"        # Novo: estoque por estado")
    print(f"        store_id: \"store_id\"   # PDV dentro do estoque")
    print(f"   2. Execute: python main.py")

if __name__ == "__main__":
    main()
