"""
transform_vendas_sample.py
Versão de amostra: processa apenas PB para teste rápido
Arquitetura multi-nível: warehouse (UF) → PDV → Produto
"""
import os
import glob
import pandas as pd

def main():
    print("="*70)
    print("  Transform Vendas (AMOSTRA - PB para teste rápido)")
    print("  Arquitetura: Warehouse (UF) → PDV → Produto")
    print("="*70)
    
    vendas_dir = "data/source/vendas"
    output_file = "data/vendas_sample.csv"
    
    # Encontrar apenas PB
    pattern = os.path.join(vendas_dir, "**", "*.parquet")
    all_files = sorted(glob.glob(pattern, recursive=True))
    
    # Filtrar apenas PB
    parquet_files = [f for f in all_files if f'uf=PB' in f]
    
    print(f"\n[1] Processando {len(parquet_files)} arquivo(s) (PB)")
    
    all_data = []
    for pfile in parquet_files:
        # Extrair UF do caminho: data/source/vendas/uf=AC/...
        uf_path = os.path.basename(os.path.dirname(pfile))
        uf = uf_path.replace("uf=", "")
        
        print(f"  {uf}...", end=" ", flush=True)
        
        try:
            df = pd.read_parquet(pfile, engine='pyarrow')
            
            # Adicionar warehouse (UF)
            df['warehouse'] = uf
            
            # Agrupar por dimensões: warehouse → produto → revendedor → período
            grouped = df.groupby([
                # 'warehouse', 
                'produto_cod', 
                'revendedor_cod', 
                'venda_ciclo'
                ]).agg({
                'venda_qtd': 'sum',
                'venda_vlr_receita_liquida': 'sum'
            }).reset_index()
            print(grouped)
            all_data.append(grouped)
            print(f"✓ {len(grouped)} registros")
        
        except Exception as e:
            print(f"❌ {str(e)[:40]}")
    
    print(f"\n[2] Combinando...")
    df_final = pd.concat(all_data, ignore_index=True)
    
    print(f"  {len(df_final)} registros")
    
    # Converter colunas
    df_final['date'] = df_final['venda_ciclo'].astype(str).apply(
        lambda x: f"{x[:4]}-{x[4:6]}-01"
    )
    df_final['item_id'] = df_final['produto_cod'].astype(str)
    df_final['store_id'] = df_final['revendedor_cod'].astype(str)
    df_final['sales'] = pd.to_numeric(df_final['venda_qtd'], errors='coerce').fillna(0).astype(int)
    df_final['revenue'] = pd.to_numeric(df_final['venda_vlr_receita_liquida'], errors='coerce').fillna(0)
    df_final['avg_price'] = (df_final['revenue'] / df_final['sales'].replace(0, 1)).fillna(0)

    print(df_final)
    
    # Selecionar colunas (incluindo warehouse)
    output_cols = ['date', 'venda_ciclo', 'warehouse', 'item_id', 'store_id', 'sales', 'revenue', 'avg_price']
    df_output = df_final[output_cols].copy()
    
    # Salvar
    print(f"\n[3] Salvando {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    print(f"  ✓ Salvo")
    
    # Estatísticas
    print(f"\n[4] Estatísticas (AMOSTRA - Arquitetura Multi-Nível):")
    print(f"    Registros: {len(df_output):,}")
    print(f"    Warehouses (UF): {df_output['warehouse'].nunique()} → {sorted(df_output['warehouse'].unique())}")
    print(f"    Produtos: {df_output['item_id'].nunique():,}")
    print(f"    Revendedores (PDVs): {df_output['store_id'].nunique():,}")
    print(f"    Períodos: {df_output['venda_ciclo'].nunique():,}")
    print(f"    Demanda: {df_output['sales'].sum():,.0f} unidades")
    print(f"    Receita: R$ {df_output['revenue'].sum():,.2f}")
    
    # Análise por warehouse
    print(f"\n[5] Detalhamento por Warehouse:")
    for uf in sorted(df_output['warehouse'].unique()):
        df_uf = df_output[df_output['warehouse'] == uf]
        print(f"    {uf}: {len(df_uf):,} registros, {df_uf['item_id'].nunique()} produtos, {df_uf['store_id'].nunique()} PDVs")
    
    print(f"\n✅ Para usar em config.yaml:")
    print(f"    file_path: \"data/vendas_sample.csv\"")
    print(f"    warehouse: \"warehouse\"  # Coluna de warehouse adicionada")
    print(f"    selected_warehouse: null  # Auto-selecionar warehouse com maior demanda")

if __name__ == "__main__":
    main()

