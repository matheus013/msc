"""
transform_vendas_parquet.py
Transforma arquivos Parquet de vendas diretamente mantendo granularidade
"""
import os
import glob
import pandas as pd
from collections import defaultdict

def process_vendas_parquets(vendas_dir):
    """
    Processa todos os arquivos parquet de vendas
    Mantém granularidade: produto_cod x revendedor_cod x período
    """
    
    print(f"\n  Processando diretório: {vendas_dir}")
    
    # Encontrar todos os arquivos parquet
    pattern = os.path.join(vendas_dir, "**", "*.parquet")
    parquet_files = glob.glob(pattern, recursive=True)
    
    print(f"    Encontrados {len(parquet_files)} arquivos parquet")
    
    all_rows = []
    
    for idx, parquet_file in enumerate(parquet_files, 1):
        print(f"      [{idx}/{len(parquet_files)}] Lendo {os.path.basename(os.path.dirname(parquet_file))}/...")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Agrupar por produto, revendedor e ciclo
            grouped = df.groupby(['produto_cod', 'revendedor_cod', 'venda_ciclo']).agg({
                'venda_qtd': 'sum',
                'venda_vlr_receita_liquida': 'sum'
            }).reset_index()
            
            grouped.columns = ['produto_cod', 'revendedor_cod', 'venda_ciclo', 'demand', 'revenue']
            
            all_rows.append(grouped)
            print(f"        ✓ {len(grouped)} registros")
        
        except Exception as e:
            print(f"        ⚠ Erro: {e}")
    
    # Combinar todos os dataframes
    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        # Agregar novamente em caso de sobreposição entre arquivos
        final_df = final_df.groupby(['produto_cod', 'revendedor_cod', 'venda_ciclo']).agg({
            'demand': 'sum',
            'revenue': 'sum'
        }).reset_index()
        print(f"\n    ✓ Total: {len(final_df)} combinações produto×revendedor×período")
        return final_df
    else:
        print(f"    ❌ Nenhum arquivo processado")
        return None

def main():
    print("="*70)
    print("  Transform Vendas Parquet: Mantendo Granularidade")
    print("  produto_cod x revendedor_cod x venda_ciclo")
    print("="*70)
    
    vendas_dir = "data/source/vendas"
    
    if not os.path.exists(vendas_dir):
        print(f"\n[ERRO] Diretório não encontrado: {vendas_dir}")
        return
    
    # Processar arquivos
    print(f"\n[1] Processando arquivos parquet...")
    df = process_vendas_parquets(vendas_dir)
    
    if df is None or len(df) == 0:
        print("\n❌ Nenhum dado foi processado")
        return
    
    # Converter ciclo para data
    df['date'] = df['venda_ciclo'].apply(
        lambda x: f"{str(x)[:4]}-{str(x)[4:6]}-01"
    )
    
    # Calcular preço médio
    df['avg_price'] = df['revenue'] / df['demand']
    df['avg_price'] = df['avg_price'].fillna(0)
    
    # Renomear colunas para compatibilidade com config.yaml
    df['item_id'] = df['produto_cod']
    df['store_id'] = df['revendedor_cod']
    df['sales'] = df['demand']
    
    print(f"\n[2] Dados processados:")
    print(f"    Registros: {len(df)}")
    print(f"    Demanda total: {df['demand'].sum():.0f} unidades")
    print(f"    Receita total: R$ {df['revenue'].sum():,.2f}")
    
    # Análise dimensional
    unique_produtos = df['item_id'].nunique()
    unique_stores = df['store_id'].nunique()
    unique_periods = df['venda_ciclo'].nunique()
    
    print(f"\n    Dimensões:")
    print(f"      Produtos: {unique_produtos}")
    print(f"      Revendedores: {unique_stores}")
    print(f"      Períodos: {unique_periods}")
    print(f"      Potencial máximo: {unique_produtos * unique_stores * unique_periods:,}")
    print(f"      Densidade: {len(df) / (unique_produtos * unique_stores * unique_periods):.1%}")
    
    # Salvar em CSV
    output_file = "data/vendas_preparadas.csv"
    fieldnames = ['date', 'venda_ciclo', 'item_id', 'store_id', 'sales', 'revenue', 'avg_price']
    
    output_df = df[fieldnames].copy()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    print(f"\n[3] CSV Salvo: {output_file}")
    print(f"    Colunas: {', '.join(fieldnames)}")
    
    # Exemplo de dados
    print(f"\n[4] Primeiros registros:")
    for _, row in output_df.head(5).iterrows():
        print(f"    Produto {row['item_id']:<8} | Revendedor {row['store_id']:<10} | "
              f"Período {row['venda_ciclo']} | Demanda {row['sales']:>5} | R$ {row['revenue']:>10,.2f}")
    
    # Dicas
    print(f"\n✅ Próximos passos:")
    print(f"   1. Ajuste config.yaml:")
    print(f"      file_path: \"data/vendas_preparadas.csv\"")
    print(f"      columns:")
    print(f"        demand: \"sales\"")
    print(f"        item_id: \"item_id\"")
    print(f"        store_id: \"store_id\"")
    print(f"        price: \"avg_price\"")
    print(f"      filter_single_item: false")
    print(f"   2. Execute: python main.py")

if __name__ == "__main__":
    main()
