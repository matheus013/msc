"""
transform_vendas_data.py
Transforma dados Parquet de vendas em CSV mantendo produto_cod × revendedor_cod granulares
"""
import json
import os
import csv
import glob
from collections import defaultdict
from datetime import datetime

def load_jsonl_file(filepath):
    """Carrega arquivo JSONL (JSON Lines - um JSON por linha)"""
    rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  ⚠ Erro ao parsear linha em {filepath}: {e}")
    except Exception as e:
        print(f"  ⚠ Erro ao ler {filepath}: {e}")
    return rows

def transform_vendas_files(vendas_dir):
    """
    Transforma arquivos de vendas (parquet convertidos em JSON Lines) 
    para formato tabular com granularidade produto x revendedor x período
    
    Retorna lista de dicts com:
      - date: data período
      - venda_ciclo: período (YYYYMM)
      - produto_cod: código do produto
      - revendedor_cod: código do revendedor (store)
      - demand: quantidade total vendida nesse período
      - revenue: receita total nesse período
    """
    
    print(f"\n  Processando diretório: {vendas_dir}")
    
    # Encontrar todos os arquivos JSON (convertidos de parquet)
    pattern = os.path.join(vendas_dir, "**", "*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"    Encontrados {len(json_files)} arquivos")
    
    # Dicionário para agregar dados: {(produto, revendedor, ciclo): {demand, revenue, count}}
    aggregated = defaultdict(lambda: {'demand': 0, 'revenue': 0, 'count': 0})
    
    for json_file in json_files:
        print(f"      Lendo {os.path.basename(json_file)}...")
        rows = load_jsonl_file(json_file)
        
        for row in rows:
            try:
                produto = str(row.get('produto_cod', 'UNKNOWN'))
                revendedor = str(row.get('revendedor_cod', 'UNKNOWN'))
                ciclo = str(row.get('venda_ciclo', 'UNKNOWN'))
                qtd = int(row.get('venda_qtd', 0))
                receita = float(row.get('venda_vlr_receita_liquida', 0.0))
                
                key = (produto, revendedor, ciclo)
                aggregated[key]['demand'] += qtd
                aggregated[key]['revenue'] += receita
                aggregated[key]['count'] += 1
            except Exception as e:
                print(f"      ⚠ Erro processando linha: {e}")
    
    # Converter para lista de dicts
    rows = []
    for (produto, revendedor, ciclo), agg_data in aggregated.items():
        # Converter ciclo YYYYMM para data
        if len(ciclo) == 6:
            year, month = ciclo[:4], ciclo[4:6]
            date_str = f"{year}-{month}-01"
        else:
            date_str = "2023-01-01"
        
        rows.append({
            'date': date_str,
            'venda_ciclo': ciclo,
            'produto_cod': produto,
            'revendedor_cod': revendedor,
            'demand': agg_data['demand'],
            'revenue': agg_data['revenue'],
            'avg_price': agg_data['revenue'] / agg_data['demand'] if agg_data['demand'] > 0 else 0
        })
    
    print(f"    ✓ {len(rows)} combinações produto×revendedor×período extraídas")
    return rows

def write_csv(filename, rows, fieldnames):
    """Escreve CSV"""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    ✓ Salvo: {filename}")

def main():
    print("="*70)
    print("  Transform Vendas: Parquet (JSONL) to CSV")
    print("  Mantendo granularidade: produto_cod x revendedor_cod x periodo")
    print("="*70)
    
    vendas_dir = "data/source/vendas"
    
    if not os.path.exists(vendas_dir):
        print(f"\n[ERRO] Diretório não encontrado: {vendas_dir}")
        return
    
    # Processar todos os arquivos
    print(f"\n[1] Processando arquivos de vendas...")
    all_rows = transform_vendas_files(vendas_dir)
    
    print(f"\n[2] Total de {len(all_rows)} registros extraídos")
    
    # Opções
    print(f"\n[3] Escolha de agregação:")
    print(f"    [1] Desagregado: produto x revendedor x período (RECOMENDADO)")
    print(f"    [2] Por produto: soma de todos revendedores")
    print(f"    [3] Por revendedor: soma de todos produtos")
    
    choice = input(f"\n  Opção (1-3) [padrão: 1]: ").strip() or "1"
    
    if choice == "1":
        final_rows = all_rows
        print(f"  Mantendo todas as {len(final_rows)} combinações")
    
    elif choice == "2":
        # Agregar por produto
        by_product = defaultdict(lambda: {
            'demand': 0, 'revenue': 0, 'ciclos': set()
        })
        for row in all_rows:
            key = (row['date'], row['venda_ciclo'], row['produto_cod'])
            by_product[key]['demand'] += row['demand']
            by_product[key]['revenue'] += row['revenue']
            by_product[key]['ciclos'].add(row['revendedor_cod'])
        
        final_rows = []
        for (date, ciclo, produto), agg in by_product.items():
            final_rows.append({
                'date': date,
                'venda_ciclo': ciclo,
                'produto_cod': produto,
                'revendedor_cod': 'ALL',
                'demand': agg['demand'],
                'revenue': agg['revenue'],
                'avg_price': agg['revenue'] / agg['demand'] if agg['demand'] > 0 else 0
            })
        print(f"  Agregado por produto: {len(final_rows)} registros")
    
    elif choice == "3":
        # Agregar por revendedor
        by_revendedor = defaultdict(lambda: {
            'demand': 0, 'revenue': 0, 'produtos': set()
        })
        for row in all_rows:
            key = (row['date'], row['venda_ciclo'], row['revendedor_cod'])
            by_revendedor[key]['demand'] += row['demand']
            by_revendedor[key]['revenue'] += row['revenue']
            by_revendedor[key]['produtos'].add(row['produto_cod'])
        
        final_rows = []
        for (date, ciclo, revendedor), agg in by_revendedor.items():
            final_rows.append({
                'date': date,
                'venda_ciclo': ciclo,
                'produto_cod': 'ALL',
                'revendedor_cod': revendedor,
                'demand': agg['demand'],
                'revenue': agg['revenue'],
                'avg_price': agg['revenue'] / agg['demand'] if agg['demand'] > 0 else 0
            })
        print(f"  Agregado por revendedor: {len(final_rows)} registros")
    
    else:
        final_rows = all_rows
        print(f"  Mantendo todas as {len(final_rows)} combinações (padrão)")
    
    # Renomear para 'sales' e 'item_id' / 'store_id' (compatível com config.yaml)
    for row in final_rows:
        row['sales'] = row.pop('demand')
        row['item_id'] = row['produto_cod']
        row['store_id'] = row['revendedor_cod']
    
    # Salvar
    output_file = "data/vendas_preparadas.csv"
    fieldnames = ['date', 'venda_ciclo', 'item_id', 'store_id', 'sales', 'revenue', 'avg_price']
    
    print(f"\n[4] Salvando...")
    write_csv(output_file, final_rows, fieldnames)
    
    # Resumo
    print(f"\n[5] Resumo dos dados:")
    total_demand = sum(r['sales'] for r in final_rows)
    min_demand = min(r['sales'] for r in final_rows)
    max_demand = max(r['sales'] for r in final_rows)
    avg_demand = total_demand / len(final_rows)
    
    print(f"    Registros: {len(final_rows)}")
    print(f"    Demanda total: {total_demand:.0f} unidades")
    print(f"    Demanda média: {avg_demand:.2f} unidades/registro")
    print(f"    Demanda min/max: {min_demand:.0f} / {max_demand:.0f}")
    
    total_revenue = sum(r['revenue'] for r in final_rows)
    avg_price = total_revenue / total_demand if total_demand > 0 else 0
    
    print(f"\n    Receita total: R$ {total_revenue:,.2f}")
    print(f"    Preco medio: R$ {avg_price:.2f}")
    
    # Análise de dimensionalidade
    if choice == "1":
        unique_produtos = len(set(r['item_id'] for r in final_rows))
        unique_stores = len(set(r['store_id'] for r in final_rows))
        unique_periods = len(set(r['venda_ciclo'] for r in final_rows))
        
        print(f"\n    Dimensões:")
        print(f"      Produtos únicos: {unique_produtos}")
        print(f"      Revendedores únicos: {unique_stores}")
        print(f"      Períodos: {unique_periods}")
        print(f"      Esperado (completo): {unique_produtos * unique_stores * unique_periods}")
        print(f"      Densidade: {len(final_rows) / (unique_produtos * unique_stores * unique_periods):.1%}")
    
    print(f"\n✅ Próximos passos:")
    print(f"   1. Verifique {output_file}")
    print(f"   2. Ajuste config.yaml:")
    print(f"      file_path: \"data/vendas_preparadas.csv\"")
    print(f"      columns:")
    print(f"        demand: \"sales\"")
    print(f"        item_id: \"item_id\"")
    print(f"        store_id: \"store_id\"")
    print(f"      filter_single_item: false")
    print(f"   3. Execute: python main.py")

if __name__ == "__main__":
    main()
