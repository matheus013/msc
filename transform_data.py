"""
transform_data.py
Transforma dados JSON da estrutura multi-dimensional para CSV compatível com main.py
"""
import json
import os
import csv
from datetime import datetime
from pathlib import Path

# Tentar importar pandas, se não tiver, usar alternativa com csv
try:
    import pandas as pd
    HAS_PANDAS = True
except:
    HAS_PANDAS = False
    print("ℹ pandas não disponível, usando CSV puro")

def load_json_safely(filepath):
    """Carrega JSON com tratamento de erro"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Erro ao ler {filepath}: {e}")
        return {}

def transform_scenario_data(scenario_dir):
    """
    Transforma dados de um cenário (ex: 202301) em formato tabular
    
    Retorna DataFrame com colunas:
      - date
      - item_id
      - store_id
      - demand
      - production_cost
      - initial_stock
      - capacity
    """
    
    print(f"\n  Processando: {scenario_dir}")
    
    # Carregar arquivos
    demand_file = os.path.join(scenario_dir, "demand.json")
    capacity_file = os.path.join(scenario_dir, "capacity.json")
    initial_stock_file = os.path.join(scenario_dir, "initial_stock.json")
    production_cost_file = os.path.join(scenario_dir, "production_cost.json")
    transport_cost_file = os.path.join(scenario_dir, "transport_cost.json")
    
    demand = load_json_safely(demand_file)  # {item_id: {store_id: qty}}
    capacity = load_json_safely(capacity_file)  # {store|item: cap}
    initial_stock = load_json_safely(initial_stock_file)  # {store: {item_id: qty}}
    prod_cost = load_json_safely(production_cost_file)  # {factory: {item_id: cost}}
    transport_cost = load_json_safely(transport_cost_file)  # {origem|destino: cost}
    
    # Extrair período a partir do nome da pasta (ex: "202301" → 2023-01)
    period = os.path.basename(scenario_dir)
    if len(period) == 6:  # Formato: YYYYMM
        year, month = period[:4], period[4:6]
        start_date = datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d")
    else:
        start_date = datetime.now()
    
    # Montar dados em lista
    rows = []
    
    for item_id, stores_dict in demand.items():
        for store_id, qty_demanded in stores_dict.items():
            
            # Extrair capacidade: procura por "store|item" ou equivalente
            cap_key = f"{store_id}|{item_id}"
            cap_value = capacity.get(cap_key, 1000)  # default 1000
            
            # Extrair estoque inicial
            initial = initial_stock.get(store_id, {}).get(item_id, 0)
            
            # Extrair custo de produção (usar primeiro factory encontrado ou média)
            prod_cost_value = None
            if prod_cost:
                # Tentar usar o custo da primeira fábrica que tem esse item
                for factory, items_dict in prod_cost.items():
                    if item_id in items_dict:
                        prod_cost_value = items_dict[item_id]
                        break
            if prod_cost_value is None:
                prod_cost_value = 20.0  # default
            
            rows.append({
                'date': start_date,
                'period': period,
                'item_id': str(item_id),
                'store_id': str(store_id),
                'demand': int(qty_demanded),
                'production_cost': float(prod_cost_value),
                'initial_stock': int(initial),
                'capacity': int(cap_value),
            })
    
    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df)} registros extraídos")
    return df

def aggregate_multi_store_demand(df, method='sum'):
    """
    Agrega demanda de múltiplas lojas em uma série única
    
    Métodos:
      - 'sum': soma de todas as lojas (demanda agregada)
      - 'mean': média das lojas
      - 'single': pega a primeira loja (para um item específico)
    """
    print(f"\n  Agregando demanda (método: {method})...")
    
    if method == 'sum':
        agg_df = df.groupby('date').agg({
            'demand': 'sum',
            'production_cost': 'mean',
            'capacity': 'sum',
            'initial_stock': 'mean'
        }).reset_index()
        agg_df['item_id'] = 'TOTAL'
        agg_df['store_id'] = 'AGGREGATED'
    
    elif method == 'mean':
        agg_df = df.groupby('date').agg({
            'demand': 'mean',
            'production_cost': 'mean',
            'capacity': 'mean',
            'initial_stock': 'mean'
        }).reset_index()
        agg_df['item_id'] = 'MEAN'
        agg_df['store_id'] = 'ALL'
    
    else:  # single - primeira loja
        first_item_store = df[['item_id', 'store_id']].drop_duplicates().iloc[0]
        agg_df = df[(df['item_id'] == first_item_store['item_id']) & 
                    (df['store_id'] == first_item_store['store_id'])].copy()
        agg_df = agg_df.sort_values('date')
    
    print(f"    ✓ {len(agg_df)} períodos únicos")
    return agg_df

def main():
    print("="*70)
    print("  Transform Data: JSON → CSV (compatível com main.py)")
    print("="*70)
    
    # Paths
    base_data_dir = "data/scenario"
    output_file = "data/dados_preparados.csv"
    
    if not os.path.exists(base_data_dir):
        print(f"\n❌ Diretório não encontrado: {base_data_dir}")
        return
    
    # Encontrar todas as pastas de cenário (202301, 202302, etc)
    scenario_dirs = sorted([
        d for d in os.listdir(base_data_dir) 
        if os.path.isdir(os.path.join(base_data_dir, d)) and d.isdigit()
    ])
    
    print(f"\n[1] Encontrados {len(scenario_dirs)} cenários")
    
    # Processar cada cenário
    all_dfs = []
    for scenario in scenario_dirs:
        scenario_path = os.path.join(base_data_dir, scenario)
        df = transform_scenario_data(scenario_path)
        all_dfs.append(df)
    
    # Combinar todos
    print(f"\n[2] Combinando dados de todos os cenários...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    print(f"    ✓ Total: {len(combined_df)} registros | "
          f"Períodos: {combined_df['date'].min()} a {combined_df['date'].max()}")
    
    # Opções de agregação
    print(f"\n[3] Escolha de agregação:")
    print(f"    [1] Soma (demanda agregada de todas as lojas)")
    print(f"    [2] Média (média das lojas)")
    print(f"    [3] Por Item+Loja (tabela desagregada)")
    print(f"    [4] Primeira loja (exemplo para teste rápido)")
    
    choice = input(f"\n  Opção (1-4) [padrão: 1]: ").strip() or "1"
    
    if choice == "1":
        final_df = aggregate_multi_store_demand(combined_df, method='sum')
    elif choice == "2":
        final_df = aggregate_multi_store_demand(combined_df, method='mean')
    elif choice == "3":
        final_df = combined_df[['date', 'item_id', 'store_id', 'demand', 'production_cost', 'capacity', 'initial_stock']]
        print(f"  Mantendo {len(final_df)} registros desagregados")
    else:  # 4 ou outro
        final_df = aggregate_multi_store_demand(combined_df, method='single')
    
    # Preparar para o código
    print(f"\n[4] Preparando para o código...")
    final_df['sales'] = final_df['demand']  # Coluna esperada pelo config
    final_df = final_df.rename(columns={
        'production_cost': 'cost',
        'capacity': 'max_capacity'
    })
    
    # Salvar
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"    ✓ Salvo em: {output_file}")
    
    # Resumo
    print(f"\n[5] Resumo dos dados:")
    print(f"    Shape: {final_df.shape}")
    print(f"    Colunas: {list(final_df.columns)}")
    print(f"\n    Demanda:")
    print(f"      Total: {final_df['demand'].sum():.0f} unidades")
    print(f"      Média: {final_df['demand'].mean():.2f} unidades/período")
    print(f"      Min/Max: {final_df['demand'].min():.0f} / {final_df['demand'].max():.0f}")
    print(f"      σ: {final_df['demand'].std():.2f}")
    
    print(f"\n    Custo de Produção:")
    print(f"      Média: ${final_df['cost'].mean():.2f}")
    print(f"      Min/Max: ${final_df['cost'].min():.2f} / ${final_df['cost'].max():.2f}")
    
    print(f"\n✅ Transformação completa! Próximo passo: ajustar config.yaml")
    print(f"\n   # Em config.yaml, altere:")
    print(f"   file_path: \"data/dados_preparados.csv\"")
    print(f"   columns:")
    print(f"     demand: \"sales\"")
    print(f"     price: \"cost\"")

if __name__ == "__main__":
    main()
