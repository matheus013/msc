"""
transform_data_simple.py
Transforma dados JSON → CSV (versão sem dependências externas, apenas json + csv)
"""
import json
import os
import csv
from datetime import datetime
from collections import defaultdict

def load_json_safely(filepath):
    """Carrega JSON com tratamento de erro"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Erro ao ler {filepath}: {e}")
        return {}

def extract_period_date(period_str):
    """Extrai data de string como '202301'"""
    if len(period_str) == 6:
        year, month = period_str[:4], period_str[4:6]
        return f"{year}-{month}-01"
    return "2023-01-01"

def transform_scenario_data(scenario_dir):
    """Transforma dados de um cenário em lista de dicts"""
    
    print(f"\n  Processando: {os.path.basename(scenario_dir)}")
    
    # Carregar arquivos
    demand = load_json_safely(os.path.join(scenario_dir, "demand.json"))
    capacity = load_json_safely(os.path.join(scenario_dir, "capacity.json"))
    initial_stock = load_json_safely(os.path.join(scenario_dir, "initial_stock.json"))
    prod_cost = load_json_safely(os.path.join(scenario_dir, "production_cost.json"))
    
    period = os.path.basename(scenario_dir)
    date_str = extract_period_date(period)
    
    rows = []
    for item_id, stores_dict in demand.items():
        for store_id, qty_demanded in stores_dict.items():
            cap_key = f"{store_id}|{item_id}"
            cap_value = capacity.get(cap_key, 1000)
            
            initial = initial_stock.get(store_id, {}).get(item_id, 0)
            
            # Custo de produção
            prod_cost_value = None
            for factory, items_dict in prod_cost.items():
                if item_id in items_dict:
                    prod_cost_value = items_dict[item_id]
                    break
            if prod_cost_value is None:
                prod_cost_value = 20.0
            
            rows.append({
                'date': date_str,
                'period': period,
                'item_id': str(item_id),
                'store_id': str(store_id),
                'demand': int(qty_demanded),
                'production_cost': float(prod_cost_value),
                'initial_stock': int(initial),
                'capacity': int(cap_value),
            })
    
    print(f"    ✓ {len(rows)} registros extraídos")
    return rows

def write_csv(filename, rows, fieldnames):
    """Escreve CSV"""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    ✓ Salvo: {filename}")

def main(choice=None):
    print("="*70)
    print("  Transform Data: JSON to CSV")
    print("="*70)
    
    base_data_dir = "data/scenario"
    
    if not os.path.exists(base_data_dir):
        print(f"\n[ERRO] Diretório não encontrado: {base_data_dir}")
        return
    
    scenario_dirs = sorted([
        d for d in os.listdir(base_data_dir) 
        if os.path.isdir(os.path.join(base_data_dir, d)) and d.isdigit()
    ])
    
    print(f"\n[1] Encontrados {len(scenario_dirs)} cenários: {scenario_dirs[:3]}...")
    
    # Processar
    all_rows = []
    for scenario in scenario_dirs:
        scenario_path = os.path.join(base_data_dir, scenario)
        rows = transform_scenario_data(scenario_path)
        all_rows.extend(rows)
    
    print(f"\n[2] Total de {len(all_rows)} registros")
    
    # Opção de agregação - interativa ou por parâmetro
    if choice is None:
        print(f"\n[3] Escolha de agregação:")
        print(f"    [1] Soma (demanda agregada)")
        print(f"    [2] Por item+loja (desagregado)")
        print(f"    [3] Primeira loja (teste rápido)")
        choice = input(f"\n  Opção (1-3) [padrão: 1]: ").strip() or "1"
    else:
        print(f"\n[3] Usando agregação: {choice}")
    
    if choice == "1":
        # Agregar por data
        agg = defaultdict(lambda: {
            'demand': 0, 'production_cost': 0, 'capacity': 0, 'initial_stock': 0, 'count': 0
        })
        for row in all_rows:
            agg[row['date']]['demand'] += row['demand']
            agg[row['date']]['production_cost'] += row['production_cost']
            agg[row['date']]['capacity'] += row['capacity']
            agg[row['date']]['initial_stock'] += row['initial_stock']
            agg[row['date']]['count'] += 1
        
        final_rows = []
        for date_key in sorted(agg.keys()):
            a = agg[date_key]
            final_rows.append({
                'date': date_key,
                'item_id': 'TOTAL',
                'store_id': 'ALL',
                'sales': a['demand'],  # já usando sales
                'production_cost': a['production_cost'] / a['count'],
                'initial_stock': a['initial_stock'],
                'capacity': a['capacity']
            })
        print(f"  Agregado: {len(final_rows)} períodos únicos")
    
    elif choice == "3":
        # Primeira loja
        first_item = all_rows[0]['item_id']
        first_store = all_rows[0]['store_id']
        final_rows = [r for r in all_rows if r['item_id'] == first_item and r['store_id'] == first_store]
        # Renomear demand para sales
        for row in final_rows:
            row['sales'] = row.pop('demand')
        print(f"  Selecionada: item={first_item}, store={first_store} ({len(final_rows)} períodos)")
    
    else:
        # Desagregado
        final_rows = all_rows
        # Renomear demand para sales
        for row in final_rows:
            row['sales'] = row.pop('demand')
        print(f"  Mantendo todos os {len(final_rows)} registros")
    
    # Salvar
    output_file = "data/dados_preparados.csv"
    fieldnames = ['date', 'item_id', 'store_id', 'sales', 'production_cost', 'initial_stock', 'capacity']
    
    print(f"\n[4] Salvando...")
    write_csv(output_file, final_rows, fieldnames)
    
    # Resumo
    print(f"\n[5] Resumo:")
    total_demand = sum(r['sales'] for r in final_rows)
    min_demand = min(r['sales'] for r in final_rows)
    max_demand = max(r['sales'] for r in final_rows)
    avg_demand = total_demand / len(final_rows)
    
    print(f"    Períodos: {len(final_rows)}")
    print(f"    Demanda total: {total_demand} unidades")
    print(f"    Demanda média: {avg_demand:.1f} unidades/período")
    print(f"    Demanda min/max: {min_demand} / {max_demand}")
    
    avg_cost = sum(r['production_cost'] for r in final_rows) / len(final_rows)
    print(f"    Custo médio: ${avg_cost:.2f}")
    
    print(f"\n✅ Próximos passos:")
    print(f"   1. Verifique {output_file}")
    print(f"   2. Ajuste config.yaml:")
    print(f"      file_path: \"data/dados_preparados.csv\"")
    print(f"      demand: \"sales\"")
    print(f"   3. Execute: python main.py")

if __name__ == "__main__":
    import sys
    # Se passou argumento, usa como choice; caso contrário interativo
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    main(choice=choice)
