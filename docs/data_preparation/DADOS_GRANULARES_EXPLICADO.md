# Dados Ajustados - Granularidade Completa: produto_cod × revendedor_cod × período

## ✅ O que foi feito

### Transformação de Dados
Convertemos os dados **parquet de vendas** para **CSV mantendo a granularidade completa**:

- **Origem:** `data/source/vendas/uf=*/` (27 arquivos parquet, um por estado)
- **Estrutura:** Cada linha = 1 combinação de:
  - `produto_cod` (código do produto)
  - `revendedor_cod` (código do revendedor/loja)
  - `venda_ciclo` (período YYYYMM)

### Arquivos Criados

#### 1. **vendas_sample.csv** ⚡ (228 registros - AC+AL apenas)
**Para teste rápido AGORA:**
- ✓ 167 produtos diferentes
- ✓ 4 revendedores
- ✓ 17 períodos
- ✓ Demanda: 397 unidades
- ✓ Receita: R$ 16.373,07

#### 2. **vendas_preparadas.csv** 📊 (processando...)
**Para uso completo (27 estados):**
- Ainda está processando (arquivo BA tem 3.3M registros)
- Será criado automaticamente quando terminar
- Use-o quando quiser análise completa de TODOS os dados

---

## 🚀 Como Usar AGORA

### Para teste rápido com amostra:

1. **Config.yaml já atualizado** ✓
   ```yaml
   file_path: "data/vendas_sample.csv"
   columns:
     demand: "sales"
     item_id: "item_id"
     store_id: "store_id"
     price: "avg_price"
   filter_single_item: false
   ```

2. **Execute:**
   ```bash
   python main.py
   ```
   ✓ Vai rodar com 228 registros em segundos
   ✓ Testará todas 12 políticas em paralelo

### Para usar dados completos depois:

1. Quando `vendas_preparadas.csv` ficar pronto:
   ```yaml
   file_path: "data/vendas_preparadas.csv"
   ```

2. Execute novamente:
   ```bash
   python main.py
   ```

---

## 📊 Estrutura do CSV

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `date` | data | Primeiro dia do mês (YYYY-MM-01) |
| `venda_ciclo` | str | Período (YYYYMM) |
| `item_id` | str | Código do produto |
| `store_id` | str | Código do revendedor/loja |
| `sales` | int | Quantidade vendida neste período |
| `revenue` | float | Receita total neste período (R$) |
| `avg_price` | float | Preço médio (revenue / sales) |

---

## 🎯 Granularidade Mantida

✅ **Cada linha é única para:** `(produto, revendedor, período)`

Exemplos:
- Produto `84485` × Revendedor `7711051` × Período `202303` = 1 linha
- Produto `84485` × Revendedor `7711051` × Período `202304` = 1 linha (diferente)
- Produto `84056` × Revendedor `7711051` × Período `202303` = 1 linha (diferente)

### Dimensionalidade da Amostra (AC+AL):
- 167 produtos × 4 revendedores × 17 períodos = 11.356 combinações potenciais
- **228 registros** = 2.0% densidade
- (Isso é normal - nem todo produto é vendido em todo período em todo revendedor)

---

## 📝 Próximos Passos

### Opção 1: Teste Rápido (RECOMENDADO - faça agora)
```bash
python main.py
```
✓ Usa `vendas_sample.csv`
✓ Rápido (segundos)
✓ Testa o pipeline completo
✓ Gera gráficos e relatórios em `outputs/`

### Opção 2: Dados Completos (depois)
- Espere o processamento terminar
- O arquivo `vendas_preparadas.csv` aparecerá em `data/`
- Altere `config.yaml` para apontá-lo
- Execute `python main.py` novamente

### Opção 3: Análise por Produto Específico
Edite `config.yaml`:
```yaml
filter_single_item: true
selected_item: "84485"        # especifique o produto
selected_store: "7711051"     # especifique o revendedor (opcional)
```

---

## 🔧 Scripts Disponíveis

| Script | Função |
|--------|--------|
| `transform_vendas_sample.py` | Gera amostra (AC+AL) - **2 min** |
| `transform_vendas_quick.py` | Processa completo (todos estados) - **~30 min** |
| `transform_vendas_parquet.py` | Versão alternativa (não usada) |

---

## 📌 Estado Atual

- [x] Config.yaml atualizado
- [x] vendas_sample.csv pronto (228 registros)
- [ ] vendas_preparadas.csv (processando... ~30 min)
- [ ] main.py pronto para executar

**Status:** ✅ Pronto para teste com amostra

---

**Criado:** Abril 27, 2026  
**Atualização:** Granularidade produto × revendedor × período  
**Versão:** 2.0
