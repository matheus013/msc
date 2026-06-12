# Technical Glossary: Plain Language Explanations

This glossary can be inserted as a highlighted box at the beginning of your dissertation 
or introduction, helping readers understand key concepts as they appear.

---

## Quick Reference: Key Terms

### **Demanda Intermitente** (Intermittent Demand)
**What it is:** Products that sell unpredictably—with long periods of no sales, then sudden spikes.

**Real example:** 
- Specialty cleaning gel: doesn't sell for weeks, then suddenly someone buys 5 bottles

**Why it matters:** 
Classic inventory formulas assume regular sales. They break when demand is this unpredictable.

**In the dissertation:** Central problem we solve.

---

### **Política de Reposição** (Replenishment Policy)
**What it is:** A decision rule that answers: "When should I order? How much should I order?"

**Examples:**
- EOQ: Classic formula (order a fixed amount when stock drops below a level)
- Newsvendor: Order based on expected demand (good for trendy products)
- GA-DQN: Machine learning approach (learns from past decisions)

**Real analogy:**
Different stores use different "policies":
- Coffee shop: Order when stock drops to 50%
- Bookstore: Order based on seasonal trends  
- Convenience store: Order large quantities infrequently

**Why it matters:** Wrong policy = wasted money and empty shelves.

---

### **Syntetos-Boylan Classification** (ADI × CV²)
**What it is:** A way to categorize products by two measures:
- **ADI** (Average Demand Interval): How often does this product sell? (Once a week? Once a month?)
- **CV²** (Coefficient of Variation squared): When it does sell, how variable is the amount?

**Creates 4 categories:**
| Category | Pattern | Example | Best Approach |
|---|---|---|---|
| **Smooth** | Regular, predictable | Daily coffee sales | Classic formulas work |
| **Erratic** | Frequent but unpredictable amounts | Popular but variable items | Modified formulas |
| **Intermittent** | Rare but stable amounts | Seasonal specialty items | Special forecasting |
| **Lumpy** | Rare AND unpredictable amounts | Specialty cleaners, niche products | Machine learning needed |

**Why it matters:** Different categories need different strategies. This dissertation handles the hardest category (Lumpy).

---

### **Bullwhip Effect** (Efeito Chicote)
**What it is:** Demand variability gets BIGGER as you move up the supply chain.

**How it happens:**
```
Store sees small demand fluctuation
    ↓
Store orders slightly more/less from warehouse
    ↓
Warehouse sees THIS as huge variation (from all stores)
    ↓
Warehouse orders hugely different amounts from factory
    ↓
Factory sees wild swings in demand
```

**Real world example:**
- 5% fewer customers at Store A means that store orders 20% less
- When multiplied across 100 stores, the warehouse sees demand drop 50%
- Factory responds by cutting production sharply, then overcompensating

**Result:** Excess inventory in some places, shortages in others, unused capacity.

**Why it matters:** A replenishment policy that amplifies this effect hurts the entire supply chain, even if it looks good for one store.

---

### **CTI: Custo Total de Inventário** (Total Inventory Cost)
**What it includes:**
1. **Holding cost:** Money tied up in inventory on shelves
2. **Ordering cost:** Transportation, administrative work, etc.
3. **Shortage cost:** Lost sales when products are out of stock
4. **Waste cost:** Products that expire or become obsolete

**Why it matters:** Our system minimizes this cost. Lower CTI = more money saved.

---

### **Nível de Serviço** (Service Level / Shelf Availability)
**What it is:** Percentage of customer demand that's met (not out of stock).

**Calculation:** 
```
Service Level = (Units sold) / (Units customers wanted to buy)
```

**Examples:**
- 95% service level: 1 out of 20 customer requests fail (out of stock)
- 98% service level: 1 out of 50 customer requests fail
- 99% service level: 1 out of 100 customer requests fail

**Why it matters:** Too low → customers unhappy and shop elsewhere. Too high → costs too much (excessive inventory).

**The challenge:** Minimize cost AND keep service level high. Our system does this better than traditional approaches.

---

### **Taxa de Ruptura** (Stockout Rate)
**What it is:** How often the store runs out of stock.

**Example:**
- If a product is out of stock 2 out of every 20 review periods = 10% stockout rate

**Why it matters:** Every stockout is a lost sale and disappointed customer.

---

### **GA-RL: Hybrid Architecture** (Genetic Algorithm + Reinforcement Learning)
**What it is:** A combination of two machine learning techniques:

**Genetic Algorithm (GA):**
- Mimics evolution: "test many solutions, keep the best, breed them together"
- Good at exploring broadly, finding good regions quickly
- But slow at fine-tuning

**Reinforcement Learning (RL/DQN):**
- Agent learns by trying actions and getting rewards
- Good at fine-tuning a solution
- But slow at exploring, can get stuck

**How they combine (GA-RL):**
1. GA finds broadly good solutions (fast exploration)
2. DQN starts from GA's good solutions and fine-tunes them (local optimization)
3. Result: Finds better solutions faster than either alone

**Why it's important:** Traditional RL needs millions of tries to learn. On real sales data (only 38 weeks of history), that's impossible. GA-RL solves this.

---

### **Policy Selection Engine (PSE)** / **AIPE**
**What it is:** The intelligent "recommendation system" that answers: "Which replenishment strategy should this product use?"

**How it works:**
1. Analyzes the product's sales pattern
2. Checks its historical catalog of strategies
3. Recommends the best one (like Netflix recommends movies)

**Why "policy selection"?** Instead of saying "all products use Strategy X," we say "each product uses the strategy that's best for IT."

**Why it matters:** Different products need different strategies. This system learns which is which.

---

### **Features Operacionais** (Operational Features)
**What they are:** Measurable characteristics of a product's sales pattern.

**Examples of features:**
- How often does it sell? (frequency)
- When it sells, how much? (volume)
- Is it seasonal? (predictable pattern?)
- How unpredictable is it? (volatility)
- Any recent trends? (is demand changing?)

**Why we measure them:** To predict which strategy will work best.

**Real analogy:** Just like doctors measure blood pressure and cholesterol to recommend treatment, we measure demand "vital signs" to recommend inventory strategies.

---

### **Validação Estatística** (Statistical Validation)
**What it is:** Rigorous testing to make sure observed differences aren't just luck/coincidence.

**The problem:** If we test Strategy A vs. Strategy B and see Strategy A is 2% cheaper, is that real or just random variation?

**The solution:** Statistical tests (Wilcoxon, Friedman) answer: "How confident are we that A is truly better than B?"

**Example of the tests used:**
- **Wilcoxon test:** Compares two strategies, accounts for randomness
- **Friedman test:** Compares many strategies at once
- **Cohen's d:** Measures "how much better" one strategy is

**Why it matters:** In supply chain, a 2% difference matters. We don't just show numbers—we prove they're real.

---

### **Meta-modelo** (Meta-Model / Master Model)
**What it is:** A model that learns from OTHER models' performance.

**Analogy:** 
- Regular model: "Based on weather data, predict tomorrow's temperature"
- Meta-model: "Based on how well weather models performed, choose the best weather model for this situation"

**In this dissertation:**
- Regular models: The 12 replenishment strategies
- Meta-model: The PSE (chooses which strategy to use)

**Why it's clever:** Instead of saying "all products use the same strategy," we say "use data to choose the best strategy for each product."

---

### **Walk-Forward Validation** (Validação Prospectiva)
**What it is:** Testing a model on data it hasn't seen before, in a realistic way.

**How it works:**
1. Train on data from Week 1-10
2. Test on Week 11
3. Train on data from Week 1-11
4. Test on Week 12
5. Repeat...

**Why it matters:** Prevents "cheating"—models trained on data they then test on always look artificially good. This tests real-world performance.

**The guarantee:** Results you see are realistic, not inflated.

---

## How to Read This Glossary

**First read:** Start here if you're not familiar with inventory management

**Reference:** Use this while reading the dissertation when you encounter a term you don't know

**Check yourself:** After finishing the dissertation, you should be able to explain each term in simple language

---

## Progression: From Simple to Complex

If you want to build understanding gradually, read glossary entries in this order:

**Basic (Essential):**
1. Demanda Intermitente
2. Política de Reposição
3. CTI
4. Nível de Serviço

**Intermediate (Important):**
5. Syntetos-Boylan
6. Bullwhip Effect
7. Features Operacionais

**Advanced (Details):**
8. GA-RL
9. Policy Selection Engine
10. Meta-modelo
11. Validação Estatística
12. Walk-Forward Validation

---

## Quick Summary Table

| Concept | Simple Definition | Why Important |
|---|---|---|
| Intermittent Demand | Products that sell unpredictably | This is the problem we solve |
| Replenishment Policy | "When to order?" decision rule | Different products need different rules |
| Syntetos-Boylan | 4-category classification of products | Groups products with similar characteristics |
| Bullwhip Effect | Demand variability grows up the supply chain | Bad policies hurt everyone, not just one store |
| CTI | Total inventory cost | Lower cost = better solution |
| Service Level | % of demand met (no stockout) | Can't sacrifice customer satisfaction |
| GA-RL | ML approach combining evolution + learning | Solves practical constraints of real data |
| PSE/AIPE | Intelligent strategy selector | Automatically picks best policy per product |
| Statistical Validation | Proof that results aren't just chance | Real evidence, not just numbers |

