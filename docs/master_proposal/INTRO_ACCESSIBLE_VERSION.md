# Dissertation: Accessible Language Version

## Abstract (Simplified, Accessible Version)

### The Problem
A retail network with hundreds of stores across different cities faces a recurring question: **how much of each product should we replenish each week?**

The challenge is that each product sells differently:
- Some sell predictably every day (shampoo, soap)
- Others sell rarely but in large quantities when they do (specialty cleaning gel)
- Some follow seasonal patterns (holiday items, seasonal products)

Each pattern requires a different replenishment strategy. Using the same rule for all products and stores results in:
- **Overstocked items** (tied-up capital, waste risk)
- **Frequent stockouts** (lost sales, customer frustration)
- **Wasted opportunities** to reduce costs

**The Impact:** Inefficient inventory management costs 3.5% to 8% of annual revenue.

### The Solution
This dissertation proposes **an intelligent system that automatically learns which replenishment strategy works best for each product in each store.**

The system:
1. **Analyzes** historical sales patterns (frequency, volume, seasonality)
2. **Tests** 12 different replenishment strategies (from classic formulas to machine learning algorithms)
3. **Validates** results rigorously with statistical testing
4. **Recommends** the best strategy automatically for each product profile

### Initial Results
Tested in 15 stores in Paraíba with highly unpredictable products, the system recommended strategies that:
- **Reduced total costs by up to 48%**
- **Maintained shelf availability** at equivalent levels
- **Reduced supply chain variability amplification** by up to 24x

**Key discovery:** There is no universally best strategy. The best choice varies by sales pattern and volume. Mid-volume stores benefit from one strategy; low-volume stores benefit from another.

### Scale & Rigor
- **Dataset:** ~48 million real sales transactions
- **Products:** ~12,000 different SKUs
- **Geographic scope:** Multiple Brazilian states
- **Validation:** Rigorous statistical tests to ensure results aren't due to chance
- **Generalization:** Machine learning to apply findings to new, unseen sales patterns

**Keywords:** Inventory Optimization, Unpredictable Demand, Automatic Strategy Selection, Machine Learning, Demand Forecasting, Statistical Validation, Supply Chain.

---

## Introduction: Accessible Version

### Context, Relevance, and Research Gap

#### **Why Choosing the Right Stock Level is Hard**

A retail network with hundreds of stores faces a question every week: **how much of each product should we replenish?**

It sounds simple in theory, but it's complex in practice. Each store must balance two opposite risks:
- **Too much stock:** Money tied up in inventory, risk of waste or expiration
- **Too little stock:** Customers don't find the product, sales are lost, satisfaction drops

And here's the complication: **each product sells completely differently.**

Some products sell regularly and predictably. Others rarely sell, but when they do, they sell in large, unexpected quantities. Others follow seasonal patterns: huge sales in certain periods, almost nothing in others.

The difficulty multiplies in decentralized networks. Each store decides independently how much to order from the regional warehouse, without perfect coordination. When many stores order at the same time, the warehouse receives a demand spike. When all stores reduce orders, the warehouse is underutilized.

This cascading effect—**amplification of variability**—is known in supply chain management as the *bullwhip effect*: a small sales fluctuation in one store becomes a large wave of variability in the warehouse, and an even larger one at the factory. Result: **costs rise, inventory accumulates in the wrong places, and capacity is wasted.**

The impact is material: inefficient inventory management costs **3.5% to 8% of annual revenue** in tied-up capital and lost sales.

#### **Why Existing Strategies Don't Work for All Products**

Over the decades, operations engineers developed several **decision rules** (called *replenishment policies*) for choosing how much to order. The most famous is the **EOQ formula** (Economic Order Quantity), which works excellently when demand is predictable and regular. Others are more sophisticated, adjusting decisions based on current stock level and recent demand.

But there's a problem: **these classic rules assume demand follows a regular, predictable pattern** with small, foreseeable variations.

In Brazilian regional retail—especially for slow-moving products—this assumption **fails completely.** These products have:
- **Long periods with zero sales**
- **Sporadic demand spikes** of unpredictable magnitude
- **Extreme variability**

This characteristic is called **intermittent demand**: periods of zero sales alternating with irregular spikes. In regional Brazilian retail, this pattern is the **rule, not the exception.** The variability is so extreme that classic formulas like EOQ simply don't work.

And when classic formulas fail, the cost is invisible but real: bad inventory decisions, frequent stockouts, and wasted capital.

#### **Why One Size Doesn't Fit All**

The core insight is simple but powerful: **different products with different demand patterns need different strategies.**

- A product with steady, regular demand might need one approach
- A product with rare, large spikes needs a completely different approach
- A seasonal product needs yet another approach

It's not a matter of finding "the best strategy." It's about finding **the right strategy for each specific product and its unique sales pattern.**

#### **The Central Research Question**

Here's the key question this dissertation addresses:

**Can we build a system that automatically learns which replenishment strategy works best for each product, based on that product's historical sales pattern?**

The hypothesis: Yes. The dissertation proposes the **Adaptive Inventory Policy Engine (AIPE)**—a system that:
1. Analyzes the sales pattern of each product
2. Tests multiple replenishment strategies
3. Learns which strategy works best for that specific pattern
4. Recommends the optimal strategy automatically

This isn't about finding one winning algorithm. It's about understanding that **the optimal policy depends on the context** (the demand pattern), and building an intelligent system that learns this relationship from data.

---

## Translation Guide: Technical Terms to Accessible Language

| Technical Term | Accessible Explanation |
|---|---|
| **Demanda intermitente** | Products that sell unpredictably: long periods of no sales, then sudden spikes |
| **Política de reposição** | Decision rule for when and how much to order |
| **ADI × CV²** (Syntetos-Boylan classification) | A way to categorize products by their sales pattern (frequency vs. irregularity) |
| **Bullwhip effect** | Amplification of demand variability as it moves up the supply chain (store → warehouse → factory) |
| **EOQ** (Economic Order Quantity) | Classic formula for deciding order quantity, assumes regular demand |
| **Custo Total de Inventário (CTI)** | Total cost including: holding cost, stockout cost, ordering cost |
| **Nível de Serviço (NS)** | Percentage of customer demand met (shelf availability) |
| **GA-RL** (Hybrid) | Advanced machine learning combining global optimization (GA) with local refinement (RL) |
| **Policy Selection Engine (PSE)** | The intelligent system that recommends which strategy to use |
| **Features operacionais** | Measurable characteristics of a product's sales pattern |
| **Meta-modelo** | A model that learns from the performance of other models/strategies |

---

## Key Differences: Technical vs. Accessible Version

| Aspect | Technical Version | Accessible Version |
|---|---|---|
| **Opening** | Starts with formal definitions of the problem | Starts with concrete retail scenarios |
| **Jargon** | Uses technical terms freely | Explains technical terms or uses plain language |
| **Motivation** | Assumes reader understands why the problem is hard | Explains why existing solutions fail with examples |
| **Narrative** | Linear, formal structure | Story-like, with concrete examples before abstraction |
| **Audience** | Supply chain specialists, academics | Retail managers, graduate students, educated public |
| **Takeaway** | Methodological contribution to algorithm selection problem | Practical system that saves money and improves service |

