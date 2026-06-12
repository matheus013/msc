# Implementation Plan: Making Your Dissertation Accessible

## Status: Completed & In Progress

✅ **Done:**
- Abstract (tcc.tex) completely rewritten with plain language
- Comprehensive accessible markdown version created

🔄 **In Progress:**
- Integration of accessible version into LaTeX files

---

## Step-by-Step Implementation Guide

### Phase 1: Introduction Section (capitulos/introducao.tex)

#### Current Issue
The introduction starts with dense, technical language that assumes reader familiarity with supply chain concepts.

**Current opening (lines 8-10):**
```latex
A gestão integrada do abastecimento em redes varejistas geograficamente dispersas 
constitui um problema central da engenharia de operações, pois condiciona simultaneamente 
(i) o nível de serviço percebido na ponta da cadeia...
```

#### Proposed Solution: Two-Tier Approach

**Option A: Full Rewrite (Recommended)**
Replace dense opening with concrete retail example, then transition to technical details.

**Option B: Parallel Version**
Keep original text but add a new "Executive Summary" section before "Contexto, relevância..." that explains in plain language.

#### Implementation Instructions

If you choose **Option A** (Full rewrite), here's how to edit LaTeX files manually:

1. **Open** `capitulos/introducao.tex` in your editor
2. **Find** the line starting with "A gestão integrada do abastecimento..."
3. **Replace** with content from `INTRO_ACCESSIBLE_VERSION.md` → "Context, Relevance, and Research Gap" section
4. **Adapt** citations to maintain LaTeX `\cite{}` format

**Example Translation:**
```latex
% OLD (Technical)
A gestão integrada do abastecimento em redes varejistas geograficamente dispersas 
constitui um problema central...

% NEW (Accessible)
Uma rede varejista com centenas de lojas enfrenta uma pergunta que se repete toda 
semana: quanto de cada produto repor? O desafio é que cada produto vende diferente. 
Alguns vendem regularmente, outros raramente mas em quantidade grande. Cada padrão 
exige uma estratégia diferente, mas não existe fórmula única para todos.
```

### Phase 2: Concepts Chapter (capitulos/conceitos.tex)

**Goal:** Add plain-language explanations before diving into formal definitions.

**Suggested structure:**
```latex
\section{Termos-Chave e Conceitos Fundamentais}

% NEW: Plain language intro
Esta seção explica os conceitos principais usados nesta dissertação. 
Se você já está familiarizado com gestão de inventário, pode pular 
para a Seção~\ref{sec:avancado}.

\subsection{Demanda Intermitente: Por Que Alguns Produtos São Imprevisíveis?}

Imagine dois produtos em uma loja de varejo:
\begin{itemize}
  \item \textbf{Xampu:} Vende praticamente todo dia, quantidade previsível
  \item \textbf{Gel de limpeza especial:} Vende uma vez a cada 3 meses, mas 10 unidades de uma vez
\end{itemize}

Produtos assim são chamados de \textit{intermitentes}...

% THEN: Formal definition
\subsubsection{Definição Formal}

Formalmente, denomina-se \textit{intermitente} a demanda caracterizada por...
```

---

### Phase 3: Results Chapter (capitulos/resultados.tex)

**Goal:** Frame results for both technical and general audiences.

**Suggested change:**
```latex
% Add at chapter start
\section*{Para o Gestor: Resultados em Linguagem Prática}

O sistema recomendou estratégias que reduziram custos em até 48\% 
enquanto mantinham a disponibilidade de produtos nos prateleiros. 
Descobrimos também que não existe uma estratégia única melhor para todos 
os produtos: a estratégia ótima varia conforme o padrão de venda de cada um.

\section{Para o Especialista: Análise Técnica Detalhada}

% Original technical results...
```

---

## Tools for Simplifying LaTeX

### Easy Edits to Prioritize

1. **Abstract** ✅ Already done - check `tcc.tex` around line 89
2. **First 2 paragraphs of Introduction** - Replace jargon with concrete examples
3. **Add glossary boxes** - For technical terms used first time
4. **Add figures early** - Visual example of 3 demand patterns before formal definition

### Recommended Visual Aids to Add

```latex
% Add after Introduction opening
\begin{figure}[h]
\centering
\begin{tabular}{|c|c|c|}
\hline
\textbf{Product Type} & \textbf{Sales Pattern} & \textbf{Best Strategy} \\
\hline
Regular (Shampoo) & Predictable, every day & Formula EOQ \\
Intermittent (Specialty Gel) & Rare + Large spike & Strategy A \\
Seasonal (Holiday item) & Regular, seasonal spike & Strategy B \\
\hline
\end{tabular}
\caption{Different products need different strategies. 
This dissertation learns which strategy fits each product.}
\label{fig:intro_strategies}
\end{figure}
```

---

## Readability Checklist

After each section rewrite, verify these criteria:

- [ ] **Jargon Test:** Can I find 3 technical terms used without explanation?
- [ ] **First Impression:** Does the first paragraph tell a non-specialist what problem you solve?
- [ ] **Metaphor Test:** Are there at least 2 concrete examples (shampoo, specialty items)?
- [ ] **Structure Test:** Is there a clear flow from Problem → Why It Matters → How You Solve It?
- [ ] **Elevator Pitch:** Can I explain the entire dissertation in 1 paragraph?

---

## Elevator Pitch for Your Dissertation

**The One-Paragraph Version:**

"Different retail products sell differently—some regularly, others unpredictably. A standard rule for deciding inventory levels doesn't work for all products. My research builds an intelligent system that learns which replenishment strategy works best for each product based on its unique sales pattern. The system was tested on 15 stores with unpredictable products and reduced costs by 48% while maintaining service levels, showing that there's no one-size-fits-all solution—but there is a smarter way to choose."

---

## Next Steps

1. **Review** `INTRO_ACCESSIBLE_VERSION.md` carefully
2. **Choose** which sections to rewrite (full vs. parallel approach)
3. **Edit** LaTeX files using the translation guide above
4. **Test** readability with the checklist
5. **Add** visual elements (figures, glossary boxes)
6. **Share** with non-specialist readers for feedback

---

## Files Created/Modified

| File | Status | Purpose |
|---|---|---|
| `tcc.tex` (Abstract) | ✅ Modified | New abstract in plain language |
| `INTRO_ACCESSIBLE_VERSION.md` | ✅ Created | Full accessible version for reference |
| `REWRITE_IMPLEMENTATION_PLAN.md` | ✅ Created | This guide |
| `capitulos/introducao.tex` | ⏳ Pending | To be updated with accessible opening |
| `capitulos/conceitos.tex` | ⏳ Pending | To add plain-language explanations |

---

## Questions to Guide Your Edits

As you rewrite, ask:

1. **Would a retail manager understand this sentence?** If no → simplify
2. **Is the jargon explained?** If no → add 1-2 words of explanation
3. **Is there a concrete example?** If no → add one
4. **Does this serve the story?** If no → consider removing or moving to appendix

---

## Tips for Effective Simplification

✅ **DO:**
- Use active voice ("the system learns" vs. "learning occurs")
- Start with problems, not methods
- Use numbers and concrete examples
- Build from familiar to unfamiliar
- Explain why before how

❌ **DON'T:**
- Remove all technical detail (keep rigor)
- Change meaning for sake of simplicity
- Assume reader knows jargon
- Make sentences too long
- Hide important findings behind complexity

