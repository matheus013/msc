"""
metaheuristics_torch.py — GA, SA, PSO e DE vetorizados em PyTorch.

Substituem as versões que dependiam de DEAP e scipy. A mudança central não é a
biblioteca: é que a população inteira passou a ser avaliada em UMA chamada ao
simulador vetorizado, em vez de um laço Python por indivíduo. Com população de
100 e 50 gerações, isso troca 5.000 simulações sequenciais por 50 simulações
de largura 100 — o que efetivamente aproveita GPU.

Todas as quatro compartilham:
  - a mesma representação  theta = (ROP, Q, SS)
  - o mesmo simulador      BatchInventoryEnv
  - a mesma função objetivo, a formulação restrita da Equação (4.2):
        min CTI(theta)  s.a.  NS(theta) >= alpha_min
  - os mesmos limites operacionais por série

Isso preserva o requisito metodológico da proposta: os algoritmos diferem
apenas no mecanismo de busca, não no que está sendo medido.

Nota sobre estado da arte: conforme levantado em
docs/references/estado_da_arte_politicas.md (Seção 3.2), NÃO existe estado da
arte publicado em veículo relevante para meta-heurísticas aplicadas a
inventário — a frente teórica migrou para DRL. Estas implementações são,
portanto, as formulações canônicas (Holland 1992; Kirkpatrick et al. 1983;
Kennedy & Eberhart 1995; Storn & Price 1997) com a função objetivo corrigida
e execução vetorizada. Não se afirma upgrade para estado da arte aqui, porque
não há um para onde subir.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from simulation.core.device import default_dtype, get_device_for_batch
from simulation.core.inventory_env_torch import BatchInventoryEnv, fitness_cost

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Base comum
# ═══════════════════════════════════════════════════════════════════════════

class _ThresholdOptimizer:
    """
    Infraestrutura compartilhada: limites de busca, avaliação vetorizada da
    população e construção da política final.
    """

    name = "base"

    def __init__(self, demand, cfg: dict, device=None):
        self.demand = np.asarray(demand, dtype=float)
        self.cfg = cfg
        gcfg = cfg.get("GENETIC_ALGORITHM", {})
        # Padrão "zabraoui": aptidão da Eq. (3) do artigo-base, lambda1*SL - lambda2*TIC.
        # Alternativa "constrained": Eq. (4.2) da proposta, min CTI s.a. NS >= alpha_min.
        self.fitness_mode = str(gcfg.get("fitness_mode", "zabraoui"))
        w = gcfg.get("fitness_weights", [1.0, 1e-4])
        self.lambda1, self.lambda2 = float(w[0]), float(w[1])
        self.alpha_min = float(gcfg.get("alpha_min", cfg.get("alpha_min", 0.70)))
        self.penalty_w = float(gcfg.get("penalty_weight", 10.0))
        # O lote é o tamanho da população: é ele que decide se GPU compensa.
        self.device = device if device is not None else get_device_for_batch(
            self._population_hint(cfg),
            cfg.get("SIMULATION", {}).get("device", "auto"))
        self.dtype = default_dtype(self.device)
        self.lb, self.ub = self._bounds()
        self.best = None
        self.cost_history: list[float] = []
        self._env_cache: dict[int, BatchInventoryEnv] = {}

    @staticmethod
    def _population_hint(cfg: dict) -> int:
        """Maior lote que este otimizador vai avaliar de uma vez."""
        return max(
            int(cfg.get("GENETIC_ALGORITHM", {}).get("population_size", 100)),
            int(cfg.get("SA", {}).get("n_chains", 32)),
            int(cfg.get("PSO", {}).get("n_particles", 40)),
            int(cfg.get("DE", {}).get("population_size", 15)) * 3,
        )

    def _bounds(self):
        cfg, d = self.cfg, self.demand
        sp = cfg.get("GENETIC_ALGORITHM", {}).get("search_space", {})
        mu = float(np.mean(d)) if d.size else 0.0
        sd = float(np.std(d)) if d.size else 0.0
        lt = cfg["SIMULATION"].get("lead_time", 2)
        z = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
        K = cfg["COST"].get("ordering_cost_per_order", 50.0)
        h = cfg["COST"].get("holding_cost_per_unit", 1.0)

        q_eoq = float(np.sqrt(2 * mu * max(d.size, 1) * K / max(h, 1e-6)))
        ss_ref = z * sd * np.sqrt(lt)
        rop_ref = mu * lt + ss_ref

        hi = [
            max(sp.get("ROP", [0, 2000])[1], rop_ref * 2),
            max(sp.get("Q", [1, 2000])[1], q_eoq * 2),
            max(sp.get("SS", [0, 1000])[1], ss_ref * 2),
        ]
        lo = [0.0, 1.0, 0.0]
        t = lambda v: torch.tensor(v, dtype=self.dtype, device=self.device)
        return t(lo), t(hi)

    def _env(self, batch: int) -> BatchInventoryEnv:
        """Reaproveita o ambiente por tamanho de lote (evita recomputar stats)."""
        if batch not in self._env_cache:
            self._env_cache[batch] = BatchInventoryEnv(
                self.demand, self.cfg, batch_size=batch, device=self.device)
        return self._env_cache[batch]

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """pop (B,3) -> custo (B,), a minimizar. Uma simulação para todo o lote."""
        pop = torch.clamp(pop, self.lb, self.ub)
        env = self._env(pop.shape[0])
        k = env.run_threshold(pop[:, 0], torch.clamp(pop[:, 1], min=1.0), pop[:, 2])
        return fitness_cost(k, self.fitness_mode,
                            lambda1=self.lambda1, lambda2=self.lambda2,
                            alpha_min=self.alpha_min, penalty_weight=self.penalty_w)

    def _rand(self, n: int, gen: torch.Generator) -> torch.Tensor:
        u = torch.rand((n, 3), dtype=self.dtype, device=self.device, generator=gen)
        return self.lb + u * (self.ub - self.lb)

    def _finalize(self, theta: torch.Tensor, cost: float):
        theta = torch.clamp(theta, self.lb, self.ub)
        self.best = {
            "ROP": float(theta[0].item()),
            "Q": float(max(1.0, theta[1].item())),
            "SS": float(theta[2].item()),
            "cost": float(cost),
            "fitness": -float(cost),          # compat: pipeline espera maximizar
            "cost_history": list(self.cost_history),
            "fitness_history": [-c for c in self.cost_history],
        }
        log.info("[%s] ROP=%.1f Q=%.1f SS=%.1f cost=%.1f", self.name,
                 self.best["ROP"], self.best["Q"], self.best["SS"], cost)
        return self.best

    def make_policy(self):
        """Política escalar, para avaliação no `InventoryEnv` de referência."""
        ROP, Q, SS = self.best["ROP"], self.best["Q"], self.best["SS"]
        thr = ROP + SS

        def p(state, env):
            return Q if env.inventory + sum(env.pipeline) <= thr else 0.0
        return p

    def _generator(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g


# ═══════════════════════════════════════════════════════════════════════════
# 1. Algoritmo Genético
# ═══════════════════════════════════════════════════════════════════════════

class TorchGA(_ThresholdOptimizer):
    """
    GA com seleção por torneio, crossover blend e mutação gaussiana.

    Acrescenta elitismo em relação à versão DEAP anterior: o melhor indivíduo
    sobrevive intacto a cada geração. Sem elitismo, o crossover blend podia
    destruir a melhor solução encontrada — o que explica parte da instabilidade
    do GA entre séries nos resultados anteriores.
    """

    name = "GA"

    def __init__(self, demand, cfg, device=None):
        super().__init__(demand, cfg, device)
        g = cfg.get("GENETIC_ALGORITHM", {})
        self.pop_n = int(g.get("population_size", 100))
        self.n_gen = int(g.get("n_generations", 50))
        self.cx_p = float(g.get("crossover_prob", 0.9))
        self.mut_p = float(g.get("mutation_rate", 0.05))
        self.tourn = int(g.get("tournament_size", 3))
        self.seed = int(g.get("seed", 42))

    def optimize(self, verbose: bool = False):
        gen = self._generator(self.seed)
        pop = self._rand(self.pop_n, gen)
        cost = self.evaluate(pop)
        span = (self.ub - self.lb) / 10.0

        best_i = int(torch.argmin(cost).item())
        best_theta, best_cost = pop[best_i].clone(), float(cost[best_i].item())

        for it in range(self.n_gen):
            # seleção por torneio, vetorizada
            idx = torch.randint(0, self.pop_n, (self.pop_n, self.tourn),
                                device=self.device, generator=gen)
            winners = idx.gather(1, torch.argmin(cost[idx], dim=1, keepdim=True)).squeeze(1)
            off = pop[winners].clone()

            # crossover blend em pares
            half = self.pop_n // 2
            a, b = off[:half], off[half:2 * half]
            do_cx = (torch.rand(half, device=self.device, generator=gen) < self.cx_p)
            alpha = torch.rand((half, 3), dtype=self.dtype,
                               device=self.device, generator=gen)
            na = torch.where(do_cx.unsqueeze(1), alpha * a + (1 - alpha) * b, a)
            nb = torch.where(do_cx.unsqueeze(1), alpha * b + (1 - alpha) * a, b)
            off[:half], off[half:2 * half] = na, nb

            # mutação gaussiana
            mask = (torch.rand((self.pop_n, 3), device=self.device,
                               generator=gen) < self.mut_p)
            noise = torch.randn((self.pop_n, 3), dtype=self.dtype,
                                device=self.device, generator=gen) * span
            off = torch.clamp(off + mask * noise, self.lb, self.ub)

            off_cost = self.evaluate(off)

            # elitismo: preserva o melhor histórico
            worst = int(torch.argmax(off_cost).item())
            off[worst], off_cost[worst] = best_theta, best_cost

            pop, cost = off, off_cost
            i = int(torch.argmin(cost).item())
            if float(cost[i].item()) < best_cost:
                best_theta, best_cost = pop[i].clone(), float(cost[i].item())
            self.cost_history.append(best_cost)
            if verbose and (it + 1) % 10 == 0:
                log.info("  [GA] gen %d/%d best=%.1f", it + 1, self.n_gen, best_cost)

        return self._finalize(best_theta, best_cost)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Simulated Annealing
# ═══════════════════════════════════════════════════════════════════════════

class TorchSA(_ThresholdOptimizer):
    """
    Simulated Annealing com cadeias paralelas.

    O SA canônico é sequencial. Para aproveitar a vetorização sem descaracterizar
    o método, executam-se `n_chains` cadeias independentes com o MESMO
    cronograma de temperatura (T0=1000, alpha=0.95, conforme Seção 4.4 da
    proposta), avaliadas em paralelo; a saída é a melhor solução entre elas.

    Cada cadeia isolada segue exatamente o critério de Metropolis:
        aceita se dC < 0, ou com probabilidade exp(-dC / T_k)

    A diferença em relação a `scipy.dual_annealing` usado antes é que agora o
    cronograma e o critério são explícitos e auditáveis, em vez de delegados a
    uma implementação de caixa preta com parâmetros próprios.
    """

    name = "SA"

    def __init__(self, demand, cfg, device=None):
        super().__init__(demand, cfg, device)
        s = cfg.get("SA", {})
        self.T0 = float(s.get("initial_temp", 1000.0))
        self.cooling = float(s.get("cooling_rate", 0.95))
        self.n_iter = int(s.get("maxiter", s.get("max_iter", 500)))
        self.n_chains = int(s.get("n_chains", 32))
        self.seed = int(s.get("seed", 42))

    def optimize(self, verbose: bool = False):
        gen = self._generator(self.seed)
        cur = self._rand(self.n_chains, gen)
        cur_cost = self.evaluate(cur)
        step = (self.ub - self.lb) / 10.0

        bi = int(torch.argmin(cur_cost).item())
        best_theta, best_cost = cur[bi].clone(), float(cur_cost[bi].item())

        T = self.T0
        for it in range(self.n_iter):
            noise = torch.randn((self.n_chains, 3), dtype=self.dtype,
                                device=self.device, generator=gen) * step
            cand = torch.clamp(cur + noise, self.lb, self.ub)
            cand_cost = self.evaluate(cand)

            delta = cand_cost - cur_cost
            # Metropolis: aceita melhora sempre; piora com prob. exp(-delta/T)
            u = torch.rand(self.n_chains, dtype=self.dtype,
                           device=self.device, generator=gen)
            accept = (delta < 0) | (u < torch.exp(-delta / max(T, 1e-9)))
            cur = torch.where(accept.unsqueeze(1), cand, cur)
            cur_cost = torch.where(accept, cand_cost, cur_cost)

            i = int(torch.argmin(cur_cost).item())
            if float(cur_cost[i].item()) < best_cost:
                best_theta, best_cost = cur[i].clone(), float(cur_cost[i].item())
            self.cost_history.append(best_cost)

            T *= self.cooling
            if verbose and (it + 1) % 100 == 0:
                log.info("  [SA] iter %d/%d T=%.1f best=%.1f",
                         it + 1, self.n_iter, T, best_cost)

        return self._finalize(best_theta, best_cost)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Particle Swarm Optimization
# ═══════════════════════════════════════════════════════════════════════════

class TorchPSO(_ThresholdOptimizer):
    """
    PSO com fator de constrição de Clerc & Kennedy (2002).

    A versão anterior usava w=0.7, c1=c2=1.5 sem constrição, combinação em que
    a velocidade pode divergir; observava-se isso nos resultados como Q colando
    no limite superior do espaço de busca. O fator de constrição

        chi = 2 / |2 - phi - sqrt(phi^2 - 4 phi)|,   phi = c1 + c2 > 4

    garante convergência sem necessidade de clamp arbitrário de velocidade.
    Ativável/desativável por `use_constriction`.
    """

    name = "PSO"

    def __init__(self, demand, cfg, device=None):
        super().__init__(demand, cfg, device)
        p = cfg.get("PSO", {})
        self.n_part = int(p.get("n_particles", 40))
        self.n_iter = int(p.get("n_iterations", p.get("iterations", 80)))
        self.w = float(p.get("inertia", 0.7))
        self.c1 = float(p.get("cognitive", 1.5))
        self.c2 = float(p.get("social", 1.5))
        self.seed = int(p.get("seed", 42))
        self.use_constriction = bool(p.get("use_constriction", True))
        if self.use_constriction:
            phi = max(self.c1 + self.c2, 4.001)
            self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        else:
            self.chi = 1.0

    def optimize(self, verbose: bool = False):
        gen = self._generator(self.seed)
        pos = self._rand(self.n_part, gen)
        vel = torch.zeros_like(pos)
        cost = self.evaluate(pos)

        pbest, pbest_cost = pos.clone(), cost.clone()
        gi = int(torch.argmin(cost).item())
        gbest, gbest_cost = pos[gi].clone(), float(cost[gi].item())

        for it in range(self.n_iter):
            r1 = torch.rand((self.n_part, 3), dtype=self.dtype,
                            device=self.device, generator=gen)
            r2 = torch.rand((self.n_part, 3), dtype=self.dtype,
                            device=self.device, generator=gen)
            vel = self.chi * (self.w * vel
                              + self.c1 * r1 * (pbest - pos)
                              + self.c2 * r2 * (gbest.unsqueeze(0) - pos))
            pos = torch.clamp(pos + vel, self.lb, self.ub)
            cost = self.evaluate(pos)

            improved = cost < pbest_cost
            pbest = torch.where(improved.unsqueeze(1), pos, pbest)
            pbest_cost = torch.where(improved, cost, pbest_cost)

            i = int(torch.argmin(pbest_cost).item())
            if float(pbest_cost[i].item()) < gbest_cost:
                gbest, gbest_cost = pbest[i].clone(), float(pbest_cost[i].item())
            self.cost_history.append(gbest_cost)
            if verbose and (it + 1) % 20 == 0:
                log.info("  [PSO] iter %d/%d best=%.1f", it + 1, self.n_iter, gbest_cost)

        return self._finalize(gbest, gbest_cost)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Differential Evolution
# ═══════════════════════════════════════════════════════════════════════════

class TorchDE(_ThresholdOptimizer):
    """
    DE/rand/1/bin (Storn & Price, 1997), com F e CR conforme Seção 4.4.

        v_i = x_r1 + F (x_r2 - x_r3)
        u_i = crossover binomial(v_i, x_i) com taxa CR
        seleção greedy elemento a elemento

    Toda a geração é avaliada em uma chamada, e a seleção greedy é um
    `torch.where` — não há laço Python sobre indivíduos.
    """

    name = "DE"

    def __init__(self, demand, cfg, device=None):
        super().__init__(demand, cfg, device)
        d = cfg.get("DE", {})
        self.pop_n = int(d.get("population_size", d.get("popsize", 15)) * 3)
        self.n_iter = int(d.get("maxiter", d.get("max_iter", 100)))
        mut = d.get("mutation", 0.8)
        self.F = float(mut[0] if isinstance(mut, (list, tuple)) else mut)
        self.CR = float(d.get("recombination", 0.9))
        self.seed = int(d.get("seed", 42))

    def optimize(self, verbose: bool = False):
        gen = self._generator(self.seed)
        n = max(self.pop_n, 8)
        pop = self._rand(n, gen)
        cost = self.evaluate(pop)
        bi = int(torch.argmin(cost).item())
        best_theta, best_cost = pop[bi].clone(), float(cost[bi].item())

        for it in range(self.n_iter):
            # três índices distintos por indivíduo
            r = torch.randint(0, n, (3, n), device=self.device, generator=gen)
            mutant = pop[r[0]] + self.F * (pop[r[1]] - pop[r[2]])

            cross = torch.rand((n, 3), dtype=self.dtype,
                               device=self.device, generator=gen) < self.CR
            # garante ao menos uma dimensão herdada do mutante
            jrand = torch.randint(0, 3, (n,), device=self.device, generator=gen)
            cross[torch.arange(n, device=self.device), jrand] = True

            trial = torch.clamp(torch.where(cross, mutant, pop), self.lb, self.ub)
            trial_cost = self.evaluate(trial)

            better = trial_cost < cost
            pop = torch.where(better.unsqueeze(1), trial, pop)
            cost = torch.where(better, trial_cost, cost)

            i = int(torch.argmin(cost).item())
            if float(cost[i].item()) < best_cost:
                best_theta, best_cost = pop[i].clone(), float(cost[i].item())
            self.cost_history.append(best_cost)
            if verbose and (it + 1) % 20 == 0:
                log.info("  [DE] iter %d/%d best=%.1f", it + 1, self.n_iter, best_cost)

        return self._finalize(best_theta, best_cost)


# Aliases de compatibilidade com o pipeline existente
GAPolicyOptimizer = TorchGA
SimulatedAnnealingPolicy = TorchSA
PSOPolicy = TorchPSO
DEPolicy = TorchDE
