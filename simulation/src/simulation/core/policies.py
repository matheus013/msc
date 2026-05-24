"""
policies.py — 12 Políticas de Inventário para Comparativo Estendido
Todas operam no MESMO cenário fixo (InventoryEnv padronizado).

Políticas implementadas:
  Baseline clássico:
    1. EOQ          — Economic Order Quantity clássico
    2. (s,S)        — Banda min-max com reposição até S
    3. NewsvendorQ  — Newsvendor quantile otimizado por demanda histórica
  Metaheurísticas:
    4. GA           — Algoritmo Genético (DEAP) — ROP, Q, SS
    5. SA           — Simulated Annealing (scipy)
    6. PSO          — Particle Swarm Optimization (numpy puro)
    7. DE           — Differential Evolution (scipy)
  Aprendizado por Reforço:
    8. DQN          — Deep Q-Network
    9. PPO          — Proximal Policy Optimization
   10. SARSA        — On-policy TD learning
  Híbridos:
   11. GA-DQN       — GA inicializa limiar, DQN decide quantidade
   12. GA-PPO       — GA inicializa limiar, PPO decide quantidade
"""
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
from deap import base, creator, tools
import warnings
warnings.filterwarnings("ignore")


def _get_search_bounds(cfg: dict, demand: np.ndarray) -> tuple:
    """
    Retorna (rop_range, q_range, ss_range) escalados pela demanda real.
    """
    sp  = cfg.get("GENETIC_ALGORITHM", {}).get("search_space", {})
    mu  = float(np.mean(demand))
    std = float(np.std(demand))
    lt  = cfg["SIMULATION"].get("lead_time", 2)
    z   = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
    K   = cfg["COST"].get("ordering_cost_per_order", 50.0)
    h   = cfg["COST"].get("holding_cost_per_unit", 1.0)

    q_eoq   = float(np.sqrt(2 * mu * max(len(demand), 1) * K / max(h, 1e-6)))
    ss_ref  = z * std * np.sqrt(lt)
    rop_ref = mu * lt + ss_ref

    rop_r = (0, max(sp.get("ROP", [0, 2000])[1], rop_ref * 2))
    q_r   = (1, max(sp.get("Q",   [1, 2000])[1], q_eoq  * 2))
    ss_r  = (0, max(sp.get("SS",  [0, 1000])[1], ss_ref * 2))
    return rop_r, q_r, ss_r


def _eval_static(demand, cfg, ROP, Q, SS):
    from simulation.core.inventory_env import InventoryEnv
    env = InventoryEnv(demand, cfg, seed=42)
    w   = cfg.get("GENETIC_ALGORITHM", {}).get("fitness_weights", [1.0, 0.0001])

    def policy(state, e):
        return Q if e.inventory + sum(e.pipeline) <= ROP + SS else 0.0

    state = env._state(); done = False
    while not done:
        state, _, done, _ = env.step(policy(state, env))

    k = env.kpis()
    return w[0] * k["ServiceLevel"] - w[1] * k["TIC"], k


# ══════════════════════════════════════════════════════════════
# 1. EOQ
# ══════════════════════════════════════════════════════════════
class EOQPolicy:
    def __init__(self, demand, cfg):
        hcfg = cfg.get("HEURISTIC", {})
        scfg = cfg["SIMULATION"]
        ccfg = cfg["COST"]
        mu   = np.mean(demand); std = np.std(demand)
        lt   = scfg.get("lead_time", 2)
        z    = hcfg.get("z_score", 1.645)
        D    = mu * len(demand)
        K    = ccfg.get("ordering_cost_per_order", 50.0)
        h    = ccfg.get("holding_cost_per_unit", 1.0)
        self.Q  = max(1.0, np.sqrt(2 * D * K / max(h, 0.01)))
        self.SS = z * std * np.sqrt(lt)
        self.ROP= mu * lt + self.SS
        print(f"  [EOQ] Q={self.Q:.1f} | ROP={self.ROP:.1f} | SS={self.SS:.1f}")

    def __call__(self, state, env):
        return self.Q if env.inventory + sum(env.pipeline) <= self.ROP else 0.0


# ══════════════════════════════════════════════════════════════
# 2. Política (s, S) — banda min-max
# ══════════════════════════════════════════════════════════════
class SsPolicyClass:
    def __init__(self, demand, cfg):
        scfg = cfg["SIMULATION"]; ccfg = cfg["COST"]
        mu  = np.mean(demand); std = np.std(demand)
        lt  = scfg.get("lead_time", 2)
        z   = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
        self.s = mu * lt + z * std * np.sqrt(lt)
        self.S = self.s + max(1.0, np.sqrt(
            2 * mu * len(demand) *
            ccfg.get("ordering_cost_per_order", 50.0) /
            max(ccfg.get("holding_cost_per_unit", 1.0), 0.01)))
        print(f"  [(s,S)] s={self.s:.1f} | S={self.S:.1f}")

    def __call__(self, state, env):
        pos = env.inventory + sum(env.pipeline)
        return max(0.0, self.S - pos) if pos <= self.s else 0.0


# ══════════════════════════════════════════════════════════════
# 3. Newsvendor
# ══════════════════════════════════════════════════════════════
class NewsvendorPolicy:
    def __init__(self, demand, cfg):
        ccfg = cfg["COST"]
        cu   = ccfg.get("stockout_cost_per_unit", 5.0)
        co   = ccfg.get("holding_cost_per_unit", 1.0)
        q_cr = cu / (cu + co)
        self.Q_opt = float(np.quantile(demand, q_cr))
        lt = cfg["SIMULATION"].get("lead_time", 2)
        self.ROP   = np.mean(demand) * lt
        print(f"  [Newsvendor] Q*={self.Q_opt:.1f} | CR={q_cr:.2f} | ROP={self.ROP:.1f}")

    def __call__(self, state, env):
        pos = env.inventory + sum(env.pipeline)
        return self.Q_opt if pos <= self.ROP else 0.0


# ══════════════════════════════════════════════════════════════
# 4. Genetic Algorithm
# ══════════════════════════════════════════════════════════════
class GAPolicyOptimizer:
    def __init__(self, demand, cfg):
        self.demand = demand; self.cfg = cfg
        gc = cfg.get("GENETIC_ALGORITHM", {})
        self.rop_r, self.q_r, self.ss_r = _get_search_bounds(cfg, demand)
        self.pop_n = gc.get("population_size", 100)
        self.n_gen = gc.get("n_generations", 50)
        self.cx_p  = gc.get("crossover_prob", 0.8)
        self.mut_r = gc.get("mutation_rate", 0.05)
        self.w     = gc.get("fitness_weights", [1.0, 0.0001])
        self.fitness_history = []
        self.best = None
        self._setup()

    def _setup(self):
        if not hasattr(creator, "FitnessMaxGA"):
            creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualGA"):
            creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)
        tb = base.Toolbox()
        tb.register("individual", lambda: creator.IndividualGA([
            np.random.uniform(*self.rop_r),
            np.random.uniform(*self.q_r),
            np.random.uniform(*self.ss_r)]))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", self._eval)
        tb.register("mate",   tools.cxBlend, alpha=0.5)
        max_range = max(self.rop_r[1] - self.rop_r[0],
                        self.q_r[1]   - self.q_r[0],
                        self.ss_r[1]  - self.ss_r[0])
        sigma = max_range / 10.0
        tb.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=self.mut_r)
        tb.register("select", tools.selTournament, tournsize=3)
        self.tb = tb

    def _eval(self, ind):
        ROP = np.clip(ind[0], *self.rop_r)
        Q   = max(1.0, np.clip(ind[1], *self.q_r))
        SS  = np.clip(ind[2], *self.ss_r)
        fit, _ = _eval_static(self.demand, self.cfg, ROP, Q, SS)
        return (fit,)

    def optimize(self, verbose=True):
        pop = self.tb.population(n=self.pop_n)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda i: i.fitness.values)
        stats.register("max", np.max); stats.register("avg", np.mean)
        print(f"  [GA] {self.pop_n} ind x{self.n_gen} gen")
        for gen in range(self.n_gen):
            off = list(map(self.tb.clone, self.tb.select(pop, len(pop))))
            for i in range(0, len(off)-1, 2):
                if np.random.rand() < self.cx_p:
                    self.tb.mate(off[i], off[i+1])
                    del off[i].fitness.values, off[i+1].fitness.values
            for ind in off:
                if np.random.rand() < self.mut_r:
                    self.tb.mutate(ind); del ind.fitness.values
            inv = [i for i in off if not i.fitness.valid]
            for ind, fit in zip(inv, map(self.tb.evaluate, inv)):
                ind.fitness.values = fit
            pop[:] = off; hof.update(pop)
            rec = stats.compile(pop)
            self.fitness_history.append(rec["max"])
            if verbose and (gen+1) % 10 == 0:
                print(f"    Gen {gen+1:3d}/{self.n_gen} | Best={rec['max']:.4f}")
        b = hof[0]
        self.best = {
            "ROP": np.clip(b[0], *self.rop_r),
            "Q":   max(1.0, np.clip(b[1], *self.q_r)),
            "SS":  np.clip(b[2], *self.ss_r),
            "fitness": b.fitness.values[0],
            "fitness_history": self.fitness_history,
        }
        print(f"  [GA] ROP={self.best['ROP']:.1f} Q={self.best['Q']:.1f} SS={self.best['SS']:.1f}")
        return self.best

    def make_policy(self):
        ROP, Q, SS = self.best["ROP"], self.best["Q"], self.best["SS"]
        def p(state, env):
            return Q if env.inventory + sum(env.pipeline) <= ROP + SS else 0.0
        return p


# ══════════════════════════════════════════════════════════════
# 5. Simulated Annealing
# ══════════════════════════════════════════════════════════════
class SimulatedAnnealingPolicy:
    def __init__(self, demand, cfg):
        self.demand = demand; self.cfg = cfg
        self.best = None
        self.cost_history = []
        self._rop_r, self._q_r, self._ss_r = _get_search_bounds(cfg, demand)

    def optimize(self, verbose=True):
        bounds = [list(self._rop_r), list(self._q_r), list(self._ss_r)]
        print(f"  [SA] Dual Annealing...")
        calls = [0]
        hist  = []

        def objective(x):
            ROP, Q, SS = x
            fit, _ = _eval_static(self.demand, self.cfg,
                                  max(0,ROP), max(1,Q), max(0,SS))
            calls[0] += 1
            hist.append(-fit)
            return -fit

        result = dual_annealing(
            objective, bounds=bounds,
            maxiter=self.cfg.get("SA", {}).get("maxiter", 500),
            seed=42, minimizer_kwargs={"method": "Nelder-Mead"})

        self.cost_history = hist
        ROP, Q, SS = result.x
        self.best = {"ROP": max(0,ROP), "Q": max(1,Q), "SS": max(0,SS),
                     "fitness": -result.fun, "cost_history": hist}
        print(f"  [SA] ROP={self.best['ROP']:.1f} Q={self.best['Q']:.1f} "
              f"SS={self.best['SS']:.1f} | calls={calls[0]}")
        return self.best

    def make_policy(self):
        ROP, Q, SS = self.best["ROP"], self.best["Q"], self.best["SS"]
        def p(state, env):
            return Q if env.inventory + sum(env.pipeline) <= ROP + SS else 0.0
        return p


# ══════════════════════════════════════════════════════════════
# 6. Particle Swarm Optimization
# ══════════════════════════════════════════════════════════════
class PSOPolicy:
    """PSO clássico em numpy puro."""
    def __init__(self, demand, cfg):
        self.demand = demand; self.cfg = cfg
        self.best = None
        self.fitness_history = []
        self._rop_r, self._q_r, self._ss_r = _get_search_bounds(cfg, demand)

    def optimize(self, verbose=True):
        pc = self.cfg.get("PSO", {})
        n_particles = pc.get("n_particles", 40)
        n_iter      = pc.get("n_iterations", 80)
        w           = pc.get("inertia", 0.7)
        c1          = pc.get("cognitive", 1.5)
        c2          = pc.get("social", 1.5)

        lb = np.array([self._rop_r[0], self._q_r[0], self._ss_r[0]], float)
        ub = np.array([self._rop_r[1], self._q_r[1], self._ss_r[1]], float)

        rng = np.random.default_rng(42)
        pos = lb + rng.random((n_particles, 3)) * (ub - lb)
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pbest_fit = np.full(n_particles, -np.inf)
        gbest = pos[0].copy(); gbest_fit = -np.inf

        print(f"  [PSO] {n_particles} partículas x{n_iter} iterações")
        for it in range(n_iter):
            for i in range(n_particles):
                ROP, Q, SS = np.clip(pos[i], lb, ub)
                Q = max(1.0, Q)
                fit, _ = _eval_static(self.demand, self.cfg, ROP, Q, SS)
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit; pbest[i] = pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit; gbest = pos[i].copy()

            r1 = rng.random((n_particles, 3))
            r2 = rng.random((n_particles, 3))
            vel = (w * vel
                   + c1 * r1 * (pbest - pos)
                   + c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, lb, ub)
            self.fitness_history.append(gbest_fit)

            if verbose and (it+1) % 20 == 0:
                print(f"    Iter {it+1:3d}/{n_iter} | Best={gbest_fit:.4f}")

        ROP, Q, SS = np.clip(gbest, lb, ub)
        self.best = {"ROP": float(ROP), "Q": max(1.0, float(Q)), "SS": float(SS),
                     "fitness": gbest_fit, "fitness_history": self.fitness_history}
        print(f"  [PSO] ROP={self.best['ROP']:.1f} Q={self.best['Q']:.1f} SS={self.best['SS']:.1f}")
        return self.best

    def make_policy(self):
        ROP, Q, SS = self.best["ROP"], self.best["Q"], self.best["SS"]
        def p(state, env):
            return Q if env.inventory + sum(env.pipeline) <= ROP + SS else 0.0
        return p


# ══════════════════════════════════════════════════════════════
# 7. Differential Evolution
# ══════════════════════════════════════════════════════════════
class DEPolicy:
    def __init__(self, demand, cfg):
        self.demand = demand; self.cfg = cfg
        self.best = None
        self.fitness_history = []
        self._rop_r, self._q_r, self._ss_r = _get_search_bounds(cfg, demand)

    def optimize(self, verbose=True):
        dec = self.cfg.get("DE", {})
        bounds = [list(self._rop_r), list(self._q_r), list(self._ss_r)]
        calls = [0]; hist = []

        def objective(x):
            ROP, Q, SS = x
            fit, _ = _eval_static(self.demand, self.cfg,
                                  max(0,ROP), max(1,Q), max(0,SS))
            calls[0] += 1; hist.append(-fit)
            return -fit

        print(f"  [DE] Differential Evolution...")
        result = differential_evolution(
            objective, bounds=bounds,
            maxiter=dec.get("maxiter", 100),
            popsize=dec.get("popsize", 15),
            mutation=dec.get("mutation", (0.5, 1.0)),
            recombination=dec.get("recombination", 0.7),
            seed=42, tol=1e-6, polish=True)

        self.fitness_history = hist
        ROP, Q, SS = result.x
        self.best = {"ROP": max(0,ROP), "Q": max(1,Q), "SS": max(0,SS),
                     "fitness": -result.fun, "fitness_history": hist}
        print(f"  [DE] ROP={self.best['ROP']:.1f} Q={self.best['Q']:.1f} "
              f"SS={self.best['SS']:.1f} | calls={calls[0]}")
        return self.best

    def make_policy(self):
        ROP, Q, SS = self.best["ROP"], self.best["Q"], self.best["SS"]
        def p(state, env):
            return Q if env.inventory + sum(env.pipeline) <= ROP + SS else 0.0
        return p


# ══════════════════════════════════════════════════════════════
# REDE NEURAL SIMPLES (compartilhada por DQN, PPO, SARSA)
# ══════════════════════════════════════════════════════════════
class _NN:
    def __init__(self, sizes, lr=0.001):
        self.lr = lr
        self.W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]

    def _relu(self, x): return np.maximum(0, x)
    def _drel(self, x): return (x > 0).astype(float)

    def predict(self, x):
        a = np.atleast_2d(x)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i < len(self.W) - 1:
                a = self._relu(a)
        return a

    def update(self, x, target):
        x = np.atleast_2d(x)
        acts, pre = [x], []
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b; pre.append(z)
            a = self._relu(z) if i < len(self.W)-1 else z
            acts.append(a)
        loss = float(np.mean((a - target)**2))
        delta = 2*(a - target)/len(x)
        for i in reversed(range(len(self.W))):
            dW = np.clip(acts[i].T @ delta, -1, 1)
            db = np.clip(delta.sum(axis=0, keepdims=True), -1, 1)
            self.W[i] -= self.lr * dW; self.b[i] -= self.lr * db
            if i > 0:
                delta = (delta @ self.W[i].T) * self._drel(pre[i-1])
        return loss

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


# ══════════════════════════════════════════════════════════════
# 8. DQN
# ══════════════════════════════════════════════════════════════
from collections import deque
import random as _random

class DQNPolicy:
    def __init__(self, state_dim, cfg):
        dc = cfg.get("DQN", {})
        self.n_act     = dc.get("n_actions", 20)
        self.max_ord   = dc.get("max_order_qty", 200)
        self.gamma     = dc.get("gamma", 0.95)
        self.eps       = dc.get("epsilon_start", 1.0)
        self.eps_end   = dc.get("epsilon_end", 0.01)
        self.eps_dec   = dc.get("epsilon_decay", 0.995)
        self.bs        = dc.get("batch_size", 64)
        self.tgt_freq  = dc.get("target_update_freq", 10)
        lr             = dc.get("learning_rate", 0.001)
        hidden         = dc.get("hidden_layers", [128, 64])
        sizes = [state_dim] + hidden + [self.n_act]
        self.actions   = np.linspace(0, self.max_ord, self.n_act)
        self.q_net     = _NN(sizes, lr)
        self.t_net     = _NN(sizes, lr); self.t_net.copy_from(self.q_net)
        self.mem       = deque(maxlen=dc.get("memory_size", 10000))
        self._step     = 0
        self.reward_hist = []

    def act(self, s, explore=True):
        if explore and _random.random() < self.eps:
            idx = _random.randint(0, self.n_act-1)
        else:
            idx = int(np.argmax(self.q_net.predict(s.reshape(1,-1))[0]))
        return idx, self.actions[idx]

    def remember(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def replay(self):
        if len(self.mem) < self.bs: return
        batch = _random.sample(self.mem, self.bs)
        S  = np.array([b[0] for b in batch])
        A  = np.array([b[1] for b in batch])
        R  = np.array([b[2] for b in batch])
        NS = np.array([b[3] for b in batch])
        D  = np.array([b[4] for b in batch])
        qc = self.q_net.predict(S)
        qn = self.t_net.predict(NS)
        tgt = qc.copy()
        for i in range(self.bs):
            tgt[i, A[i]] = R[i] if D[i] else R[i] + self.gamma * np.max(qn[i])
        self.q_net.update(S, tgt)
        self.eps = max(self.eps_end, self.eps * self.eps_dec)
        self._step += 1
        if self._step % self.tgt_freq == 0:
            self.t_net.copy_from(self.q_net)

    def train(self, demand, cfg, verbose=True):
        from simulation.core.inventory_env import InventoryEnv
        n_ep = cfg["DQN"].get("episodes", 500)
        print(f"  [DQN] Treinando {n_ep} episódios...")
        for ep in range(n_ep):
            env   = InventoryEnv(demand, cfg, seed=ep)
            s     = env.reset(); done = False; tot = 0.0
            while not done:
                idx, qty = self.act(s, explore=True)
                ns, r, done, _ = env.step(qty)
                self.remember(s, idx, r, ns, done)
                self.replay(); s = ns; tot += r
            self.reward_hist.append(tot)
            if verbose and (ep+1) % 100 == 0:
                print(f"    Ep {ep+1}/{n_ep} eps={self.eps:.3f} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        def p(s, env): return self.act(s, False)[1]
        return p


# ══════════════════════════════════════════════════════════════
# 9. PPO
# ══════════════════════════════════════════════════════════════
class PPOPolicy:
    def __init__(self, state_dim, cfg):
        pc = cfg.get("PPO", {})
        self.n_act   = pc.get("n_actions", 20)
        self.max_ord = pc.get("max_order_qty", 200)
        self.gamma   = pc.get("gamma", 0.99)
        self.clip    = pc.get("clip_epsilon", 0.2)
        self.k_ep    = pc.get("update_epochs", 4)
        lr           = pc.get("learning_rate", 0.0003)
        self.actions = np.linspace(0, self.max_ord, self.n_act)
        self.actor   = _NN([state_dim, 64, 32, self.n_act], lr)
        self.critic  = _NN([state_dim, 64, 32, 1], lr)
        self.reward_hist = []

    def _softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -20, 20))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def act(self, s, explore=True):
        lg = self.actor.predict(s.reshape(1,-1))[0]
        pr = self._softmax(lg)
        idx = np.random.choice(self.n_act, p=pr) if explore else np.argmax(pr)
        return idx, self.actions[idx], np.log(pr[idx]+1e-8)

    def train(self, demand, cfg, verbose=True):
        from simulation.core.inventory_env import InventoryEnv
        n_ep = cfg["PPO"].get("episodes", 300)
        print(f"  [PPO] Treinando {n_ep} episódios...")
        for ep in range(n_ep):
            env = InventoryEnv(demand, cfg, seed=ep)
            s = env.reset(); done=False
            traj = {"s":[],"a":[],"r":[],"lp":[],"v":[]}
            while not done:
                idx, qty, lp = self.act(s)
                v = float(self.critic.predict(s.reshape(1,-1))[0,0])
                ns, r, done, _ = env.step(qty)
                traj["s"].append(s); traj["a"].append(idx)
                traj["r"].append(r); traj["lp"].append(lp); traj["v"].append(v)
                s = ns
            self.reward_hist.append(sum(traj["r"]))
            G=0; rets=[]
            for r in reversed(traj["r"]): G=r+self.gamma*G; rets.insert(0,G)
            rets=np.array(rets); rets=(rets-rets.mean())/(rets.std()+1e-8)
            S=np.array(traj["s"]); A=np.array(traj["a"])
            old_lp=np.array(traj["lp"]); V=np.array(traj["v"])
            adv = rets - V
            for _ in range(self.k_ep):
                lg=self.actor.predict(S); pr=self._softmax(lg)
                new_lp=np.log(pr[np.arange(len(A)),A]+1e-8)
                rat=np.exp(new_lp-old_lp)
                cl=np.clip(rat,1-self.clip,1+self.clip)
                tgt_a=lg.copy()
                for i in range(len(A)):
                    tgt_a[i,A[i]]+=adv[i]*min(rat[i],cl[i])
                self.actor.update(S, tgt_a)
                self.critic.update(S, rets.reshape(-1,1))
            if verbose and (ep+1) % 100 == 0:
                print(f"    Ep {ep+1}/{n_ep} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        def p(s, env): return self.act(s, False)[1]
        return p


# ══════════════════════════════════════════════════════════════
# 10. SARSA (on-policy TD)
# ══════════════════════════════════════════════════════════════
class SARSAPolicy:
    def __init__(self, state_dim, cfg):
        sc = cfg.get("SARSA", {})
        self.n_act   = sc.get("n_actions", 20)
        self.max_ord = sc.get("max_order_qty", 200)
        self.gamma   = sc.get("gamma", 0.95)
        self.alpha   = sc.get("learning_rate", 0.001)
        self.eps     = sc.get("epsilon_start", 1.0)
        self.eps_end = sc.get("epsilon_end", 0.01)
        self.eps_dec = sc.get("epsilon_decay", 0.995)
        self.actions = np.linspace(0, self.max_ord, self.n_act)
        self.q_net   = _NN([state_dim, 64, 32, self.n_act], self.alpha)
        self.reward_hist = []

    def act(self, s, explore=True):
        if explore and _random.random() < self.eps:
            idx = _random.randint(0, self.n_act-1)
        else:
            idx = int(np.argmax(self.q_net.predict(s.reshape(1,-1))[0]))
        return idx, self.actions[idx]

    def train(self, demand, cfg, verbose=True):
        from simulation.core.inventory_env import InventoryEnv
        n_ep = cfg.get("SARSA", {}).get("episodes", 500)
        print(f"  [SARSA] Treinando {n_ep} episódios...")
        for ep in range(n_ep):
            env = InventoryEnv(demand, cfg, seed=ep)
            s = env.reset(); done=False; tot=0.0
            idx, qty = self.act(s)
            while not done:
                ns, r, done, _ = env.step(qty)
                nidx, nqty = self.act(ns) if not done else (0, 0.0)
                q_s  = self.q_net.predict(s.reshape(1,-1))
                q_ns = self.q_net.predict(ns.reshape(1,-1))
                tgt  = q_s.copy()
                tgt[0, idx] = r + (self.gamma * q_ns[0, nidx] if not done else r)
                self.q_net.update(s.reshape(1,-1), tgt)
                s=ns; idx=nidx; qty=nqty; tot+=r
            self.reward_hist.append(tot)
            self.eps = max(self.eps_end, self.eps * self.eps_dec)
            if verbose and (ep+1) % 100 == 0:
                print(f"    Ep {ep+1}/{n_ep} eps={self.eps:.3f} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        def p(s, env): return self.act(s, False)[1]
        return p


# ══════════════════════════════════════════════════════════════
# 11-12. Hybrid GA-RL (base compartilhada por DQN e PPO)
# ══════════════════════════════════════════════════════════════
class _HybridGARL:
    """GA decide QUANDO pedir (ROP gate); RL decide QUANTO pedir."""
    def __init__(self, ga_params: dict, rl_agent):
        self.ROP = ga_params["ROP"]; self.SS = ga_params["SS"]
        self.Q   = ga_params["Q"];   self._rl = rl_agent

    def make_policy(self):
        def p(s, env):
            if env.inventory + sum(env.pipeline) <= self.ROP + self.SS:
                return max(self._rl.act(s, False)[1], self.Q * 0.5)
            return 0.0
        return p


class HybridGADQN(_HybridGARL):
    def __init__(self, ga_params, dqn: DQNPolicy):
        super().__init__(ga_params, dqn)


class HybridGAPPO(_HybridGARL):
    def __init__(self, ga_params, ppo: PPOPolicy):
        super().__init__(ga_params, ppo)
