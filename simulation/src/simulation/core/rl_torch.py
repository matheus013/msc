"""
rl_torch.py — Agentes de aprendizado por reforço em PyTorch (versões estado da arte).

Substitui a rede neural escrita à mão em NumPy do módulo `policies.py`, cujo
gradiente do PPO estava incorreto: a atualização antiga somava a vantagem
diretamente aos logits alvo e treinava o ator por erro quadrático, o que não é
o gradiente da surrogate function recortada.

Agentes:
  DoubleDQNAgent   — Double DQN (van Hasselt, Guez & Silver, AAAI 2016) com
                     arquitetura dueling (Wang et al., ICML 2016), perda de
                     Huber e rede alvo. Escolhido como versão estado da arte
                     do slot DQN.
  PPOAgent         — PPO com clipped surrogate (Schulman et al., 2017) e
                     Generalized Advantage Estimation (Schulman et al., ICLR
                     2016), bônus de entropia, normalização de vantagem e
                     recorte da função valor.
  ExpectedSARSAAgent — Expected SARSA (van Seijen et al., ADPRL 2009) tabular.
                     O slot SARSA não tem estado da arte publicado em veículo
                     relevante para inventário (ver docs/references/
                     estado_da_arte_politicas.md, seção 3.3); mantém-se como
                     baseline tabular deliberadamente simples, com a variante
                     de menor variância e discretização bidimensional.

Todos operam sobre o mesmo `InventoryEnv`, com a mesma interface
`make_policy() -> fn(state, env) -> float` usada pelas demais políticas, de
modo que a comparação no benchmark permanece controlada.

Reprodutibilidade: `seed` fixa torch, numpy e random. Execução em CPU por
padrão — MPS/CUDA são opcionais e desativados por default porque, em redes
desta escala, o overhead de transferência domina o ganho.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Thread pool do torch
#
# `simulation.core.__init__` importa forecasting, que carrega xgboost e
# scikit-learn; ambos trazem sua própria libomp. Quando o backward do torch
# roda em múltiplas threads sobre uma libomp já inicializada por outra
# biblioteca, o processo termina em segmentation fault no macOS (a falha
# ocorre na thread do motor de autograd, sem frame Python, o que a torna
# difícil de diagnosticar pelo traceback).
#
# Fixar em 1 thread elimina o conflito. Não há custo prático de desempenho:
# as redes aqui têm no máximo ~10^4 parâmetros e lotes de 64 amostras, faixa
# em que o overhead de sincronização domina qualquer ganho de paralelismo.
# Configurável por AIPE_TORCH_THREADS para quem quiser medir.
# ─────────────────────────────────────────────────────────────────────────────
try:
    torch.set_num_threads(int(__import__("os").environ.get("AIPE_TORCH_THREADS", "1")))
except Exception:  # pragma: no cover - ambiente sem suporte a set_num_threads
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_device(cfg_device: Optional[str]) -> torch.device:
    """CPU por padrão; 'mps'/'cuda'/'auto' apenas se pedido explicitamente."""
    if cfg_device in (None, "", "cpu"):
        return torch.device("cpu")
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _action_grid(max_order_qty: float, n_actions: int) -> np.ndarray:
    """Grade de quantidades de pedido. A ação 0 é sempre 'não pedir'."""
    return np.linspace(0.0, float(max_order_qty), int(n_actions))


# ═══════════════════════════════════════════════════════════════════════════
# Redes
# ═══════════════════════════════════════════════════════════════════════════

class _DuelingQNet(nn.Module):
    """
    Dueling DQN (Wang et al., ICML 2016): separa V(s) e A(s,a).

    Q(s,a) = V(s) + [A(s,a) - mean_a' A(s,a')]

    A subtração da média identifica o modelo (sem ela, V e A são
    indistinguíveis a menos de constante) e estabiliza o treino em estados
    onde a escolha da ação pouco altera o retorno — situação frequente em
    inventário quando o estoque está muito acima do ponto de pedido.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + (a - a.mean(dim=-1, keepdim=True))


class _ActorCritic(nn.Module):
    """Ator categórico + crítico, com troncos separados."""

    def __init__(self, state_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()

        def _mlp(out_dim: int, out_gain: float) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = state_dim
            for h in hidden:
                lin = nn.Linear(prev, h)
                nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
                nn.init.zeros_(lin.bias)
                layers += [lin, nn.Tanh()]
                prev = h
            head = nn.Linear(prev, out_dim)
            nn.init.orthogonal_(head.weight, gain=out_gain)
            nn.init.zeros_(head.bias)
            layers.append(head)
            return nn.Sequential(*layers)

        # gain 0.01 na cabeça do ator deixa a política inicial quase uniforme
        self.actor = _mlp(n_actions, out_gain=0.01)
        self.critic = _mlp(1, out_gain=1.0)

    def forward(self, x: torch.Tensor):
        return self.actor(x), self.critic(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Double DQN (dueling)
# ═══════════════════════════════════════════════════════════════════════════

class DoubleDQNAgent:
    """
    Double DQN com arquitetura dueling.

    O DQN clássico superestima valores porque usa a mesma rede para escolher e
    avaliar a ação do próximo estado. O Double DQN desacopla:

        y = r + gamma * Q_target(s', argmax_a Q_online(s', a))

    Em horizontes curtos (T=38 ciclos) a superestimação é especialmente
    danosa: o agente aprende a pedir demais porque valoriza excessivamente
    estados de estoque alto.
    """

    def __init__(self, state_dim: int, cfg: dict):
        dc = cfg.get("DQN", {})
        self.state_dim = state_dim
        self.n_act = int(dc.get("n_actions", 20))
        self.max_ord = float(dc.get("max_order_qty", 200))
        self.gamma = float(dc.get("gamma", 0.95))
        self.eps = float(dc.get("epsilon_start", 1.0))
        self.eps_end = float(dc.get("epsilon_end", 0.01))
        self.eps_decay = float(dc.get("epsilon_decay", 0.995))
        self.bs = int(dc.get("batch_size", 64))
        self.tgt_freq = int(dc.get("target_update_freq", 10))
        self.tau = float(dc.get("soft_update_tau", 0.0))  # >0 usa soft update
        lr = float(dc.get("learning_rate", 1e-3))
        hidden = list(dc.get("hidden_layers", [128, 64]))
        self.grad_clip = float(dc.get("grad_clip", 10.0))
        self.device = _resolve_device(dc.get("device"))
        self.seed = int(dc.get("seed", 42))
        seed_everything(self.seed)

        self.actions = _action_grid(self.max_ord, self.n_act)
        self.q_net = _DuelingQNet(state_dim, self.n_act, hidden).to(self.device)
        self.t_net = _DuelingQNet(state_dim, self.n_act, hidden).to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.t_net.eval()
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.mem: deque = deque(maxlen=int(dc.get("memory_size", 10000)))
        self._step = 0
        self.reward_hist: list[float] = []
        self.loss_hist: list[float] = []

    # ── interação ────────────────────────────────────────────────────────
    def act(self, s: np.ndarray, explore: bool = True) -> tuple[int, float]:
        if explore and random.random() < self.eps:
            idx = random.randrange(self.n_act)
        else:
            with torch.no_grad():
                t = torch.as_tensor(s, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                idx = int(self.q_net(t).argmax(dim=-1).item())
        return idx, float(self.actions[idx])

    def remember(self, s, a, r, ns, done) -> None:
        self.mem.append((np.asarray(s, dtype=np.float32), int(a), float(r),
                         np.asarray(ns, dtype=np.float32), bool(done)))

    # ── aprendizado ──────────────────────────────────────────────────────
    def replay(self) -> Optional[float]:
        if len(self.mem) < self.bs:
            return None
        batch = random.sample(self.mem, self.bs)
        S = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        A = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.int64,
                            device=self.device).unsqueeze(-1)
        R = torch.as_tensor(np.array([b[2] for b in batch]), dtype=torch.float32,
                            device=self.device)
        NS = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        D = torch.as_tensor(np.array([b[4] for b in batch]), dtype=torch.float32,
                            device=self.device)

        q_sa = self.q_net(S).gather(1, A).squeeze(-1)

        with torch.no_grad():
            # Double DQN: online escolhe a ação, target avalia
            next_a = self.q_net(NS).argmax(dim=-1, keepdim=True)
            q_next = self.t_net(NS).gather(1, next_a).squeeze(-1)
            y = R + (1.0 - D) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, y)          # Huber
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.opt.step()

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self._step += 1
        if self.tau > 0.0:
            with torch.no_grad():
                for tp, qp in zip(self.t_net.parameters(), self.q_net.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * qp)
        elif self._step % self.tgt_freq == 0:
            self.t_net.load_state_dict(self.q_net.state_dict())

        self.loss_hist.append(float(loss.item()))
        return float(loss.item())

    def prepopulate_from_ga(self, demand, cfg, ga_params: dict,
                            n_transitions: int = 1000) -> int:
        """
        Pré-popula o replay buffer com trajetórias da política parametrizada
        pelo GA (etapa (ii) da arquitetura híbrida GA-RL da proposta).
        """
        from simulation.core.inventory_env import InventoryEnv
        ROP, Q, SS = ga_params["ROP"], ga_params["Q"], ga_params["SS"]
        q_idx = int(np.argmin(np.abs(self.actions - Q)))

        added, ep = 0, 0
        while added < n_transitions:
            env = InventoryEnv(demand, cfg, seed=2000 + ep)
            s = env.reset()
            done = False
            while not done and added < n_transitions:
                pos = env.inventory + sum(env.pipeline)
                a_idx, qty = (q_idx, Q) if pos <= ROP + SS else (0, 0.0)
                ns, r, done, _ = env.step(qty)
                self.remember(s, a_idx, r, ns, done)
                s = ns
                added += 1
            ep += 1
        return added

    def train(self, demand, cfg, verbose: bool = True) -> None:
        from simulation.core.inventory_env import InventoryEnv
        n_ep = int(cfg.get("DQN", {}).get("episodes", 500))
        for ep in range(n_ep):
            env = InventoryEnv(demand, cfg, seed=ep)
            s = env.reset()
            done, tot = False, 0.0
            while not done:
                idx, qty = self.act(s, explore=True)
                ns, r, done, _ = env.step(qty)
                self.remember(s, idx, r, ns, done)
                self.replay()
                s = ns
                tot += r
            self.reward_hist.append(tot)
            if verbose and (ep + 1) % 100 == 0:
                print(f"    [DoubleDQN] ep {ep+1}/{n_ep} eps={self.eps:.3f} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        self.q_net.eval()

        def p(s, env):
            return self.act(s, explore=False)[1]
        return p


# ═══════════════════════════════════════════════════════════════════════════
# 2. PPO com GAE
# ═══════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO com clipped surrogate objective e GAE(lambda).

    Objetivo do ator (Schulman et al., 2017):
        L = -E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]
    com r_t = pi_novo(a|s) / pi_antigo(a|s).

    A implementação anterior somava a vantagem aos logits e treinava por MSE,
    o que não é o gradiente deste objetivo. Aqui o gradiente é obtido por
    autograd sobre a razão de verossimilhanças, como na formulação original.

    GAE (Schulman et al., ICLR 2016) reduz a variância do estimador de
    vantagem, o que importa neste problema porque o horizonte é curto
    (T=38) e a recompensa por ciclo tem escala muito heterogênea entre séries.
    """

    def __init__(self, state_dim: int, cfg: dict):
        pc = cfg.get("PPO", {})
        self.state_dim = state_dim
        self.n_act = int(pc.get("n_actions", 20))
        self.max_ord = float(pc.get("max_order_qty", 200))
        self.gamma = float(pc.get("gamma", 0.99))
        self.lam = float(pc.get("gae_lambda", 0.95))
        self.clip = float(pc.get("clip_epsilon", 0.2))
        self.k_ep = int(pc.get("update_epochs", 4))
        self.ent_coef = float(pc.get("entropy_coef", 0.01))
        self.vf_coef = float(pc.get("value_coef", 0.5))
        self.minibatch = int(pc.get("minibatch_size", 64))
        self.grad_clip = float(pc.get("grad_clip", 0.5))
        self.clip_vloss = bool(pc.get("clip_value_loss", True))
        lr = float(pc.get("learning_rate", 3e-4))
        hidden = list(pc.get("hidden_layers", [64, 64]))
        self.device = _resolve_device(pc.get("device"))
        self.seed = int(pc.get("seed", 42))
        seed_everything(self.seed)

        self.actions = _action_grid(self.max_ord, self.n_act)
        self.net = _ActorCritic(state_dim, self.n_act, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.reward_hist: list[float] = []

    # ── interação ────────────────────────────────────────────────────────
    def act(self, s: np.ndarray, explore: bool = True):
        with torch.no_grad():
            t = torch.as_tensor(s, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            logits, value = self.net(t)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample() if explore else logits.argmax(dim=-1)
            logp = dist.log_prob(idx)
        i = int(idx.item())
        return i, float(self.actions[i]), float(logp.item()), float(value.item())

    # ── GAE ──────────────────────────────────────────────────────────────
    def _gae(self, rewards, values, dones, last_value: float):
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = last_value if t == n - 1 else values[t + 1]
            next_nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_v * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        ret = adv + np.asarray(values, dtype=np.float32)
        return adv, ret

    # ── atualização ──────────────────────────────────────────────────────
    def _update(self, S, A, old_logp, adv, ret, old_val) -> None:
        S = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(A, dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        old_val = torch.as_tensor(old_val, dtype=torch.float32, device=self.device)

        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = S.shape[0]
        mb = min(self.minibatch, n)
        idx = np.arange(n)
        for _ in range(self.k_ep):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = torch.as_tensor(idx[start:start + mb], dtype=torch.int64,
                                    device=self.device)
                logits, value = self.net(S[b])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(A[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[b]
                policy_loss = -torch.min(s1, s2).mean()

                if self.clip_vloss:
                    v_clipped = old_val[b] + torch.clamp(
                        value - old_val[b], -self.clip, self.clip)
                    v_loss = 0.5 * torch.max((value - ret[b]) ** 2,
                                             (v_clipped - ret[b]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((value - ret[b]) ** 2).mean()

                loss = policy_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

    # ── rollouts ─────────────────────────────────────────────────────────
    def _rollout(self, demand, cfg, seed: int, forced_policy=None):
        """
        Coleta um episódio. Se `forced_policy` for dado, as ações vêm dela
        (usado no warm-start pelo GA) mas as log-probs são as da política
        atual — o que mantém a razão de verossimilhanças bem definida.
        """
        from simulation.core.inventory_env import InventoryEnv
        env = InventoryEnv(demand, cfg, seed=seed)
        s = env.reset()
        done = False
        S, A, R, LP, V, D = [], [], [], [], [], []
        while not done:
            if forced_policy is None:
                a_idx, qty, logp, val = self.act(s, explore=True)
            else:
                qty_ga = forced_policy(env)
                a_idx = int(np.argmin(np.abs(self.actions - qty_ga)))
                qty = float(self.actions[a_idx])
                with torch.no_grad():
                    t = torch.as_tensor(s, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                    logits, value = self.net(t)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = float(dist.log_prob(
                        torch.tensor([a_idx], device=self.device)).item())
                    val = float(value.item())
            ns, r, done, _ = env.step(qty)
            S.append(s); A.append(a_idx); R.append(r)
            LP.append(logp); V.append(val); D.append(done)
            s = ns
        return (np.asarray(S, dtype=np.float32), np.asarray(A),
                np.asarray(R, dtype=np.float32), np.asarray(LP, dtype=np.float32),
                np.asarray(V, dtype=np.float32), np.asarray(D))

    def warmstart_from_ga(self, demand, cfg, ga_params: dict,
                          n_episodes: int = 20) -> None:
        """Aquece a política com trajetórias da parametrização do GA."""
        ROP, Q, SS = ga_params["ROP"], ga_params["Q"], ga_params["SS"]

        def ga_policy(env):
            pos = env.inventory + sum(env.pipeline)
            return Q if pos <= ROP + SS else 0.0

        for ep in range(n_episodes):
            S, A, R, LP, V, D = self._rollout(demand, cfg, seed=3000 + ep,
                                              forced_policy=ga_policy)
            adv, ret = self._gae(R, V, D, last_value=0.0)
            self._update(S, A, LP, adv, ret, V)
            self.reward_hist.append(float(R.sum()))

    def train(self, demand, cfg, verbose: bool = True) -> None:
        n_ep = int(cfg.get("PPO", {}).get("episodes", 500))
        for ep in range(n_ep):
            S, A, R, LP, V, D = self._rollout(demand, cfg, seed=ep)
            adv, ret = self._gae(R, V, D, last_value=0.0)
            self._update(S, A, LP, adv, ret, V)
            self.reward_hist.append(float(R.sum()))
            if verbose and (ep + 1) % 100 == 0:
                print(f"    [PPO-GAE] ep {ep+1}/{n_ep} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        self.net.eval()

        def p(s, env):
            return self.act(s, explore=False)[1]
        return p


# ═══════════════════════════════════════════════════════════════════════════
# 3. Expected SARSA tabular
# ═══════════════════════════════════════════════════════════════════════════

class ExpectedSARSAAgent:
    """
    Expected SARSA (van Seijen et al., ADPRL 2009).

    Substitui a amostra Q(s',a') do SARSA pela esperança sob a política
    epsilon-greedy:

        y = r + gamma * sum_a pi(a|s') Q(s',a)

    Isso elimina a variância induzida pela amostragem da próxima ação,
    mantendo o caráter on-policy. Não há estado da arte publicado de SARSA
    para inventário; este agente permanece no portfólio como baseline
    tabular, agora com discretização bidimensional (estoque x pipeline) em
    vez de usar apenas o nível de estoque.
    """

    def __init__(self, state_dim: int, cfg: dict):
        sc = cfg.get("SARSA", {})
        self.n_inv = int(sc.get("n_states", 5))
        self.n_pipe = int(sc.get("n_pipeline_states", 3))
        self.n_act = int(sc.get("n_actions", 10))
        self.max_ord = float(sc.get("max_order_qty", 200))
        self.gamma = float(sc.get("gamma", 0.99))
        self.alpha = float(sc.get("learning_rate", 0.1))
        self.eps = float(sc.get("epsilon", 0.1))
        self.seed = int(sc.get("seed", 42))
        seed_everything(self.seed)

        self.actions = _action_grid(self.max_ord, self.n_act)
        self.Q = np.zeros((self.n_inv * self.n_pipe, self.n_act), dtype=np.float64)
        self.reward_hist: list[float] = []

    def _discretize(self, state: np.ndarray) -> int:
        inv = float(np.clip(state[0], 0.0, 1.0))
        pipe = float(np.clip(state[2], 0.0, 1.0))
        i = min(int(inv * self.n_inv), self.n_inv - 1)
        p = min(int(pipe * self.n_pipe), self.n_pipe - 1)
        return i * self.n_pipe + p

    def act(self, s, explore: bool = True):
        si = self._discretize(s)
        if explore and random.random() < self.eps:
            idx = random.randrange(self.n_act)
        else:
            idx = int(np.argmax(self.Q[si]))
        return idx, float(self.actions[idx])

    def _expected_q(self, si: int) -> float:
        """E_pi[Q(s,.)] sob epsilon-greedy."""
        q = self.Q[si]
        best = int(np.argmax(q))
        pi = np.full(self.n_act, self.eps / self.n_act)
        pi[best] += 1.0 - self.eps
        return float(np.dot(pi, q))

    def train(self, demand, cfg, verbose: bool = True) -> None:
        from simulation.core.inventory_env import InventoryEnv
        n_ep = int(cfg.get("SARSA", {}).get("episodes", 500))
        for ep in range(n_ep):
            env = InventoryEnv(demand, cfg, seed=ep)
            s = env.reset()
            done, tot = False, 0.0
            si = self._discretize(s)
            idx, qty = self.act(s)
            while not done:
                ns, r, done, _ = env.step(qty)
                nsi = self._discretize(ns)
                target = r if done else r + self.gamma * self._expected_q(nsi)
                self.Q[si, idx] += self.alpha * (target - self.Q[si, idx])
                if not done:
                    idx, qty = self.act(ns)
                s, si = ns, nsi
                tot += r
            self.reward_hist.append(tot)
            if verbose and (ep + 1) % 100 == 0:
                print(f"    [ExpSARSA] ep {ep+1}/{n_ep} "
                      f"avg_r={np.mean(self.reward_hist[-50:]):.1f}")

    def make_policy(self):
        def p(s, env):
            return self.act(s, explore=False)[1]
        return p


# ═══════════════════════════════════════════════════════════════════════════
# 4. Híbridas GA-RL
# ═══════════════════════════════════════════════════════════════════════════

class HybridGARL:
    """
    GA decide QUANDO pedir (limiar ROP+SS); o agente de RL decide QUANTO.

    O piso `Q * floor_ratio` evita que o agente aprenda a emitir pedidos
    simbólicos apenas para satisfazer o gatilho do GA — comportamento
    degenerado observado no Experimento 1 com o PPO sem inicialização.
    """

    def __init__(self, ga_params: dict, rl_agent, floor_ratio: float = 0.5):
        self.ROP = ga_params["ROP"]
        self.SS = ga_params["SS"]
        self.Q = ga_params["Q"]
        self._rl = rl_agent
        self.floor_ratio = float(floor_ratio)

    def make_policy(self):
        def p(s, env):
            if env.inventory + sum(env.pipeline) <= self.ROP + self.SS:
                qty = self._rl.act(s, explore=False)[1]
                return max(qty, self.Q * self.floor_ratio)
            return 0.0
        return p


class HybridGADQN(HybridGARL):
    pass


class HybridGAPPO(HybridGARL):
    pass
