"""
device.py — Seleção centralizada de dispositivo de cálculo (CPU / CUDA / MPS).

Toda a pilha de simulação e otimização usa `get_device()` para escolher onde
alocar tensores, de modo que a política de dispositivo fique em um único lugar.

Prioridade quando `preference="auto"`:
    1. CUDA  (NVIDIA)
    2. MPS   (Apple Silicon)
    3. CPU

Sobrescrita por variável de ambiente ou por configuração:
    AIPE_DEVICE=cpu|cuda|mps|auto        (tem precedência sobre a config)
    simulation.device: "auto"            (conf/base/parameters/simulation.yml)

Nota honesta sobre desempenho: GPU só compensa quando o lote é grande. Nas
redes dos agentes de RL deste projeto (~10^4 parâmetros, lotes de 64) a
transferência host-device domina e a CPU costuma ganhar. O ganho real de GPU
aqui está na SIMULAÇÃO VETORIZADA das meta-heurísticas, onde centenas de
parametrizações candidatas avançam em paralelo no mesmo passo de tempo. Por
isso o default de `simulation.device` é "auto" (usa GPU quando houver), mas o
default dos agentes de RL é "cpu"; ambos são configuráveis.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

log = logging.getLogger(__name__)

_VALID = {"cpu", "cuda", "mps", "auto"}


def available_devices() -> dict:
    """Inventário do que existe nesta máquina, para log e diagnóstico."""
    info = {
        "cpu": True,
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()),
        "torch_version": torch.__version__,
        "n_threads": torch.get_num_threads(),
    }
    if info["cuda"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_name"] = torch.cuda.get_device_name(0)
    return info


@lru_cache(maxsize=8)
def _resolve(preference: str) -> torch.device:
    pref = (os.environ.get("AIPE_DEVICE") or preference or "auto").lower()
    if pref not in _VALID:
        raise ValueError(f"device invalido: {pref!r}. Use um de {sorted(_VALID)}.")

    if pref == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(pref)
        # Falha suave: pedir cuda/mps sem ter o backend cai para CPU com aviso,
        # em vez de derrubar um benchmark de horas no meio.
        if dev.type == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA pedido mas indisponivel; usando CPU.")
            dev = torch.device("cpu")
        if dev.type == "mps" and not (getattr(torch.backends, "mps", None)
                                      and torch.backends.mps.is_available()):
            log.warning("MPS pedido mas indisponivel; usando CPU.")
            dev = torch.device("cpu")

    log.info("Dispositivo de calculo: %s", dev)
    return dev


def get_device(preference: str = "auto") -> torch.device:
    """Dispositivo a usar. Resultado memoizado por preferência."""
    return _resolve(preference)


# ─────────────────────────────────────────────────────────────────────────────
# Roteamento consciente do tamanho do lote
#
# Medição nesta máquina (Apple M-series, torch 2.8, simulação vetorizada de
# T=38 ciclos, média de 3 execuções):
#
#        lote      CPU (s)     MPS (s)    ganho
#         100       0.0028      0.0694     0.04x
#       1.000       0.0042      0.0685     0.06x
#      10.000       0.0127      0.0636     0.20x
#     100.000       0.0889      0.1009     0.88x
#     400.000       0.3567      0.2620     1.36x
#
# O ponto de virada está por volta de 150.000 trajetórias simultâneas. Abaixo
# disso a GPU PERDE, porque o custo de lançar kernels e sincronizar domina
# tensores pequenos (a população do GA tem shape 100x3).
#
# Por isso "usar a GPU porque ela existe" tornaria o benchmark mais lento, não
# mais rápido. O roteamento abaixo usa o acelerador apenas quando o lote
# justifica. CUDA tem overhead menor que MPS, daí o limiar mais baixo.
# ─────────────────────────────────────────────────────────────────────────────

GPU_MIN_BATCH = {"mps": 150_000, "cuda": 20_000}


def get_device_for_batch(batch_size: int, preference: str = "auto") -> torch.device:
    """
    Dispositivo apropriado para um lote deste tamanho.

    Com `preference="auto"`, encaminha para o acelerador só quando o lote
    supera o limiar medido. Uma preferência explícita ("mps", "cuda", "cpu") é
    sempre respeitada — útil para reproduzir medições.
    """
    pref = (os.environ.get("AIPE_DEVICE") or preference or "auto").lower()
    if pref != "auto":
        return _resolve(pref)

    dev = _resolve("auto")
    if dev.type in GPU_MIN_BATCH and batch_size < GPU_MIN_BATCH[dev.type]:
        return torch.device("cpu")
    return dev


def default_dtype(device: torch.device) -> torch.dtype:
    """
    float32 em todo lugar.

    MPS não suporta float64; usar float32 uniformemente mantém os resultados
    comparáveis entre dispositivos, o que importa mais aqui do que a precisão
    extra — os custos simulados têm ordem de 10^3 a 10^5 e float32 dá ~7
    dígitos significativos.
    """
    return torch.float32


def describe() -> str:
    """Linha única para log de início de execução."""
    info = available_devices()
    parts = [f"torch={info['torch_version']}", f"threads={info['n_threads']}"]
    if info["cuda"]:
        parts.append(f"CUDA={info.get('cuda_name')} x{info.get('cuda_device_count')}")
    elif info["mps"]:
        parts.append("MPS=disponivel (Apple Silicon)")
    else:
        parts.append("GPU=nenhuma")
    return " | ".join(parts)
