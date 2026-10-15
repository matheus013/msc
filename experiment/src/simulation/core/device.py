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

Auto prioriza a GPU quando disponivel, inclusive em lotes pequenos.
Use AIPE_DEVICE=cpu para comparacoes de desempenho ou reproducao em CPU.
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
    pref = (preference or "auto").lower()
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
    return _resolve((os.environ.get("AIPE_DEVICE") or preference or "auto").lower())


def get_device_for_batch(batch_size: int, preference: str = "auto") -> torch.device:
    """Prioriza GPU em qualquer lote; mantem a interface dos simuladores."""
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo")
    return get_device(preference)


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


def xgboost_device(preference: str = "auto") -> str:
    """XGBoost suporta CUDA; MPS usa o backend CPU."""
    return "cuda" if get_device(preference).type == "cuda" else "cpu"
