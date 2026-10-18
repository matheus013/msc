"""Optional AIPE context extension, deliberately absent from the running pipeline."""

from .contracts import assemble_context, validate_config

__all__ = ["assemble_context", "validate_config", "fit_context_pse", "evaluate_context_pse"]


def __getattr__(name):
    if name in ("fit_context_pse", "evaluate_context_pse"):
        from . import selection
        return getattr(selection, name)
    raise AttributeError(name)
