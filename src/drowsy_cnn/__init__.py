"""drowsy_cnn — Taller CNN detección de somnolencia en conductores.

Paquete Python para el proyecto AAIV 2026-II · MIAA-MCD (Icesi).
Diseño modular que separa: config, datos, augmentation, modelos, losses, training,
inferencia e interpretabilidad. Cada módulo es independiente e importable.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("drowsy-cnn")
except PackageNotFoundError:  # editable install antes del build
    __version__ = "0.1.0"

__all__ = ["__version__"]
