# nexatron/__init__.py
from .hill_saturation import SurgicalHillSaturation

HillSaturation = SurgicalHillSaturation
SurgicalSaturation = SurgicalHillSaturation

__version__ = "1.1.0"
__all__ = ["SurgicalHillSaturation", "HillSaturation", "SurgicalSaturation"]