"""
mkinit kwcoco.metrics -w --relative
"""
# flake8: noqa


__submodules__ = [
    'detect_metrics',
    'confusion_vectors',
]

# <AUTOGEN_INIT>
from . import detect_metrics
from . import confusion_vectors

from .detect_metrics import (DetectionMetrics, eval_detections_cli,)
from .confusion_vectors import (BinaryConfusionVectors, ConfusionVectors,
                                Measures, OneVsRestConfusionVectors,
                                PerClass_Measures,)

__all__ = ['BinaryConfusionVectors', 'ConfusionVectors', 'DetectionMetrics',
           'Measures', 'OneVsRestConfusionVectors', 'PerClass_Measures',
           'confusion_vectors', 'detect_metrics', 'eval_detections_cli']
