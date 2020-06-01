"""
mkinit kwcoco.metrics -w --relative
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals


__submodules__ = [
    'detect_metrics',
    'confusion_vectors',
]

# <AUTOGEN_INIT>
from . import detect_metrics
from . import confusion_vectors

from .detect_metrics import (DetectionMetrics, eval_detections_cli,)
from .confusion_vectors import (BinaryConfusionVectors, ConfusionVectors,
                                DictProxy, OneVsRestConfusionVectors,
                                PR_Result, PerClass_PR_Result,
                                PerClass_ROC_Result, PerClass_Threshold_Result,
                                ROC_Result, Threshold_Result,)

__all__ = ['BinaryConfusionVectors', 'ConfusionVectors', 'DetectionMetrics',
           'DictProxy', 'OneVsRestConfusionVectors', 'PR_Result',
           'PerClass_PR_Result', 'PerClass_ROC_Result',
           'PerClass_Threshold_Result', 'ROC_Result', 'Threshold_Result',
           'confusion_vectors', 'detect_metrics', 'eval_detections_cli']
