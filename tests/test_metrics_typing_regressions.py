import numpy as np


def test_ovr_measure_combiner_preserves_growth_option():
    from kwcoco.metrics.confusion_measures import OneVersusRestMeasureCombiner

    combiner = OneVersusRestMeasureCombiner(precision=3, growth='max')
    assert combiner.precision == 3
    assert combiner.growth == 'max'


def test_confusion_vectors_json_with_sequence_classes():
    from kwcoco.metrics.confusion_vectors import ConfusionVectors

    classes = ['cat', 'dog']
    cfsn_vecs = ConfusionVectors.from_arrays(
        true=np.array([0, 1]),
        pred=np.array([0, 1]),
        classes=classes,
    )
    state = cfsn_vecs.__json__()
    recon = ConfusionVectors.from_json(state)
    assert recon.classes is not None
    assert list(recon.classes) == classes


def test_confusion_vectors_json_without_classes():
    from kwcoco.metrics.confusion_vectors import ConfusionVectors

    cfsn_vecs = ConfusionVectors.from_arrays(
        true=np.array([0, 1]),
        pred=np.array([0, 1]),
        classes=None,
    )
    state = cfsn_vecs.__json__()
    recon = ConfusionVectors.from_json(state)
    assert recon.classes is None


def test_kw18_to_coco_preserves_video_name():
    from kwcoco.formats.kw18 import KW18

    dset = KW18.demo().to_coco(video_name='demo-video')
    assert dset.videos().lookup('name') == ['demo-video']


def test_suppress_print_restores_each_module_print():
    from types import ModuleType
    from kwcoco.util.util_monkey import SupressPrint

    mod1 = ModuleType('mod1')
    mod2 = ModuleType('mod2')
    calls1 = []
    calls2 = []
    setattr(mod1, 'print', calls1.append)
    setattr(mod2, 'print', calls2.append)

    with SupressPrint(mod1, mod2):
        getattr(mod1, 'print')('hidden-1')
        getattr(mod2, 'print')('hidden-2')

    getattr(mod1, 'print')('visible-1')
    getattr(mod2, 'print')('visible-2')
    assert calls1 == ['visible-1']
    assert calls2 == ['visible-2']


def test_confusion_matrix_axis_names_are_independent():
    from kwcoco.metrics.confusion_vectors import ConfusionVectors

    cfsn_vecs = ConfusionVectors.from_arrays(
        true=np.array([0, 1]),
        pred=np.array([1, 0]),
        classes=['cat', 'dog'],
    )
    cm = cfsn_vecs.confusion_matrix()
    assert cm.index.name == 'real'
    assert cm.columns.name == 'pred'
