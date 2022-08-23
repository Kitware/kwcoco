import kwarray
import kwimage
import numpy as np

try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile


@profile
def test_shuffle_delayed_operations():
    """
    CommandLine:
        XDEV_PROFILE=1 xdoctest -m tests/test_delayed_ops.py test_shuffle_delayed_operations
    """
    # Try putting operations in differnet orders and ensure optimize always
    # fixes it.

    from kwcoco.util.delayed_ops.delayed_leafs import DelayedLoad

    fpath = kwimage.grab_test_image_fpath()
    # overviews=3)
    base = DelayedLoad(fpath, channels='r|g|b')._load_metadata()
    quantization = {'quant_max': 255, 'nodata': 0}
    base.get_overview(1).dequantize(quantization).optimize()

    operations = [
        ('warp', {'scale': 1}),
        ('crop', (slice(None), slice(None))),
        ('get_overview', 1),
        ('dequantize', quantization),
    ]

    dequant_idx = [t[0] for t in operations].index('dequantize')

    # rng = kwarray.ensure_rng(None)
    rng = kwarray.ensure_rng(86159412070383637)

    # Repeat the test multiple times.
    num_times = 10
    for _ in range(num_times):
        num_ops = rng.randint(1, 30)
        op_idxs = rng.randint(0, len(operations), size=num_ops)

        # Don't allow dequantize more than once
        keep_flags = op_idxs != dequant_idx
        if not np.all(keep_flags):
            keeper = rng.choice(np.where(~keep_flags)[0])
            keep_flags[keeper] = True
        op_idxs = op_idxs[keep_flags]

        delayed = base
        for idx in op_idxs:
            name, args = operations[idx]
            func = getattr(delayed, name)
            delayed = func(args)

        # delayed.write_network_text(with_labels="name")
        opt = delayed.optimize()
        # opt.write_network_text(with_labels="name")

        # We always expect that we will get a sequence in the form
        expected_sequence = [
            'DelayedWarp', 'DelayedDequantize', 'DelayedCrop',
            'DelayedOverview', 'DelayedLoad'
        ]
        # But we are allowed to skip steps
        import networkx as nx
        graph = opt.as_graph()
        node_order = list(nx.topological_sort(graph))
        opname_order = [graph.nodes[n]['type'] for n in node_order]
        if opname_order[-1] != expected_sequence[-1]:
            raise AssertionError('Unexpected sequence')
        prev_idx = -1
        for opname in opname_order:
            this_idx = expected_sequence.index(opname)
            if this_idx <= prev_idx:
                raise AssertionError('Unexpected sequence')
            prev_idx = this_idx
