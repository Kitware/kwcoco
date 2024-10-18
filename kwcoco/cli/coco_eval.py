#!/usr/bin/env python
"""
Wraps the logic in kwcoco/coco_evaluator.py with a command line script
"""
from kwcoco import coco_evaluator
import scriptconfig as scfg
import ubelt as ub


class CocoEvalCLI(coco_evaluator.CocoEvalConfig):
    """
    Evaluate and score predicted versus truth detections / classifications in a
    COCO dataset.

    SeeAlso:
        python -m kwcoco.metrics.segmentation_metrics --help
    """
    __command__ = 'eval'
    __alias__ = ['eval_detections']

    # These should go into the CLI args, not the class config args
    expt_title = scfg.Value('', type=str, help='title for plots', tags=['perf_param'])
    draw = scfg.Value(True, isflag=1, help='draw metric plots', tags=['perf_param'])
    out_dpath = scfg.Value('./coco_metrics', type=str, help='where to dump results', tags=['out_path'])
    out_fpath = scfg.Value('auto', type=str, help=ub.paragraph(
        '''
        Where to dump the json file containing result. If "auto",
        defaults to out_dpath / "metrics.json"
        '''), tags=['out_path', 'primary'])

    @classmethod
    def main(cls, cmdline=True, **kw):
        """

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> from kwcoco.cli.coco_eval import *  # NOQA
            >>> import ubelt as ub
            >>> from kwcoco.cli.coco_eval import *  # NOQA
            >>> from os.path import join
            >>> import kwcoco
            >>> dpath = ub.Path.appdir('kwcoco/tests/eval').ensuredir()
            >>> true_dset = kwcoco.CocoDataset.demo('shapes8')
            >>> from kwcoco.demo.perterb import perterb_coco
            >>> kwargs = {
            >>>     'box_noise': 0.5,
            >>>     'n_fp': (0, 10),
            >>>     'n_fn': (0, 10),
            >>> }
            >>> pred_dset = perterb_coco(true_dset, **kwargs)
            >>> true_dset.fpath = join(dpath, 'true.mscoco.json')
            >>> pred_dset.fpath = join(dpath, 'pred.mscoco.json')
            >>> true_dset.dump(true_dset.fpath)
            >>> pred_dset.dump(pred_dset.fpath)
            >>> draw = False  # set to false for faster tests
            >>> CocoEvalCLI.main(cmdline=False,
            >>>     true_dataset=true_dset.fpath,
            >>>     pred_dataset=pred_dset.fpath,
            >>>     draw=draw, out_dpath=dpath)
        """
        main(cmdline=cmdline, **kw)


__cli__ = CocoEvalCLI


def main(cmdline=True, **kw):
    r"""
    TODO:
        - [X] should live in kwcoco.cli.coco_eval

    CommandLine:

        # Generate test data
        xdoctest -m kwcoco.cli.coco_eval CocoEvalCLI.main

        kwcoco eval \
            --true_dataset=$HOME/.cache/kwcoco/tests/eval/true.mscoco.json \
            --pred_dataset=$HOME/.cache/kwcoco/tests/eval/pred.mscoco.json \
            --out_dpath=$HOME/.cache/kwcoco/tests/eval/out \
            --force_pycocoutils=False \
            --area_range=all,0-4096,4096-inf

        nautilus $HOME/.cache/kwcoco/tests/eval/out
    """
    import kwimage
    import kwarray
    cli_config = CocoEvalCLI.cli(cmdline=cmdline, data=kw)
    print('cli_config = {}'.format(ub.urepr(cli_config, nl=1)))

    eval_config = ub.dict_subset(cli_config, coco_evaluator.CocoEvalConfig.__default__)

    coco_eval = coco_evaluator.CocoEvaluator(eval_config)
    coco_eval._init()

    results = coco_eval.evaluate()

    out_dpath = ub.Path(cli_config['out_dpath'])

    if cli_config['out_dpath'] == 'auto':
        metrics_fpath = out_dpath / 'metrics.json'
    else:
        metrics_fpath = ub.Path(cli_config['out_fpath'])

    print('dumping metrics_fpath = {!r}'.format(metrics_fpath))
    metrics_fpath.parent.ensuredir()
    results.dump(metrics_fpath, indent='    ')

    if cli_config['draw']:
        results.dump_figures(
            out_dpath,
            expt_title=cli_config['expt_title']
        )

    if 'coco_dset' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['coco_dset']
    elif 'sampler' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['sampler'].dset
    else:
        truth_dset = None

    if truth_dset is not None and getattr(results, 'cfsn_vecs', None):
        # FIXME: results is a MultiResult, need to patch this to loop over the
        # single results.
        print('Attempting to draw examples')
        gid_to_stats = {}
        gids, groupxs = kwarray.group_indices(results.cfsn_vecs.data['gid'])
        for gid, groupx in zip(gids, groupxs):
            true_vec = results.cfsn_vecs.data['true'][groupx]
            pred_vec = results.cfsn_vecs.data['pred'][groupx]
            is_true = (true_vec > 0)
            is_pred = (pred_vec > 0)
            has_pred = is_true & is_pred

            stats = {
                'num_assigned_pred': has_pred.sum(),
                'num_true': is_true.sum(),
                'num_pred': is_pred.sum(),
            }
            stats['frac_assigned'] = stats['num_assigned_pred'] / stats['num_true']
            gid_to_stats[gid] = stats

        set([stats['frac_assigned'] for stast in gid_to_stats.values()])
        gid = ub.argmax(gid_to_stats, key=lambda x: x['num_pred'] * x['num_true'])
        stat_gids = [gid]

        rng = kwarray.ensure_rng(None)
        random_gids = rng.choice(gids, size=5).tolist()

        found_gids = truth_dset.find_representative_images(gids)
        draw_gids = list(ub.unique(found_gids + stat_gids + random_gids))

        for gid in ub.ProgIter(draw_gids):
            truth_dets = coco_eval.gid_to_true[gid]
            pred_dets = coco_eval.gid_to_pred[gid]

            canvas = truth_dset.load_image(gid)
            canvas = truth_dets.draw_on(canvas, color='green', sseg=False)
            canvas = pred_dets.draw_on(canvas, color='blue', sseg=False)

            viz_dpath = (out_dpath / 'viz').ensuredir()
            fig_fpath = viz_dpath / 'eval-gid={}.jpg'.format(gid)
            kwimage.imwrite(fig_fpath, canvas)


if __name__ == '__main__':
    r"""
    kwcoco eval \
        --true_dataset=$HOME/.cache/kwcoco/tests/eval/true.mscoco.json \
        --pred_dataset=$HOME/.cache/kwcoco/tests/eval/pred.mscoco.json \
        --out_dpath=$HOME/.cache/kwcoco/tests/eval/out
    """
    __cli__._main()
