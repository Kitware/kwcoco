"""
Wraps the logic in kwcoco/coco_evaluator.py with a command line script
"""
from kwcoco import coco_evaluator
import scriptconfig as scfg
import ubelt as ub
from os.path import join


class CocoEvalCLIConfig(scfg.Config):
    __doc__ = coco_evaluator.CocoEvalConfig.__doc__

    default = ub.dict_union(coco_evaluator.CocoEvalConfig.default, {
        # These should go into the CLI args, not the class config args
        'expt_title': scfg.Value('', type=str, help='title for plots'),
        'draw': scfg.Value(True, help='draw metric plots'),
        'out_dpath': scfg.Value('./coco_metrics', type=str),
    })


class CocoEvalCLI:
    name = 'eval'

    CLIConfig = CocoEvalCLIConfig

    @classmethod
    def main(cls, cmdline=True, **kw):
        """

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import ubelt as ub
            >>> from kwcoco.cli.coco_eval import *  # NOQA
            >>> from os.path import join
            >>> import kwcoco
            >>> dpath = ub.ensure_app_cache_dir('kwcoco/tests/eval')
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
            >>> CocoEvalCLI.main(
            >>>     true_dataset=true_dset.fpath,
            >>>     pred_dataset=pred_dset.fpath,
            >>>     draw=draw)
        """
        main(cmdline=True, **kw)


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
    cli_config = CocoEvalCLIConfig(cmdline=cmdline, default=kw)
    print('cli_config = {}'.format(ub.repr2(dict(cli_config), nl=1)))

    eval_config = ub.dict_subset(cli_config, coco_evaluator.CocoEvaluator.Config.default)

    coco_eval = coco_evaluator.CocoEvaluator(eval_config)
    coco_eval._init()

    results = coco_eval.evaluate()

    ub.ensuredir(cli_config['out_dpath'])

    metrics_fpath = join(cli_config['out_dpath'], 'metrics.json')
    print('dumping metrics_fpath = {!r}'.format(metrics_fpath))
    results.dump(metrics_fpath, indent='    ')

    if cli_config['draw']:
        results.dump_figures(
            cli_config['out_dpath'],
            expt_title=cli_config['expt_title']
        )

    if 'coco_dset' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['coco_dset']
    elif 'sampler' in coco_eval.true_extra:
        truth_dset = coco_eval.true_extra['sampler'].dset
    else:
        truth_dset = None

    if truth_dset is not None and getattr(results, 'cfsn_vecs', None):
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

            viz_dpath = ub.ensuredir((cli_config['out_dpath'], 'viz'))
            fig_fpath = join(viz_dpath, 'eval-gid={}.jpg'.format(gid))
            kwimage.imwrite(fig_fpath, canvas)


_CLI = CocoEvalCLI


if __name__ == '__main__':
    """
    kwcoco eval \
        --true_dataset=$HOME/.cache/kwcoco/tests/eval/true.mscoco.json \
        --pred_dataset=$HOME/.cache/kwcoco/tests/eval/pred.mscoco.json \
        --out_dpath=$HOME/.cache/kwcoco/tests/eval/out
    """
    _CLI._main()
