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
    __alias__ = ['eval_detections', 'evaluate_detections']

    # These should go into the CLI args, not the class config args
    expt_title = scfg.Value('', type=str, help='title for plots', tags=['perf_param'])
    draw = scfg.Value(True, isflag=1, help='draw metric plots', tags=['perf_param'])
    out_dpath = scfg.Value('./coco_metrics', type=str, help='where to dump results', tags=['out_path'])
    out_fpath = scfg.Value('auto', type=str, help=ub.paragraph(
        '''
        Where to dump the json file containing result. If "auto",
        defaults to out_dpath / "metrics.json"
        '''), tags=['out_path', 'primary'])

    confusion_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, write a kwcoco file with confusion labels for further
        inspection and visualization. A script to perform visualization will
        also be written as a sidecar. Note, that multiple confusion files may
        be written if multiple thresholds are were requested, and this path
        will symlink to that final path.
        '''))

    @classmethod
    def main(cls, cmdline=True, **kw):
        """

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> from kwcoco.cli.coco_eval import *  # NOQA
            >>> import ubelt as ub
            >>> import kwcoco
            >>> from kwcoco.demo.perterb import perterb_coco
            >>> dpath = ub.Path.appdir('kwcoco/tests/eval').ensuredir()
            >>> true_dset = kwcoco.CocoDataset.demo('shapes8')
            >>> kwargs = {
            >>>     'box_noise': 0.5,
            >>>     'n_fp': (0, 10),
            >>>     'n_fn': (0, 10),
            >>> }
            >>> pred_dset = perterb_coco(true_dset, **kwargs)
            >>> true_dset.fpath = dpath / 'true.mscoco.json'
            >>> pred_dset.fpath = dpath / 'pred.mscoco.json'
            >>> confusion_fpath = dpath / 'confusion.kwcoco.json'
            >>> true_dset.dump(true_dset.fpath)
            >>> pred_dset.dump(pred_dset.fpath)
            >>> draw = False  # set to false for faster tests
            >>> kw = dict(
            >>>     true_dataset=true_dset.fpath,
            >>>     pred_dataset=pred_dset.fpath,
            >>>     confusion_fpath=confusion_fpath,
            >>>     draw=draw,
            >>>     out_dpath=dpath
            >>> )
            >>> cmdline = False
            >>> CocoEvalCLI.main(cmdline=cmdline, **kw)

        Ignore:
            geowatch visualize ~/.cache/kwcoco/tests/eval/confusion.kwcoco.json --smart --animate=False --role_order="true,pred"
            geowatch visualize ~/.cache/kwcoco/tests/eval/confusion.kwcoco.json --smart --animate=False --channels="r|g|b,r|g|b" --role_order="true,pred"

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

    DO_CONFUSION_DUMP = cli_config.confusion_fpath is not None
    if DO_CONFUSION_DUMP:
        print('Dumping confusion kwcoco files')
        key_to_cfsn_dset = build_confusion_datasets(coco_eval, results)

        # This is a little strange because the CLI lets the user request a
        # single confusion path, but we might need to output multiple for
        # multi-threshold results. To handle this we will augment the requested
        # path, but to ensure output existence checks are met we will symlink
        # the final one to the original path.
        requested_confusion_fpath = ub.Path(cli_config.confusion_fpath)
        confusion_fpath = None

        for key, cfsn_dset in key_to_cfsn_dset.items():
            # augment confusion path based on key
            confusion_fpath = ub.Path(requested_confusion_fpath)
            confusion_fpath = confusion_fpath.augment(stemsuffix='-' + key, multidot=True)

            try:
                import kwutil
                new_name = kwutil.util_path.sanitize_path_name(confusion_fpath.name, safe=True)
                confusion_fpath = confusion_fpath.parent / new_name
            except ImportError:
                ...

            cfsn_dset.fpath = confusion_fpath
            cfsn_dset.dump()

            # Also write a script to visualize the confusion kwcoco file to disk.
            WRITE_VIZ_SCRIPT = True
            if WRITE_VIZ_SCRIPT:
                # Also write the confusion viz script, requires geowatch is
                # installed (in the future geowatch visualize should be moved
                # to kwcoco proper)
                viz_script_fpath = confusion_fpath.augment(prefix='visualize_', ext='.sh')
                channels = cfsn_dset.coco_image(ub.peek(cfsn_dset.imgs)).channels
                channel_spec = channels.spec
                viz_script_text = ub.codeblock(
                    rf'''
                    #!/bin/bash
                    geowatch visualize {cfsn_dset.fpath} --smart --animate=False \
                        --channels "{channel_spec},{channel_spec}" \
                        --role_order="true,pred" \
                        --draw_imgs=False --draw_anns=True \
                        --max_dim=640 --draw_chancode=False \
                        --draw_header=False
                    ''')
                viz_script_fpath.write_text(viz_script_text)
                try:
                    viz_script_fpath.chmod('+x')
                except Exception:
                    ...

        assert confusion_fpath is not None
        ub.symlink(real_path=confusion_fpath,
                   link_path=requested_confusion_fpath,
                   overwrite=True,
                   verbose=1)

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

    try:
        from rich import print as rich_print
    except ImportError:
        rich_print = print
    else:
        rich_print(f'Eval Dpath: [link={out_dpath}]{out_dpath}[/link]')


def build_confusion_datasets(coco_eval, results):
    """
    For each result build the confusion dataset
    """
    import kwimage  # NOQA
    # CONFUSION_COLORS = {
    #     'pred_false_positive': kwimage.Color.coerce('kitware_red').ashex(),
    #     'pred_true_positive': kwimage.Color.coerce('kitware_blue').ashex(),
    #     'true_false_negative': kwimage.Color.coerce('purple').ashex(),
    #     'true_true_positive': kwimage.Color.coerce('kitware_green').ashex(),
    # }
    CONFUSION_COLORS = {
        'pred_false_positive': '#f42836',
        'pred_true_positive': '#0068c7',
        'true_false_negative': '#800080',
        'true_true_positive': '#3eae2b',
        None: '#242a37',
    }

    pred_dset = coco_eval.pred_extra['coco_dset']
    true_dset = coco_eval.true_extra['coco_dset']
    dmet = coco_eval._dmet

    key_to_cfsn_dset = {}

    assert len(results) == 1, 'not impl'
    for key, result in results.items():

        # TODO: add an info section about the confusion

        cfsn_dset = pred_dset.copy()
        cfsn_dset.reroot(absolute=True)

        # Ensure true and pred categories exist in the cfsn dataset
        for cat in true_dset.categories().objs:
            new_cat = ub.udict(cat) - {'id'}
            cfsn_dset.ensure_category(**new_cat)

        # Mark the predicted annotations in the confusion dataset
        for ann in cfsn_dset.annots().objs_iter():
            ann.update({
                'role': 'pred',
                'confusion': None,
                'color': CONFUSION_COLORS[None],
            })

        cfsn_vecs = result.cfsn_vecs
        bin_vecs = cfsn_vecs.binarize_classless()
        for gid, tx, px in zip(bin_vecs.data['gid'], bin_vecs.data['txs'], bin_vecs.data['pxs']):

            pred_gid = gid  # is this the pred gid?

            true_dets = dmet.gid_to_true_dets[gid]
            pred_dets = dmet.gid_to_pred_dets[gid]
            true_aid = true_dets.data['aids'][tx] if tx >= 0 else None
            pred_aid = pred_dets.data['aids'][px] if px >= 0 else None

            pred_ann = None
            true_ann = None
            if pred_aid is not None:
                pred_ann = cfsn_dset.index.anns[pred_aid]
                pred_ann['role'] = 'pred'
                if true_aid is None:
                    pred_ann['confusion'] = 'false_positive'
                    pred_ann['color'] = CONFUSION_COLORS['pred_false_positive']
                else:
                    pred_ann['confusion'] = 'true_positive'
                    pred_ann['color'] = CONFUSION_COLORS['pred_true_positive']

            if true_aid is not None:
                true_ann = true_dset.index.anns[true_aid].copy()
                true_ann.pop('id')
                true_ann['role'] = 'true'
                true_ann['image_id'] = pred_gid

                if pred_aid is None:
                    true_ann['confusion'] = 'false_negative'
                    true_ann['color'] = CONFUSION_COLORS['true_false_negative']
                else:
                    true_ann['confusion'] = 'true_positive'
                    true_ann['score'] = pred_ann.get('score', None)
                    true_ann['matching_pred_annot_id'] = pred_ann['id']
                    true_ann['color'] = CONFUSION_COLORS['true_true_positive']

                # Update the category id (requires categories were ported)
                old_true_cat_id = true_ann.pop('category_id')
                old_cat = true_dset.index.cats[old_true_cat_id]
                catname = old_cat['name']
                new_cat = cfsn_dset.index.name_to_cat[catname]
                true_ann['category_id'] = new_cat['id']

                cfsn_dset.add_annotation(**true_ann)

        key_to_cfsn_dset[key] = cfsn_dset
    return key_to_cfsn_dset


if __name__ == '__main__':
    r"""
    kwcoco eval \
        --true_dataset=$HOME/.cache/kwcoco/tests/eval/true.mscoco.json \
        --pred_dataset=$HOME/.cache/kwcoco/tests/eval/pred.mscoco.json \
        --out_dpath=$HOME/.cache/kwcoco/tests/eval/out
    """
    __cli__._main()
