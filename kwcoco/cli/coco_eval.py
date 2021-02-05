from kwcoco import coco_evaluator


class CocoEvalCLI:
    name = 'eval'

    CLIConfig = coco_evaluator.CocoEvalCLIConfig

    @classmethod
    def main(cls, cmdline=True, **kw):
        """

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import ubelt as ub
            >>> from kwcoco.cli.coco_eval import *  # NOQA
            >>> from kwcoco.coco_evaluator import CocoEvaluator
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
        coco_evaluator.main(cmdline=True, **kw)


_CLI = CocoEvalCLI


if __name__ == '__main__':
    """
    kwcoco eval \
        --true_dataset=$HOME/.cache/kwcoco/tests/eval/true.mscoco.json \
        --pred_dataset=$HOME/.cache/kwcoco/tests/eval/pred.mscoco.json \
        --out_dpath=$HOME/.cache/kwcoco/tests/eval/out
    """
    _CLI._main()
