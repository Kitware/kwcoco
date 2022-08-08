#!/bin/bash
kwcoco toydata --key=vidshapes8-tracks4-frames16-speed0.05-gsize224 --dst=mytoybundle/dataset.kwcoco.json --verbose=1

# TODO: this will be ported to kwcoco eventually
smartwatch visualize mytoybundle/dataset.kwcoco.json --animate="fps: 3"


python -m kwplot.cli.gifify -i /home/joncrall/code/kwcoco/mytoybundle/_viz_mytoybundle_dataset.kwcoco_b638d3a0/toy_video_3/_anns/r_g_b/ -o foo.gif --frames_per_second=3
