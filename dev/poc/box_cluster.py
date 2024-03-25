import kwcoco
import ubelt as ub
# import kwarray
import numpy as np


def main():
    """
    Requirements:
        pip install finch-clust
        pip install pynndescent
    """
    # hard coded path
    self = kwcoco.CocoDataset('/data/joncrall/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip')

    cname_to_box_sizes = ub.ddict(list)

    gids = self.images()

    if gids is not None:
        aids = ub.flatten(ub.take(self.index.gid_to_aids, gids))
    if aids is not None:
        anns = ub.take(self.anns, aids)
    else:
        anns = self.dataset.get('annotations', [])

    for ann in anns:
        if 'bbox' in ann:
            cname = self.cats[ann['category_id']]['name']
            cname_to_box_sizes[cname].append(ann['bbox'][2:4])
    cname_to_box_sizes = ub.map_vals(np.array, cname_to_box_sizes)
    box_sizes = np.vstack(list(cname_to_box_sizes.values()))

    import kwplot
    kwplot.autompl()

    import kwarray
    unique_boxes = kwarray.unique_rows(box_sizes)
    print(len(unique_boxes))

    # Get kmeans cluster labels (chosen K as 10)
    from sklearn import cluster
    num_anchors = 10
    defaultkw = {
        'n_clusters': num_anchors,
        'n_init': 20,
        'max_iter': 1000,
        'tol': 1e-6,
        'algorithm': 'elkan',
        'verbose': 1
    }
    kmkw = ub.dict_union(defaultkw, {})
    algo = cluster.KMeans(**kmkw)
    algo.fit(unique_boxes)

    # Get finch cluster labels
    from finch import FINCH
    c, num_clust, req_c = FINCH(unique_boxes, distance='cosine', req_clust=10)

    unique_groups, groupxs = kwarray.group_indices(c.T[-1])
    finch_centroids = []
    for idxs in groupxs:
        centroid = unique_boxes[idxs].mean(axis=0)
        finch_centroids.append(centroid)

    unique_groups, groupxs = kwarray.group_indices(algo.labels_)
    kmeans_centroids = []
    for idxs in groupxs:
        centroid = unique_boxes[idxs].mean(axis=0)
        kmeans_centroids.append(centroid)

    import kwplot
    import pandas as pd
    sns = kwplot.autosns()
    data = pd.DataFrame({'width': unique_boxes.T[0], 'height': unique_boxes.T[1]})
    data['kmeans_label'] = [f'km-{label:02d}' for label in algo.labels_]
    data['finch_label'] = [f'fc-{label:02d}' for label in c.T[-1]]

    ax1 = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=1).gca()
    sns.scatterplot(data=data, x='width', y='height', hue='kmeans_label', ax=ax1)
    ax1.set_title('KMeans Clusters (K=10)')

    ax2 = kwplot.figure(fnum=1, pnum=(1, 2, 2)).gca()
    sns.scatterplot(data=data, x='width', y='height', hue='finch_label', ax=ax2)
    ax2.set_title('Finch Clusters')

    ax1.scatter(*np.array(kmeans_centroids).T, s=300, color='orange')
    ax2.scatter(*np.array(finch_centroids).T, s=300, color='orange')
