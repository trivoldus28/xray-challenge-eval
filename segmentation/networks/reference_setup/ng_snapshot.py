import daisy
import neuroglancer
import sys

# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/funlib.show.neuroglancer_v2')
from funlib.show.neuroglancer import add_layer
from funlib.show.neuroglancer.contrib import make_viewer

f = sys.argv[1]
raw = daisy.open_ds(f, 'volumes/raw')
labels = daisy.open_ds(f, 'volumes/labels/neuron_ids')
gt_affs = daisy.open_ds(f, 'volumes/gt_affinities')
affs = daisy.open_ds(f, 'volumes/pred_affinities')
labels_mask = daisy.open_ds(f, 'volumes/labels/mask')
unlabeled_mask = daisy.open_ds(f, 'volumes/labels/unlabeled')
try:
    affs_gradient = daisy.open_ds(f, 'volumes/affs_gradient')
except:
    pass

viewer = make_viewer(port_range=(33400, 33500), token='snapshot')

with viewer.txn() as s:

    add_layer(s, raw, 'raw')
    add_layer(s, labels, 'labels')
    #add_layer(s, gt_myelin_embedding, 'gt_myelin')
    #add_layer(s, myelin_embedding, 'pred_myelin')
    add_layer(s, gt_affs, 'gt_affs', shader='rgb', visible=False)
    add_layer(s, affs, 'pred_affs', shader='rgb', visible=True)
    try:
        add_layer(s, affs_gradient, 'affs_gradient', shader='rgb', visible=False)
    except:
        pass
    add_layer(s, labels_mask, 'labels_mask', shader='mask', visible=False)
    add_layer(s, unlabeled_mask, 'unlabeled_mask', shader='mask', visible=False)

    s.layout = 'xy'
    # s.navigation.zoomFactor=1.5
    s.projectionScale = 256
    s.crossSectionScale = .15
    
print(viewer)
