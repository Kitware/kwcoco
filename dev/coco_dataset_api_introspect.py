import kwcoco
import ubelt as ub

dset = kwcoco.CocoDataset.demo()

groups = ub.ddict(list)
groups['classmethod'] = []
groups['slots'] = []
groups['property'] = []

for key in dset.__dict__.keys():
    if key.startswith('_'):
        pass
    else:
        groups['slots'].append(key)
for key in dir(dset):
    if key.startswith('_'):
        continue

    if key in groups['slots']:
        continue

    cls_val = getattr(kwcoco.CocoDataset, key, ub.NoParam)
    self_val = getattr(dset, key)

    import inspect
    # inspect.ismethod(cls_val), inspect.ismethod(self_val), inspect.ismethod(self_val), inspect.ismethod(self_val)
    if cls_val is ub.NoParam:
        heuristic = 'unknown'
    elif hasattr(cls_val, 'fset'):
        heuristic = 'property'
    elif callable(cls_val) and callable(self_val):
        if self_val.__self__ is kwcoco.CocoDataset:
            heuristic = 'classmethod'
        elif inspect.ismethod(self_val):
            self_val.__func__
            self_val.__doc__
            self_val.__self__
            self_val.__name__
            cands = [cls for cls in kwcoco.CocoDataset.__mro__ if hasattr(cls, key)]
            method_definer = cands[-1].__name__.split('.')[-1]
            heuristic = 'method(via {})'.format(method_definer)
        else:
            heuristic = 'callable'
    else:
        heuristic = None
    groups[heuristic].append(key)

print('For Reference, the following are grouped attributes/methods of a kwcoco.CocoDataset')
print('{}'.format(ub.repr2(groups, sort=0, nl=2)))


# import redbaron
# baron = redbaron.RedBaron(open(kwcoco.coco_dataset.__file__, 'r').read())

"""
Naming crisis: I want to add a new way of manipulating images in kwcoco. Forcing the user to deal with raw image dictionaries via `kwcoco.index.imgs` and `kwcoco.dataset['images']` has been useful in forcing users to become familiar with backend datastructure, and the `kwcoco.images(<list-of-image-ids>)` provides some simple tooling for performing lookups on "columns" of image properties using a vectorized-like interface (via `class Images(ObjectList1D)`), but now I want to add a new object-oriented interface to doing more complex things with a single image. This will keep prevent the API overhead on the raw `kwcoco.CocoDataset` from growing - the class is over 5000 lines long (including comments, doctests, and demodata, so its bad but not that bad). But I do need to add at least one new method that allows the user to construct an instance of this object-oriented interface, and I'm not sure what to call it.

The method will accept a single image id, and then construct a new class `CocoImage` that has access to the image dictionary (and optionally a pointer to the parent dataset), and then return it. I need to find a good name for this method
"""
