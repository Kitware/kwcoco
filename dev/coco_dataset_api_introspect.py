import kwcoco
import ubelt as ub

dset = kwcoco.CocoDataset.demo()

groups = ub.ddict(lambda: ub.ddict(list))
groups['classmethods'] = ub.ddict(list)
groups['slots'] = ub.ddict(list)
groups['properties'] = ub.ddict(list)
groups['methods'] = ub.ddict(list)

for key in dset.__dict__.keys():
    method_definer = None
    if key.startswith('_'):
        pass
    else:
        groups['slots'][method_definer].append(key)

for key in dir(dset):
    if key.startswith('_'):
        continue

    if key in groups['slots'][None]:
        continue

    method_definer = None
    cls_val = getattr(kwcoco.CocoDataset, key, ub.NoParam)
    self_val = getattr(dset, key)

    import inspect
    # inspect.ismethod(cls_val), inspect.ismethod(self_val), inspect.ismethod(self_val), inspect.ismethod(self_val)
    if cls_val is ub.NoParam:
        heuristic = 'unknown'
    elif hasattr(cls_val, 'fset'):
        heuristic = 'properties'
    elif callable(cls_val) and callable(self_val):
        if self_val.__self__ is kwcoco.CocoDataset:
            cands = [cls for cls in kwcoco.CocoDataset.__mro__ if hasattr(cls, key)]
            method_definer = cands[-1].__name__.split('.')[-1]
            heuristic = 'method (via {})'.format(method_definer)
            heuristic = 'classmethods'
        elif inspect.ismethod(self_val):
            self_val.__func__
            self_val.__doc__
            self_val.__self__
            self_val.__name__
            cands = [cls for cls in kwcoco.CocoDataset.__mro__ if hasattr(cls, key)]
            method_definer = cands[-1].__name__.split('.')[-1]
            heuristic = 'methods'
        else:
            heuristic = 'callable'
    else:
        heuristic = None
    groups[heuristic][method_definer].append(key)

print('For Reference, the following are grouped attributes/methods of a kwcoco.CocoDataset')
print('{}'.format(ub.urepr(groups, sort=0, nl=3)))

# Try and make a nice RTD RST style

autogen = []
aprint = autogen.append

aprint('CocoDataset API')
aprint('###############')
aprint('')
aprint(ub.paragraph(
    '''
    The following is a logical grouping of the public kwcoco.CocoDataset API
    attributes and methods.  See the in-code documentation for further details.
    '''))

for group, def_to_items in groups.items():

    for definer, items in def_to_items.items():

        aprint('')
        subtitle = 'CocoDataset ' + group
        if definer is not None:
            subtitle = subtitle + ' (via {})'.format(definer)

        aprint(subtitle)
        aprint('*' * len(subtitle))
        aprint('')

        if definer is None:
            definer = 'CocoDataset'

        for item in items:
            try:
                doclines = []
                for line in getattr(kwcoco.CocoDataset, item).__doc__.split('\n')[1:]:
                    line = line.strip()
                    if not line:
                        break
                    doclines.append(line)
                docline = ' '.join(doclines)
                if docline.startswith('Example'):
                    docline = ''
            except Exception:
                docline = ''
            if group in {'slots', 'properties'}:
                reffer = ':attr:'
            else:
                reffer = ':func:'

            # https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html
            # https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
            line = f' * {reffer}`.CocoDataset.{item}<kwcoco.coco_dataset.{definer}.{item}>` - {docline}'
            aprint(line)

print('\n'.join(autogen))


# import redbaron
# baron = redbaron.RedBaron(open(kwcoco.coco_dataset.__file__, 'r').read())

"""
Naming crisis: I want to add a new way of manipulating images in kwcoco. Forcing the user to deal with raw image dictionaries via `kwcoco.index.imgs` and `kwcoco.dataset['images']` has been useful in forcing users to become familiar with backend datastructure, and the `kwcoco.images(<list-of-image-ids>)` provides some simple tooling for performing lookups on "columns" of image properties using a vectorized-like interface (via `class Images(ObjectList1D)`), but now I want to add a new object-oriented interface to doing more complex things with a single image. This will keep prevent the API overhead on the raw `kwcoco.CocoDataset` from growing - the class is over 5000 lines long (including comments, doctests, and demodata, so its bad but not that bad). But I do need to add at least one new method that allows the user to construct an instance of this object-oriented interface, and I'm not sure what to call it.

The method will accept a single image id, and then construct a new class `CocoImage` that has access to the image dictionary (and optionally a pointer to the parent dataset), and then return it. I need to find a good name for this method
"""
