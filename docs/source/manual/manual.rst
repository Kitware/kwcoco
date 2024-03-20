
Manual Docs
===========

.. python -c "if 1:
.. import xdev
.. import ubelt as ub
.. dpath = ub.Path('.').absolute()
.. walker = xdev.cli.dirstats.DirectoryWalker(dpath).build()
.. for node in walker.graph.nodes:
.. if node.suffix in {'.rst', '.md'}:
..     rel_fpath = node.relative_to(dpath)
..     print(rel_fpath.parent / rel_fpath.stem)
.. "

.. toctree::
    getting_started
    on_autocomplete
    concepts/gotchas
    concepts/index
    concepts/vectorized_interface
    concepts/warping_and_spaces
    concepts/bundle_dpath
