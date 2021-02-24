# -*- coding: utf-8 -*-
"""
Notes:
    http://docs.readthedocs.io/en/latest/getting_started.html

    pip install sphinx sphinx-autobuild sphinx_rtd_theme sphinxcontrib-napoleon

    cd ~/code/kwcoco
    mkdir docs
    cd docs

    sphinx-quickstart

    # need to edit the conf.py

    cd ~/code/kwcoco/docs
    make html
    sphinx-apidoc -f -o ~/code/kwcoco/docs/source ~/code/kwcoco/kwcoco --separate
    make html


    Also:
        To turn on PR checks

        https://docs.readthedocs.io/en/stable/guides/autobuild-docs-for-pull-requests.html

        https://readthedocs.org/dashboard/kwcoco/advanced/

        ensure your github account is connected to readthedocs
        https://readthedocs.org/accounts/social/connections/

"""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sphinx_rtd_theme
from os.path import exists
from os.path import dirname
from os.path import join


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


# -- Project information -----------------------------------------------------

project = 'kwcoco'
copyright = '2020, kitware'
author = 'joncrall'

modpath = join(dirname(dirname(dirname(__file__))), 'kwcoco', '__init__.py')
# The full version, including alpha/beta/rc tags
release = parse_version(modpath)
version = '.'.join(release.split('.')[0:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme  # NOQA
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
source_suffix = ['.rst', '.md']

pygments_style = 'sphinx'


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/3/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True
autodoc_inherit_docstrings = False
autodoc_member_order = 'bysource'

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    # 'logo_only': True,
}

master_doc = 'index'
