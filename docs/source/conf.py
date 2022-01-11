"""
Notes:
    http://docs.readthedocs.io/en/latest/getting_started.html

    pip install sphinx sphinx-autobuild sphinx_rtd_theme sphinxcontrib-napoleon
    pip install sphinx-autoapi

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
            * Click "Build pull requests for this project", then click save

        ensure your github account is connected to readthedocs
        https://readthedocs.org/accounts/social/connections/


        # Add RTD as a GitLab Hook

        https://docs.readthedocs.io/en/stable/webhooks.html

        https://gitlab.kitware.com/computer-vision/kwcoco/-/settings/integrations

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
modname = project
copyright = '2021, Kitware Inc'
author = 'Jon Crall'

modpath = join(dirname(dirname(dirname(__file__))), 'kwcoco', '__init__.py')
# The full version, including alpha/beta/rc tags
release = parse_version(modpath)
# version = '.'.join(release.split('.')[0:2])
version = release


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
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# TODO: figure out how to use autoapi
USE_AUTOAPI = True
if USE_AUTOAPI:
    # redirects = {
    #     "index": "autoapi/kwcoco/index.html",
    # }

    autoapi_modules = {
        modname: {
            'override': False,
            'output': 'auto'
        }
    }

    autoapi_dirs = [f'../../{modname}']
    autoapi_keep_files = True

    extensions.extend([
        'autoapi.extension',
        # 'sphinx.ext.inheritance_diagram',
        # 'autoapi.sphinx',
    ])

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
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
    # '.txt': 'markdown',
    '.md': 'markdown',
}


pygments_style = 'sphinx'


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    # 'pytorch': ('http://pytorch.org/docs/master/', None),
    'python': ('https://docs.python.org/3', None),
    'click': ('https://click.palletsprojects.com/', None),
    # 'xxhash': ('https://pypi.org/project/xxhash/', None),
    # 'pygments': ('pygments.github.io/en/latest', None),

    # Requries that the repo have objects.inv
    # 'tqdm': ('https://tqdm.github.io', None),
    'kwarray': ('https://kwarray.readthedocs.io/en/latest/', None),
    'kwimage': ('https://kwimage.readthedocs.io/en/latest/', None),
    # 'kwplot': ('https://kwplot.readthedocs.io/en/latest/', None),
    'ndsampler': ('https://ndsampler.readthedocs.io/en/latest/', None),
    'ubelt': ('https://ubelt.readthedocs.io/en/latest/', None),
    'xdoctest': ('https://xdoctest.readthedocs.io/en/latest/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'scriptconfig': ('https://scriptconfig.readthedocs.io/en/latest/', None),
}

__dev_note__ = """
python -m sphinx.ext.intersphinx https://kwarray.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://kwimage.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://ubelt.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://networkx.org/documentation/stable/objects.inv
"""


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

from sphinx.domains.python import PythonDomain  # NOQA


class PatchedPythonDomain(PythonDomain):
    """
    References:
        https://github.com/sphinx-doc/sphinx/issues/3866
    """
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        # if target.startswith('kwcoco') or 'CategoryTree' in target:
        if target == 'CategoryTree' or target == 'kwcoco.CategoryTree':
            target = 'kwcoco.category_tree.CategoryTree'

        # if 0:
        #     if 'CategoryTree' in target:
        #         import ubelt as ub
        #         print('')
        #         print('========')
        #         print('Maybe fix?')
        #         print(target)
        #         print('contnode = {!r}'.format(contnode))
        #         print('node = {!r}'.format(node))
        #         print('target = {!r}'.format(target))
        #         print('typ = {!r}'.format(typ))
        #         print('builder = {!r}'.format(builder))
        #         print('fromdocname = {!r}'.format(fromdocname))
        #         print('env = {!r}'.format(env))
        #         print('node.__dict__ = {}'.format(ub.repr2(node.__dict__, nl=1)))
        #         print('contnode.__dict__ = {}'.format(ub.repr2(contnode.__dict__, nl=1)))
        #         print('--')
        #         # if 'refspecific' in node:
        #         #     del node['refspecific']
        return_value = super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)
        # if 'CategoryTree' in target:
        #     print('return_value = {!r}'.format(return_value))
        #     if return_value is not None:
        #         print('return_value.__dict__ = {}'.format(ub.repr2(return_value.__dict__, nl=1)))
        #     print('========')
        #     print('')
        return return_value


def setup(app):
    app.add_domain(PatchedPythonDomain, override=True)

    if 1:
        # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
        from sphinx.application import Sphinx
        from typing import Any, List

        what = None
        # Custom process to transform docstring lines
        # Remove "Ignore" blocks
        def process(app: Sphinx, what_: str, name: str, obj: Any, options: Any, lines: List[str]
                    ) -> None:
            if what and what_ not in what:
                return
            orig_lines = lines[:]

            # text = '\n'.join(lines)
            # if 'Example' in text and 'CommandLine' in text:
            #     import xdev
            #     xdev.embed()

            ignore_tags = tuple(['Ignore'])

            mode = None
            # buffer = None
            new_lines = []
            for i, line in enumerate(orig_lines):

                # See if the line triggers a mode change
                if line.startswith(ignore_tags):
                    mode = 'ignore'
                elif line.startswith('CommandLine'):
                    mode = 'cmdline'
                elif line and not line.startswith(' '):
                    # if the line startswith anything but a space, we are no
                    # longer in the previous nested scope
                    mode = None

                if mode is None:
                    new_lines.append(line)
                elif mode == 'ignore':
                    pass
                elif mode == 'cmdline':
                    if line.startswith('CommandLine'):
                        new_lines.append('.. rubric:: CommandLine')
                        new_lines.append('')
                        new_lines.append('.. code-block:: bash')
                        new_lines.append('')
                        # new_lines.append('    # CommandLine')
                    else:
                        # new_lines.append(line.strip())
                        new_lines.append(line)
                else:
                    raise KeyError(mode)

            lines[:] = new_lines
            # make sure there is a blank line at the end
            if lines and lines[-1]:
                lines.append('')

        app.connect('autodoc-process-docstring', process)
    else:
        # https://stackoverflow.com/questions/26534184/can-sphinx-ignore-certain-tags-in-python-docstrings
        # Register a sphinx.ext.autodoc.between listener to ignore everything
        # between lines that contain the word IGNORE
        # from sphinx.ext.autodoc import between
        # app.connect('autodoc-process-docstring', between('^ *Ignore:$', exclude=True))
        pass

    return app
