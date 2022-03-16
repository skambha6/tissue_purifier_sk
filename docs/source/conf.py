import os
import sys
import sphinx_rtd_theme

dir_, _ = os.path.split(__file__)  # current directory of this file
root_dir = os.path.abspath(os.path.join(dir_, '../../src'))
sys.path.insert(0, root_dir)
print(root_dir)
os.environ["SPHINX_BUILD"] = "1"

# -- Project information -----------------------------------------------------

project = u"TissuePurifier"
copyright = u""
author = u"Luca Dalessio"

version = ""
if "READTHEDOCS" not in os.environ:
    from tissue_purifier import __version__  # noqaE402
    version = __version__
    html_context = {"github_version": "master"}
# release version
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
needs_sphinx = '4.0.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxarg.ext',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.programoutput',
    'sphinx.ext.intersphinx'
]

master_doc = 'index'

autodoc_inherit_docstrings = True

# Napoleon settings (for google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Make the return section behave like the args section
napoleon_custom_sections = [('Returns', 'params_style')]

PACKAGE_MAPPING = {
    "pyro-ppl": "pyro",
    "PyYAML": "yaml",
    "neptune-client": "neptune",
    "google-cloud": "google",
    "scikit-learn": "sklearn",
    "umap_learn": "umap",
    "pytorch-lightning": "pytorch_lightning",
    "lightning-bolts": "pl_bolts",
}
MOCK_PACKAGES = [
    'numpy',
    'anndata',
    'scanpy',
    'leidenalg',
    'igraph',
    'pyro-ppl',
    'google-cloud',
    'scikit-learn',
    'pytorch-lightning',
    'torch',
    'torchvision',
    'matplotlib',
    'umap_learn',
    'neptune-client',
    'scipy',
    'pandas',
    'lightly',
    'lightning-bolts',
    'seaborn',
]
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]
autodoc_mock_imports = MOCK_PACKAGES

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = [".rst", ".ipynb"]
nbsphinx_execute = "never"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    ".ipynb_checkpoints",
    "notebooks/*ipynb",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# do not prepend module name to functions
add_module_names = False

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = ""

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = [] #["_static"]
html_style = "css/tissuepurifier.css"
htmlhelp_basename = "tissuepurifierdoc"