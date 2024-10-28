# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

import sphinx_rtd_theme

project = "ml_security"
copyright = "2024, Rui Melo"
author = "Rui Melo"
release = "0.0.5"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx_link",
    "sphinx_rtd_theme",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = [
    "structlog==21.1.0",
    "pandas==1.3.3",
    "numpy==1.21.2",
    "typing==3.7.4.3",
    "isort==5.13.2",
    "black==24.8.0",
    "torch==1.9.1",
    "einops==0.3.2",
    "tqdm==4.66.5",
    "torchvision",
    "matplotlib",
    "scikit-learn",
    "datasets",
    "pre-commit",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autosummary_generate = False
add_module_names = False
pickle_factory = None

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
