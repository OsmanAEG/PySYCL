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
import furo
import os
import sys
sys.path.insert(0, os.path.abspath('../build/'))

# -- Project information -----------------------------------------------------

project = 'pysycl'
copyright = '2023, Osman El-Ghotmi'
author = 'Osman El-Ghotmi'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinxcontrib.doxylink']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme_options = {
    'collapse_navigation': True,
    'navigation_depth': 4,
    'github_url': '',
    'navbar_start': ["navbar-logo"],
    'navbar_center': ["navbar-nav"],
    'navbar_end': ["navbar-icon-links"],
    'navbar_persistent': ["search-button"],
    'navbar_align': "content"
}

highlight_language = 'python'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "_static/images/pysycl_logo.png"
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = "colorful"

