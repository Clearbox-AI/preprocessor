# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Clearbox Preprocessor'
copyright = '2024, Clearbox AI'
author = 'Dario Brunelli'
release = '0.9.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.coverage",
    'sphinx.ext.napoleon',
    "myst_parser",
    'sphinx_rtd_theme'
]

myst_enable_extensions = [
    "html_image", # Allows html images format conversion
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "style_nav_header_background": "#483a8f",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "img/cb_white_logo_compact.png"
html_static_path = ['_static', 'img']

master_doc = 'index'  # Ensure this points to your main document

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True