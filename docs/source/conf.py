# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Continuous CyberBattleSim'
copyright = '2025, Franco Terranova, Abdelkader Lahmadi, Isabelle Chrisment'
author = 'Franco Terranova, Abdelkader Lahmadi, Isabelle Chrisment'
release = '1.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

master_doc = 'index'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_static_path = ['_static']
html_css_files = ['custom.css']

def setup(app):
    app.add_css_file('css/custom.css')  # Link the CSS file
