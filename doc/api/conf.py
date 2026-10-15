# Sphinx configuration for the coxdev API reference.
#
# The narrative docs are a Jupyter Book 2 (MyST) site, which has no autodoc;
# this small Sphinx project builds the API reference from docstrings into
# doc/_build/html/api, next to the book. See .github/workflows/docs.yml.
#
# Build:  sphinx-build -b html doc/api doc/_build/html/api

project = 'coxdev API Reference'
author = 'Jonathan Taylor, Trevor Hastie, Balasubramanian Narasimhan'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

autodoc_default_options = {
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
napoleon_use_rtype = False

html_theme = 'sphinx_book_theme'
html_title = 'coxdev API Reference'
html_theme_options = {
    'repository_url': 'https://github.com/jonathan-taylor/coxdev',
    'use_repository_button': True,
    'home_page_in_toc': True,
}
