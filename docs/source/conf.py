# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath('../../main/'))  # Adjust the path as necessary
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Capstone Project - DRL for automated trading'
copyright = '2024, Amine, Leopold, Julius, Loic, Lina, Louis'
author = 'Amine, Leopold, Julius, Loic, Lina, Louis'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google or NumPy style docstrings
    'sphinx_autodoc_typehints'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set a more modern theme
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
