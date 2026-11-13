"""Sphinx configuration for UniKT docs."""

import os
import sys

from docutils import nodes
from sphinx.ext.autodoc.mock import _MockObject
from sphinx.locale import init as init_locale
from sphinx.transforms import SphinxTransform
from sphinx.util.i18n import docname_to_domain
from sphinx.util.nodes import extract_messages


# Mock the __or__ and __ror__ methods of _MockObject to
# avoid errors when using the | operator in type hints.
def _mock_or(self, _other):
    return self


_MockObject.__or__ = _mock_or
_MockObject.__ror__ = _mock_or

project = "UniKT"
author = "UniKT Team"
language = os.environ.get("READTHEDOCS_LANGUAGE", "zh_CN")
project_copyright = "2025-%Y, UniKT Team"

locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = True
gettext_additional_targets = {"literal-block"}

# Let sphinx find the modules to document
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

source_suffix = ".md"

myst_enable_extensions = ["colon_fence", "dollarmath"]

# autodoc
autodoc_mock_imports = [
    "torch",
    "numpy",
    "polars",
    "scipy",
    "sklearn",
    "torch_geometric",
    "mamba_ssm",
    "causal_conv1d",
    "xlstm",
    "swanlab",
    "wandb",
    "optuna",
    "pandas",
    "matplotlib",
    "rich",
    "tqdm",
    "pyarrow",
    "jsonargparse",
    "dotenv",
    "yaml",
    "psutil",
]
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
    "pyg": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}

# linkcheck
linkcheck_ignore = [
    r"https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/|$)",
    r"https://github\.com/.*#L\d+",
    r"https://www\.sciencedirect\.com/.*",
    r"https://dl\.acm\.org/.*",
]

html_theme = "furo"
html_title = "UniKT 文档"
html_favicon = "_static/favicon.svg"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["unikt.css", "arena.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4338ca",
        "color-brand-content": "#4338ca",
        "color-brand-visited": "#5546a3",
        "color-foreground-primary": "#202b40",
        "color-foreground-secondary": "#526078",
        "color-foreground-muted": "#617087",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f6f7fb",
        "color-background-hover": "#eceef8",
        "color-background-border": "#e0e5ef",
        "color-admonition-background": "#f6f7fb",
        "unikt-accent": "#087f78",
        "unikt-brand-soft": "#eeedfc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#b3adff",
        "color-brand-content": "#b3adff",
        "color-brand-visited": "#c7b4ee",
        "color-foreground-primary": "#e1e7f1",
        "color-foreground-secondary": "#adb9ce",
        "color-foreground-muted": "#9aa9c1",
        "color-background-primary": "#141923",
        "color-background-secondary": "#1a202e",
        "color-background-hover": "#272f43",
        "color-background-border": "#343d52",
        "color-admonition-background": "#1a202e",
        "unikt-accent": "#71d6c5",
        "unikt-brand-soft": "#282541",
    },
}
pygments_style = "friendly"
pygments_dark_style = "native"

mermaid_version = "11"

exclude_patterns = ["_build", "Thumbs.db", "**/.DS_Store"]


def _page_context(app, pagename, _template, context, _doctree):
    section = pagename.split("/")[0]
    if pagename == "index":
        page_kind = "home"
    elif pagename in {"model-zoo/index", "datasets/index"}:
        page_kind = "catalog"
    elif pagename.endswith("/index"):
        page_kind = "landing"
    elif section == "model-zoo":
        page_kind = "model"
    elif section == "datasets":
        page_kind = "dataset"
    elif section == "api":
        page_kind = "api"
    else:
        page_kind = "guide"
    context["unikt_page_kind"] = page_kind


def _config_inited(app, config):
    if (
        config.language
        and config.language.startswith("en")
        and (not config.html_title or config.html_title == "UniKT 文档")
    ):
        config.html_title = "UniKT Documentation"


def setup(app):
    """Register documentation build hooks and i18n transforms."""
    app.connect("config-inited", _config_inited)
    app.connect("html-page-context", _page_context)
    app.connect("doctree-read", _mark_table_cell_sources)
    app.add_transform(_TableCellSourceTransform)
    app.add_transform(_LiteralBlockTranslationTransform)


def _mark_table_cell_sources(_app, doctree):
    # MyST table cells carry no source, so Sphinx skips them when extracting
    # gettext messages. Tag them with the document source to make them translatable.
    source = doctree["source"]
    for entry in doctree.findall(nodes.entry):
        for paragraph in entry.findall(nodes.paragraph):
            if paragraph.source is None:
                paragraph.source = source


class _TableCellSourceTransform(SphinxTransform):
    """Re-tag MyST table cells after the doctree is unpickled.

    ``source`` is not preserved by the doctree pickle, so the gettext extractor
    sees the cells only during the initial read; the i18n transform needs them
    tagged again before it runs.
    """

    default_priority = 10

    def apply(self, **_kwargs):
        source = self.document["source"]
        for entry in self.document.findall(nodes.entry):
            for paragraph in entry.findall(nodes.paragraph):
                if paragraph.source is None:
                    paragraph.source = source


class _LiteralBlockTranslationTransform(SphinxTransform):
    """Apply literal translations without parsing code as Markdown."""

    default_priority = 19

    def apply(self, **_kwargs):
        from sphinx.builders.gettext import MessageCatalogBuilder

        if isinstance(self.app.builder, MessageCatalogBuilder):
            return

        domain = docname_to_domain(self.env.docname, self.config.gettext_compact)
        directories = [
            self.env.srcdir / directory for directory in self.config.locale_dirs
        ]
        catalog, has_catalog = init_locale(directories, self.config.language, domain)
        if not has_catalog:
            return

        for node, message in extract_messages(self.document):
            if not isinstance(node, nodes.literal_block):
                continue
            translation = catalog.gettext(message)
            if not translation.strip() or translation == message:
                continue
            # Sphinx reparses translated literals as reStructuredText, but MyST
            # interprets its ``::`` wrapper as code. Preserve the original node.
            node[:] = [nodes.Text(translation)]
            node["translatable"] = False
            node["translated"] = True
