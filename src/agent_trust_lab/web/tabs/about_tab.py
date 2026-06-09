"""Tab 3: About — project overview and trap library statistics."""

import gradio as gr

from agent_trust_lab.web.i18n import t as _t


def build_about_tab(_i18n_outputs: list, about_trailer: str, **kwargs) -> None:
    """Build the About tab inside the current Gradio context.

    Args:
        _i18n_outputs: Shared list — append every Gradio component here
            in creation order so the language-switch callback can update them.
        about_trailer: Trailer text with trap/category statistics appended
            to the about markdown. Computed in ``create_ui()`` so it can
            be reused by the language-switch callback.
    """
    about_md = gr.Markdown(_t("md_about", "en") + about_trailer)
    _i18n_outputs.append(about_md)
