"""Inject CSS/JS into the real Streamlit parent document (not a sandboxed iframe)."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components


def inject_parent_css(css: str, *, style_id: str = "sf-parent-css") -> None:
    """Append/replace a <style> tag on window.parent.document.head.

    Also mirrors the CSS via a tiny markdown <style> fallback for hosts where
    the component iframe cannot touch parent (rare), and re-applies once after
    a short delay so Emotion styles that mount later lose the race.
    """
    payload = json.dumps(css)
    sid = json.dumps(style_id)
    components.html(
        f"""
<script>
(function () {{
  function write(doc) {{
    if (!doc || !doc.head) return false;
    var id = {sid};
    var el = doc.getElementById(id);
    if (!el) {{
      el = doc.createElement('style');
      el.id = id;
      doc.head.appendChild(el);
    }}
    el.textContent = {payload};
    return true;
  }}
  try {{
    var parentDoc = null;
    try {{ parentDoc = window.parent && window.parent.document; }} catch (e) {{ parentDoc = null; }}
    write(parentDoc);
    write(document);
    // Re-assert after Streamlit Emotion styles land
    setTimeout(function () {{ write(parentDoc); }}, 300);
    setTimeout(function () {{ write(parentDoc); }}, 1200);
  }} catch (e) {{
    console && console.warn && console.warn('inject_parent_css failed', e);
  }}
}})();
</script>
""",
        height=0,
        width=0,
    )

    # Extra fallback: some Streamlit builds still honor <style> in markdown
    try:
        st.markdown(
            f"<style id=\"{style_id}-md\">{css}</style>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def inject_parent_js(js: str, *, height: int = 0) -> None:
    """Run JS with access to window.parent (Streamlit app DOM)."""
    components.html(
        f"<script>(function(){{try{{{js}\n}}catch(e){{console.warn(e);}}}})();</script>",
        height=height,
        width=0,
    )
