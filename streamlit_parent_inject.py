"""Inject CSS/JS into the real Streamlit parent document (not a sandboxed iframe)."""

from __future__ import annotations

import json

import streamlit.components.v1 as components


def inject_parent_css(css: str, *, style_id: str = "sf-parent-css") -> None:
    """Append/replace a <style> tag on window.parent.document.head."""
    payload = json.dumps(css)
    sid = json.dumps(style_id)
    components.html(
        f"""
<script>
(function () {{
  try {{
    var doc = window.parent.document;
    if (!doc || !doc.head) return;
    var id = {sid};
    var el = doc.getElementById(id);
    if (!el) {{
      el = doc.createElement('style');
      el.id = id;
      doc.head.appendChild(el);
    }}
    el.textContent = {payload};
  }} catch (e) {{
    console && console.warn && console.warn('inject_parent_css failed', e);
  }}
}})();
</script>
""",
        height=0,
    )


def inject_parent_js(js: str, *, height: int = 0) -> None:
    """Run JS with access to window.parent (Streamlit app DOM)."""
    components.html(f"<script>(function(){{try{{{js}\n}}catch(e){{console.warn(e);}}}})();</script>", height=height)
