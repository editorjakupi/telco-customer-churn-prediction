"""Inject CSS/JS into the real Streamlit parent document (not a sandboxed iframe)."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components


def inject_parent_css(css: str, *, style_id: str = "sf-parent-css") -> None:
    """Write a <style> tag on window.parent.document.head.

    IMPORTANT: height/width must be >= 1 — zero-size iframes often never run
    their scripts on Streamlit Cloud, which left themes/translate broken.
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
    setTimeout(function () {{ write(parentDoc); }}, 200);
    setTimeout(function () {{ write(parentDoc); }}, 800);
    setTimeout(function () {{ write(parentDoc); }}, 2000);
  }} catch (e) {{
    console && console.warn && console.warn('inject_parent_css failed', e);
  }}
}})();
</script>
""",
        height=1,
        width=1,
    )


def inject_parent_js(js: str, *, height: int = 1) -> None:
    """Run JS with access to window.parent (Streamlit app DOM)."""
    components.html(
        f"<script>(function(){{try{{{js}\n}}catch(e){{console.warn(e);}}}})();</script>",
        height=max(1, height),
        width=1,
    )


def inject_react_dom_patch() -> None:
    """Prevent Google Translate + React removeChild / insertBefore crashes."""
    inject_parent_js(
        """
var win = window.parent || window;
if (win.__sfDomPatch) return;
win.__sfDomPatch = true;
var doc = win.document;
function patch(proto, name) {
  var original = proto[name];
  if (!original || original.__sfPatched) return;
  var wrapped = function (a, b) {
    try {
      if (name === 'removeChild') {
        if (a && a.parentNode !== this) return a;
      } else if (name === 'insertBefore') {
        if (b && b.parentNode !== this) return a;
      }
      return original.call(this, a, b);
    } catch (e) {
      return a;
    }
  };
  wrapped.__sfPatched = true;
  proto[name] = wrapped;
}
patch(win.Node.prototype, 'removeChild');
patch(win.Node.prototype, 'insertBefore');
"""
    )
