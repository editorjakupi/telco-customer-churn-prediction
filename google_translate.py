"""Google Translate for Streamlit — native sidebar picker + cookie reload.

Uses a visible Streamlit selectbox (always renders) and only mutates the parent
DOM for the hidden Google engine. Cookie + full reload avoids fragile combo
events; a React removeChild patch prevents Translate+Streamlit crashes.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from streamlit_parent_inject import inject_react_dom_patch

LANGS = [
    ("", "English"),
    ("sv", "Svenska"),
    ("sq", "Shqip"),
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("ar", "العربية"),
    ("he", "עברית"),
    ("el", "Ελληνικά"),
    ("ru", "Русский"),
    ("zh-CN", "中文"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("tr", "Türkçe"),
    ("pl", "Polski"),
    ("nl", "Nederlands"),
    ("hi", "हिन्दी"),
]

LANG_LABELS = [label for _, label in LANGS]
LANG_CODES = [code for code, _ in LANGS]


def _boot_engine(page_language: str = "en") -> None:
    """Mount Google Translate element.js on the parent document + apply cookie."""
    included = ",".join(c for c in LANG_CODES if c)
    page_language_js = json.dumps(page_language)
    included_js = json.dumps(included)
    components.html(
        f"""
<script>
(function () {{
  var pageLang = {page_language_js};
  var included = {included_js};
  function pDoc() {{ try {{ return window.parent.document; }} catch (e) {{ return null; }} }}
  function pWin() {{ try {{ return window.parent; }} catch (e) {{ return null; }} }}
  var doc = pDoc();
  var win = pWin();
  if (!doc || !win) return;

  function readLang() {{
    var m = doc.cookie.match(/(?:^|;\\s*)googtrans=([^;]+)/);
    if (!m) return '';
    var parts = decodeURIComponent(m[1]).split('/');
    return parts[2] || '';
  }}

  function applyCombo(code) {{
    var combo = doc.querySelector('select.goog-te-combo');
    if (!combo) return false;
    var want = code || '';
    if (combo.value === want) return true;
    combo.value = want;
    combo.dispatchEvent(new Event('change'));
    return true;
  }}

  function boot() {{
    var mount = doc.getElementById('google_translate_element');
    if (!mount) {{
      mount = doc.createElement('div');
      mount.id = 'google_translate_element';
      mount.className = 'notranslate';
      mount.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;opacity:0.01;pointer-events:none;z-index:-1;';
      doc.body.appendChild(mount);
    }}
    if (mount.dataset.ready !== '1' && win.google && win.google.translate && win.google.translate.TranslateElement) {{
      mount.dataset.ready = '1';
      new win.google.translate.TranslateElement({{
        pageLanguage: pageLang,
        includedLanguages: included,
        autoDisplay: false
      }}, 'google_translate_element');
    }}
    var code = readLang();
    if (code) {{
      setTimeout(function () {{ applyCombo(code); }}, 200);
      setTimeout(function () {{ applyCombo(code); }}, 900);
    }}
  }}

  win.googleTranslateElementInit = boot;
  if (!doc.getElementById('google-translate-script')) {{
    var s = doc.createElement('script');
    s.id = 'google-translate-script';
    s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    s.async = true;
    doc.body.appendChild(s);
  }} else {{
    boot();
  }}

  // Hide Google's top banner so it never covers Streamlit UI
  var style = doc.getElementById('sf-gt-hide-banner');
  if (!style) {{
    style = doc.createElement('style');
    style.id = 'sf-gt-hide-banner';
    style.textContent = '.goog-te-banner-frame,body>{{.skiptranslate,iframe.goog-te-banner-frame,.VIpgJd-ZVi9od-ORHb-OEVmcd,#goog-gt-tt{{display:none!important;visibility:hidden!important;}}body{{top:0!important;position:static!important;}}';
    doc.head.appendChild(style);
  }}
}})();
</script>
""",
        height=1,
        width=1,
    )


def _set_cookie_and_reload(code: str, page_language: str = "en") -> None:
    code_js = json.dumps(code or "")
    page_language_js = json.dumps(page_language)
    components.html(
        f"""
<script>
(function () {{
  var code = {code_js};
  var pageLang = {page_language_js};
  var doc, win;
  try {{ doc = window.parent.document; win = window.parent; }} catch (e) {{ return; }}
  var host = (win.location && win.location.hostname) || '';
  var clear = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
  doc.cookie = clear;
  if (host && host !== 'localhost') {{
    doc.cookie = clear + '; domain=' + host;
    doc.cookie = clear + '; domain=.' + host;
  }}
  if (code) {{
    var value = '/' + pageLang + '/' + code;
    doc.cookie = 'googtrans=' + value + '; path=/';
    if (host && host !== 'localhost') {{
      doc.cookie = 'googtrans=' + value + '; path=/; domain=' + host;
      doc.cookie = 'googtrans=' + value + '; path=/; domain=.' + host;
    }}
    win.location.hash = 'googtrans(' + pageLang + '|' + code + ')';
  }} else {{
    win.location.hash = '';
  }}
  win.location.reload();
}})();
</script>
""",
        height=1,
        width=1,
    )


def render_translate_sidebar(page_language: str = "en", *, theme: Optional[str] = None) -> None:
    """Sidebar: heading + native Streamlit language select (Google Translate engine)."""
    del theme  # theme reserved for future chrome styling
    inject_react_dom_patch()
    _boot_engine(page_language=page_language)

    st.markdown("### Translate")
    st.caption("Google Translate · page language")

    # Prefer cookie via query param mirror so select stays in sync after reload
    qp = st.query_params
    current = ""
    if "lang" in qp:
        current = str(qp.get("lang") or "")
    if current not in LANG_CODES:
        current = ""

    idx = LANG_CODES.index(current) if current in LANG_CODES else 0
    choice = st.selectbox(
        "Translate page",
        LANG_LABELS,
        index=idx,
        key="sf_google_translate_lang",
        help="Choose a language — the page reloads with Google Translate.",
    )
    code = LANG_CODES[LANG_LABELS.index(choice)]

    if code != current:
        st.query_params["lang"] = code if code else None
        if not code and "lang" in st.query_params:
            del st.query_params["lang"]
        _set_cookie_and_reload(code, page_language=page_language)
        st.info("Applying translation…")
        st.stop()
