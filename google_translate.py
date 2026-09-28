"""Google Translate for Streamlit — theme-aware picker, English source/default."""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

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


def inject_google_translate(page_language: str = "en", *, theme: str = "light") -> None:
    """Visible translate control. Source language is English; Google handles the rest."""
    is_dark = theme == "dark"
    bg = "#1e293b" if is_dark else "#ffffff"
    fg = "#f8fafc" if is_dark else "#0f172a"
    border = "rgba(148,163,184,0.45)" if is_dark else "rgba(15,23,42,0.18)"
    label = "#94a3b8" if is_dark else "#475569"
    options_html = "".join(
        f'<option value="{code}">{label_}</option>' for code, label_ in LANGS
    )
    included = ",".join(c for c, _ in LANGS if c)
    page_language_js = json.dumps(page_language)
    included_js = json.dumps(included)

    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html,body{{margin:0;padding:0;background:transparent;font-family:system-ui,sans-serif;}}
  .wrap{{padding:2px 0 4px;}}
  label{{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:{label};margin:0 0 6px;}}
  select{{
    width:100%;min-height:42px;border-radius:8px;border:1px solid {border};
    background:{bg};color:{fg};padding:.4rem .55rem;font-size:14px;
  }}
</style>
</head>
<body>
<div class="wrap notranslate">
  <label for="sf-lang">Translate page</label>
  <select id="sf-lang">{options_html}</select>
</div>
<script>
(function () {{
  var pageLang = {page_language_js};
  var included = {included_js};
  var sel = document.getElementById('sf-lang');

  function pDoc() {{ try {{ return window.parent.document; }} catch (e) {{ return null; }} }}
  function pWin() {{ try {{ return window.parent; }} catch (e) {{ return null; }} }}

  function readLang() {{
    var doc = pDoc();
    if (!doc) return '';
    var m = doc.cookie.match(/(?:^|;\\s*)googtrans=([^;]+)/);
    if (!m) return '';
    var parts = decodeURIComponent(m[1]).split('/');
    return parts[2] || '';
  }}

  function setCookie(code) {{
    var doc = pDoc();
    var win = pWin();
    if (!doc || !win) return;
    var host = win.location.hostname || '';
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
    }}
  }}

  function applyCombo(code) {{
    var doc = pDoc();
    if (!doc) return false;
    var combo = doc.querySelector('select.goog-te-combo');
    if (!combo) return false;
    combo.value = code || '';
    combo.dispatchEvent(new Event('change'));
    return true;
  }}

  function ensureEngine(cb) {{
    var doc = pDoc();
    var win = pWin();
    if (!doc || !win) return;
    function boot() {{
      var mount = doc.getElementById('google_translate_element');
      if (!mount) {{
        mount = doc.createElement('div');
        mount.id = 'google_translate_element';
        mount.style.cssText = 'position:fixed;left:-9999px;top:0;width:1px;height:1px;opacity:0;pointer-events:none;';
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
      if (typeof cb === 'function') setTimeout(cb, 250);
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
  }}

  function switchTo(code) {{
    setCookie(code);
    ensureEngine(function () {{
      if (!applyCombo(code)) {{
        var win = pWin();
        if (win) {{
          if (code) win.location.hash = 'googtrans(' + pageLang + '|' + code + ')';
          else win.location.hash = '';
          win.location.reload();
        }}
      }}
    }});
  }}

  var current = readLang();
  if (sel) {{
    sel.value = current || '';
    sel.addEventListener('change', function () {{
      switchTo(sel.value || '');
    }});
  }}
  ensureEngine(function () {{
    if (current) applyCombo(current);
  }});
}})();
</script>
</body>
</html>
""",
        height=84,
    )


def render_translate_sidebar(page_language: str = "en", *, theme: Optional[str] = None) -> None:
    """Sidebar block: heading + themed Google Translate picker."""
    st.markdown("### Translate")
    inject_google_translate(page_language=page_language, theme=theme or "light")
