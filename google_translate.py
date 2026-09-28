"""Google Translate for Streamlit — custom HTML picker (not BaseWeb).

Important: on Streamlit Cloud the component iframe's window.top is the outer
shell (cross-origin). Always use window.parent (the app frame), never top.
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


def render_translate_sidebar(page_language: str = "en", *, theme: Optional[str] = None) -> None:
    """Sidebar: heading + themed native select that drives Google Translate."""
    inject_react_dom_patch()

    is_dark = (theme or "light") == "dark"
    bg = "#1e293b" if is_dark else "#ffffff"
    fg = "#f8fafc" if is_dark else "#0f172a"
    border = "rgba(148,163,184,0.45)" if is_dark else "rgba(15,23,42,0.22)"
    label = "#94a3b8" if is_dark else "#475569"
    chevron = (
        "data:image/svg+xml,"
        + "%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E"
        + f"%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='{fg.replace('#', '%23')}' stroke-width='1.8' "
        + "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"
    )

    options_html = "".join(
        f'<option value="{code}">{label_}</option>' for code, label_ in LANGS
    )
    included = ",".join(c for c, _ in LANGS if c)
    page_language_js = json.dumps(page_language)
    included_js = json.dumps(included)

    st.markdown("### Translate")
    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {{
    margin: 0; padding: 0; background: transparent;
    font-family: system-ui, -apple-system, sans-serif;
  }}
  .wrap {{ padding: 2px 0 6px; }}
  label {{
    display: block; font-size: 11px; letter-spacing: .06em;
    text-transform: uppercase; color: {label}; margin: 0 0 6px;
  }}
  select {{
    width: 100%; min-height: 42px; box-sizing: border-box;
    border-radius: 10px; border: 1px solid {border};
    background-color: {bg};
    background-image: url("{chevron}");
    background-repeat: no-repeat;
    background-position: right 12px center;
    background-size: 12px 8px;
    color: {fg};
    padding: .45rem 2rem .45rem .7rem;
    font-size: 14px; font-weight: 500;
    appearance: none; -webkit-appearance: none; -moz-appearance: none;
    cursor: pointer;
  }}
  select:focus {{ outline: 2px solid {fg}; outline-offset: 1px; }}
  option {{ background: {bg}; color: {fg}; }}
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

  // Prefer parent (Streamlit app frame). top is often the Cloud shell (cross-origin).
  function pWin() {{
    try {{
      var d = window.parent && window.parent.document;
      if (d) return window.parent;
    }} catch (e) {{}}
    try {{
      var d2 = window.top && window.top.document;
      if (d2) return window.top;
    }} catch (e2) {{}}
    return null;
  }}
  function pDoc() {{
    var w = pWin();
    return w ? w.document : null;
  }}

  function readLang() {{
    var doc = pDoc();
    var win = pWin();
    if (doc) {{
      var m = doc.cookie.match(/(?:^|;\\s*)googtrans=([^;]+)/);
      if (m) {{
        var parts = decodeURIComponent(m[1]).split('/');
        if (parts[2]) return parts[2];
      }}
    }}
    try {{
      if (win) {{
        var qp = new URL(win.location.href).searchParams.get('lang');
        if (qp) return qp;
      }}
    }} catch (e) {{}}
    return '';
  }}

  function setCookie(code) {{
    var doc = pDoc();
    var win = pWin();
    if (!doc || !win) return false;
    var host = win.location.hostname || '';
    var clear = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
    doc.cookie = clear;
    if (host && host !== 'localhost' && host !== '127.0.0.1') {{
      doc.cookie = clear + '; domain=' + host;
      doc.cookie = clear + '; domain=.' + host;
    }}
    if (code) {{
      var value = '/' + pageLang + '/' + code;
      doc.cookie = 'googtrans=' + value + '; path=/';
      if (host && host !== 'localhost' && host !== '127.0.0.1') {{
        doc.cookie = 'googtrans=' + value + '; path=/; domain=' + host;
        doc.cookie = 'googtrans=' + value + '; path=/; domain=.' + host;
      }}
    }}
    return true;
  }}

  function applyCombo(code) {{
    var doc = pDoc();
    if (!doc) return false;
    var combo = doc.querySelector('select.goog-te-combo');
    if (!combo) return false;
    var want = code || '';
    combo.value = want;
    combo.dispatchEvent(new Event('change', {{ bubbles: true }}));
    try {{
      combo.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }} catch (e) {{}}
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
        mount.className = 'notranslate';
        mount.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;opacity:0.01;pointer-events:none;z-index:-1;overflow:visible;';
        doc.body.appendChild(mount);
      }}
      if (mount.dataset.ready !== '1' && win.google && win.google.translate && win.google.translate.TranslateElement) {{
        try {{
          mount.innerHTML = '';
          mount.dataset.ready = '1';
          new win.google.translate.TranslateElement({{
            pageLanguage: pageLang,
            includedLanguages: included,
            autoDisplay: false
          }}, 'google_translate_element');
        }} catch (e) {{
          mount.dataset.ready = '0';
        }}
      }}
      if (typeof cb === 'function') setTimeout(cb, 350);
    }}
    win.googleTranslateElementInit = boot;
    if (!doc.getElementById('google-translate-script')) {{
      var s = doc.createElement('script');
      s.id = 'google-translate-script';
      s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
      s.async = true;
      s.onload = function () {{ setTimeout(boot, 80); }};
      doc.body.appendChild(s);
    }} else {{
      boot();
    }}
    var hide = doc.getElementById('sf-gt-hide-banner');
    if (!hide) {{
      hide = doc.createElement('style');
      hide.id = 'sf-gt-hide-banner';
      hide.textContent = '.goog-te-banner-frame,body>{{.skiptranslate,iframe.goog-te-banner-frame,.VIpgJd-ZVi9od-ORHb-OEVmcd,#goog-gt-tt{{display:none!important;visibility:hidden!important;}}body{{top:0!important;position:static!important;}}';
      doc.head.appendChild(hide);
    }}
  }}

  function reloadApp(code) {{
    var win = pWin();
    if (!win) return;
    var url = new URL(win.location.href);
    url.searchParams.delete('_gt');
    if (code) {{
      url.searchParams.set('lang', code);
      url.hash = 'googtrans(' + pageLang + '|' + code + ')';
    }} else {{
      url.searchParams.delete('lang');
      url.hash = '';
    }}
    url.searchParams.set('_gt', String(Date.now()));
    win.location.href = url.toString();
  }}

  function switchTo(code) {{
    setCookie(code);
    // Reload the Streamlit app frame so Google reads googtrans on boot.
    reloadApp(code);
  }}

  var current = readLang();
  if (sel) {{
    sel.value = current || '';
    sel.addEventListener('change', function () {{
      switchTo(sel.value || '');
    }});
  }}

  ensureEngine(function () {{
    if (!current) return;
    var tries = 0;
    function tick() {{
      tries += 1;
      if (applyCombo(current) || tries > 12) return;
      setTimeout(tick, 400);
    }}
    tick();
  }});
}})();
</script>
</body>
</html>
""",
        height=88,
    )
