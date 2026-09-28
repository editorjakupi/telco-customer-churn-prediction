"""Google Translate for Streamlit — visible picker + parent-DOM element.js."""

from __future__ import annotations

import streamlit.components.v1 as components


LANGS = [
    ("", "Original"),
    ("en", "English"),
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


def inject_google_translate(page_language: str = "en") -> None:
    """Show a language select and wire Google Translate into the parent Streamlit page."""
    options = "".join(
        f'<option value="{code}">{label}</option>' for code, label in LANGS
    )
    included = ",".join(code for code, _ in LANGS if code)

    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    }}
    .wrap {{ padding: 2px 0 6px; }}
    label {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #94a3b8;
      margin-bottom: 6px;
    }}
    select {{
      width: 100%;
      min-height: 40px;
      border-radius: 8px;
      border: 1px solid rgba(148,163,184,0.45);
      background: #1e293b;
      color: #f8fafc;
      padding: 0.4rem 0.55rem;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap notranslate">
    <label for="sf-lang">Translate page</label>
    <select id="sf-lang">{options}</select>
  </div>
  <script>
  (function () {{
    var pageLang = {page_language!r};
    var included = {included!r};
    var sel = document.getElementById('sf-lang');

    function parentDoc() {{
      try {{ return window.parent.document; }} catch (e) {{ return null; }}
    }}
    function parentWin() {{
      try {{ return window.parent; }} catch (e) {{ return null; }}
    }}

    function readLang() {{
      var doc = parentDoc();
      if (!doc) return '';
      var m = doc.cookie.match(/(?:^|;\\s*)googtrans=([^;]+)/);
      if (!m) return '';
      var parts = decodeURIComponent(m[1]).split('/');
      return parts[2] || '';
    }}

    function writeLang(code) {{
      var doc = parentDoc();
      var win = parentWin();
      if (!doc || !win) return;
      var host = win.location.hostname;
      var clear = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
      doc.cookie = clear;
      if (host && host !== 'localhost') {{
        doc.cookie = clear + '; domain=' + host;
        doc.cookie = clear + '; domain=.' + host;
      }}
      if (code) {{
        var value = '/auto/' + code;
        doc.cookie = 'googtrans=' + value + '; path=/';
        if (host && host !== 'localhost') {{
          doc.cookie = 'googtrans=' + value + '; path=/; domain=' + host;
          doc.cookie = 'googtrans=' + value + '; path=/; domain=.' + host;
        }}
      }}
      win.location.reload();
    }}

    function ensureTranslateEngine() {{
      var doc = parentDoc();
      var win = parentWin();
      if (!doc || !win) return;

      function boot() {{
        var mount = doc.getElementById('google_translate_element');
        if (!mount) {{
          mount = doc.createElement('div');
          mount.id = 'google_translate_element';
          mount.style.cssText = 'position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden;';
          doc.body.appendChild(mount);
        }}
        if (mount.dataset.ready === '1') return;
        if (!(win.google && win.google.translate && win.google.translate.TranslateElement)) return;
        mount.dataset.ready = '1';
        new win.google.translate.TranslateElement({{
          pageLanguage: pageLang,
          includedLanguages: included,
          autoDisplay: false
        }}, 'google_translate_element');
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

    var current = readLang();
    if (sel) {{
      sel.value = current || '';
      sel.addEventListener('change', function () {{
        writeLang(sel.value || '');
      }});
    }}
    ensureTranslateEngine();
  }})();
  </script>
</body>
</html>
""",
        height=78,
    )
