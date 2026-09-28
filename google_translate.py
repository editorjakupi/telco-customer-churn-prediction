"""Google Translate helpers for Streamlit apps (cookie + element.js)."""

from __future__ import annotations

import streamlit as st


def inject_google_translate(page_language: str = "sv") -> None:
    """Mount Google Translate dropdown (same idea as Gematrior)."""
    markup = f"""
<div class="notranslate" style="margin:0.35rem 0 0.75rem;">
  <div id="google_translate_element"></div>
</div>
<script>
(function() {{
  function boot() {{
    if (!window.google || !window.google.translate || !window.google.translate.TranslateElement) return;
    var el = document.getElementById('google_translate_element');
    if (!el || el.dataset.ready === '1') return;
    el.dataset.ready = '1';
    new google.translate.TranslateElement({{
      pageLanguage: '{page_language}',
      includedLanguages: 'en,sv,sq,de,fr,es,it,pt,ar,he,el,ru,zh-CN,ja,ko,tr,pl,nl,hi',
      autoDisplay: false
    }}, 'google_translate_element');
  }}
  window.googleTranslateElementInit = boot;
  if (!document.getElementById('google-translate-script')) {{
    var s = document.createElement('script');
    s.id = 'google-translate-script';
    s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    s.async = true;
    document.body.appendChild(s);
  }} else {{
    boot();
  }}
}})();
</script>
"""
    inject = getattr(st, "html", None)
    if inject:
        inject(markup)
