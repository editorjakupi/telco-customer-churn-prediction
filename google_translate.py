"""Google Translate for Streamlit — custom HTML picker (not BaseWeb).

Streamlit Cloud: the component iframe can write parent cookies but often cannot
navigate the app frame. A <script> injected into the parent document owns
cookie + reload + Google Translate boot so language changes apply immediately.
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

# Runs as a real script in the Streamlit app document (parent), not the component iframe.
_PARENT_BRIDGE_JS = r"""
(function () {
  var pageLang = window.__sfGtPageLang || "en";
  var included = window.__sfGtIncluded || "";

  function readLang() {
    var m = document.cookie.match(/(?:^|;\s*)googtrans=([^;]+)/);
    if (m) {
      var p = decodeURIComponent(m[1]).split("/");
      if (p[2]) return p[2];
    }
    try {
      var q = new URL(location.href).searchParams.get("lang");
      if (q) return q;
    } catch (e) {}
    return "";
  }

  function setCookie(code) {
    pageLang = window.__sfGtPageLang || pageLang;
    var host = location.hostname || "";
    var clear = "googtrans=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    document.cookie = clear;
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      document.cookie = clear + "; domain=" + host;
      document.cookie = clear + "; domain=." + host;
    }
    if (code) {
      var value = "/" + pageLang + "/" + code;
      document.cookie = "googtrans=" + value + "; path=/";
      if (host && host !== "localhost" && host !== "127.0.0.1") {
        document.cookie = "googtrans=" + value + "; path=/; domain=" + host;
        document.cookie = "googtrans=" + value + "; path=/; domain=." + host;
      }
    }
  }

  function applyCombo(code) {
    var combo = document.querySelector("select.goog-te-combo");
    if (!combo) return false;
    combo.value = code || "";
    combo.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function hideBanner() {
    if (document.getElementById("sf-gt-hide-banner")) return;
    var hide = document.createElement("style");
    hide.id = "sf-gt-hide-banner";
    hide.textContent =
      ".goog-te-banner-frame,body>.skiptranslate,iframe.goog-te-banner-frame,.VIpgJd-ZVi9od-ORHb-OEVmcd,#goog-gt-tt{display:none!important;visibility:hidden!important;}body{top:0!important;position:static!important;}";
    document.head.appendChild(hide);
  }

  function boot(cb) {
    pageLang = window.__sfGtPageLang || pageLang;
    included = window.__sfGtIncluded || included;
    hideBanner();
    var mount = document.getElementById("google_translate_element");
    if (!mount) {
      mount = document.createElement("div");
      mount.id = "google_translate_element";
      mount.className = "notranslate";
      mount.style.cssText =
        "position:fixed;left:0;top:0;width:1px;height:1px;opacity:0.01;pointer-events:none;z-index:-1;";
      document.body.appendChild(mount);
    }
    function init() {
      if (
        mount.dataset.ready !== "1" &&
        window.google &&
        google.translate &&
        google.translate.TranslateElement
      ) {
        try {
          mount.innerHTML = "";
          mount.dataset.ready = "1";
          new google.translate.TranslateElement(
            {
              pageLanguage: pageLang,
              includedLanguages: included,
              autoDisplay: false,
            },
            "google_translate_element"
          );
        } catch (e) {
          mount.dataset.ready = "0";
        }
      }
      if (typeof cb === "function") setTimeout(cb, 300);
    }
    window.googleTranslateElementInit = init;
    if (!document.getElementById("google-translate-script")) {
      var s = document.createElement("script");
      s.id = "google-translate-script";
      s.src =
        "https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
      s.async = true;
      s.onload = function () {
        setTimeout(init, 50);
      };
      document.body.appendChild(s);
    } else {
      init();
    }
  }

  window.__sfGtBoot = boot;

  window.__sfGtSwitch = function (code) {
    setCookie(code || "");
    var url = new URL(location.href);
    url.searchParams.delete("_gt");
    if (code) {
      url.searchParams.set("lang", code);
      url.hash =
        "googtrans(" + (window.__sfGtPageLang || pageLang) + "|" + code + ")";
    } else {
      url.searchParams.delete("lang");
      url.hash = "";
    }
    url.searchParams.set("_gt", String(Date.now()));
    location.replace(url.toString());
  };

  window.addEventListener("message", function (ev) {
    var d = ev && ev.data;
    if (!d || d.type !== "sf-gt-switch") return;
    window.__sfGtSwitch(d.code || "");
  });

  boot(function () {
    var current = readLang();
    if (!current) return;
    var tries = 0;
    (function tick() {
      tries += 1;
      if (applyCombo(current) || tries > 15) return;
      setTimeout(tick, 350);
    })();
  });
})();
"""


def _install_parent_bridge(page_language: str, included: str) -> None:
    """Install GT controller as a real parent-document script (not iframe JS)."""
    page_language_js = json.dumps(page_language)
    included_js = json.dumps(included)
    bridge_js = json.dumps(_PARENT_BRIDGE_JS)
    components.html(
        f"""
<script>
(function () {{
  var parentWin = null;
  try {{ parentWin = window.parent; }} catch (e) {{ return; }}
  if (!parentWin || !parentWin.document) return;

  parentWin.__sfGtPageLang = {page_language_js};
  parentWin.__sfGtIncluded = {included_js};

  if (parentWin.__sfGtBridgeInstalled) {{
    try {{ parentWin.__sfGtBoot && parentWin.__sfGtBoot(); }} catch (e) {{}}
    return;
  }}
  parentWin.__sfGtBridgeInstalled = true;

  var script = parentWin.document.createElement('script');
  script.id = 'sf-gt-bridge';
  script.textContent = {bridge_js};
  parentWin.document.documentElement.appendChild(script);
}})();
</script>
""",
        height=1,
        width=1,
    )


def render_translate_sidebar(page_language: str = "en", *, theme: Optional[str] = None) -> None:
    """Sidebar: heading + themed native select that drives Google Translate."""
    inject_react_dom_patch()

    included = ",".join(c for c, _ in LANGS if c)
    _install_parent_bridge(page_language, included)

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
    page_language_js = json.dumps(page_language)

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
  var sel = document.getElementById('sf-lang');

  function pWin() {{
    try {{ if (window.parent && window.parent.document) return window.parent; }} catch (e) {{}}
    return null;
  }}

  function readLang() {{
    var win = pWin();
    if (!win) return '';
    try {{
      var m = win.document.cookie.match(/(?:^|;\\s*)googtrans=([^;]+)/);
      if (m) {{
        var parts = decodeURIComponent(m[1]).split('/');
        if (parts[2]) return parts[2];
      }}
    }} catch (e) {{}}
    try {{
      var qp = new URL(win.location.href).searchParams.get('lang');
      if (qp) return qp;
    }} catch (e2) {{}}
    return '';
  }}

  function switchTo(code) {{
    var win = pWin();
    try {{
      if (win && typeof win.__sfGtSwitch === 'function') {{
        win.__sfGtSwitch(code || '');
        return;
      }}
    }} catch (e) {{}}
    try {{
      if (win) win.postMessage({{ type: 'sf-gt-switch', code: code || '', pageLang: pageLang }}, '*');
    }} catch (e2) {{}}
  }}

  var current = readLang();
  if (sel) {{
    sel.value = current || '';
    sel.addEventListener('change', function () {{
      switchTo(sel.value || '');
    }});
  }}

  try {{
    var win = pWin();
    if (win && typeof win.__sfGtBoot === 'function') win.__sfGtBoot();
  }} catch (e) {{}}
}})();
</script>
</body>
</html>
""",
        height=88,
    )
