"""Force Streamlit widget colors + scrollbars via parent-DOM JS (beats Emotion/inline)."""

from __future__ import annotations

import streamlit.components.v1 as components


def inject_theme_force(theme: str, *, accent: str = "#2dd4bf", accent_fg: str = "#042f2e") -> None:
    """Keep selects/inputs/uploader readable in light and dark."""
    is_dark = theme == "dark"
    text = "#f1f5f9" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#475569"
    surface = "#1e293b" if is_dark else "#ffffff"
    surface2 = "#0f172a" if is_dark else "#f8fafc"
    border = "rgba(148,163,184,0.45)" if is_dark else "rgba(15,23,42,0.16)"
    track = "#0b1220" if is_dark else "#e2e8f0"
    thumb = "#64748b" if is_dark else "#94a3b8"

    components.html(
        f"""
<script>
(function () {{
  try {{
    var doc = window.parent.document;
    if (!doc) return;
    var STYLE_ID = 'sf-theme-force-style';
    var css = `
      html, body, .stApp {{ color-scheme: {'dark' if is_dark else 'light'} !important; }}
      * {{
        scrollbar-width: thin;
        scrollbar-color: {thumb} {track};
      }}
      *::-webkit-scrollbar {{ width: 12px; height: 12px; }}
      *::-webkit-scrollbar-track {{ background: {track}; }}
      *::-webkit-scrollbar-thumb {{
        background: {thumb};
        border-radius: 8px;
        border: 2px solid {track};
      }}
      section[data-testid="stSidebar"] *::-webkit-scrollbar-thumb {{
        background: {accent};
      }}
    `;
    var style = doc.getElementById(STYLE_ID);
    if (!style) {{
      style = doc.createElement('style');
      style.id = STYLE_ID;
      doc.head.appendChild(style);
    }}
    style.textContent = css;

    function paint(el, bg, fg) {{
      if (!el || !el.style) return;
      el.style.setProperty('background-color', bg, 'important');
      el.style.setProperty('background-image', 'none', 'important');
      el.style.setProperty('color', fg, 'important');
      el.style.setProperty('-webkit-text-fill-color', fg, 'important');
      el.style.setProperty('caret-color', fg, 'important');
      el.style.setProperty('border-color', '{border}', 'important');
      el.style.setProperty('opacity', '1', 'important');
    }}

    function force() {{
      var text = '{text}';
      var muted = '{muted}';
      var surface = '{surface}';
      var surface2 = '{surface2}';
      var accent = '{accent}';
      var accentFg = '{accent_fg}';

      doc.querySelectorAll(
        '[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, ' +
        '.stSelectbox label, .stSelectbox label *, .stSlider label, .stSlider label *, ' +
        '.stRadio label, .stRadio label *, .stFileUploader label, .stFileUploader label *, ' +
        'section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, ' +
        'section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li, ' +
        '[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"], ' +
        '[data-testid="stCaptionContainer"] *'
      ).forEach(function (n) {{
        n.style.setProperty('color', text, 'important');
        n.style.setProperty('-webkit-text-fill-color', text, 'important');
        n.style.setProperty('opacity', '1', 'important');
      }});

      doc.querySelectorAll(
        '.stTextInput input, .stNumberInput input, .stTextArea textarea, ' +
        'div[data-baseweb="select"] > div, div[data-baseweb="base-input"], ' +
        'div[data-baseweb="input"], div[data-baseweb="input"] > div, ' +
        '[data-baseweb="input"] input, input, textarea, select'
      ).forEach(function (n) {{ paint(n, surface, text); }});

      doc.querySelectorAll('div[data-baseweb="select"] span, div[data-baseweb="select"] div').forEach(function (n) {{
        n.style.setProperty('color', text, 'important');
        n.style.setProperty('-webkit-text-fill-color', text, 'important');
      }});

      doc.querySelectorAll(
        '[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"]'
      ).forEach(function (n) {{ paint(n, surface2, text); }});

      doc.querySelectorAll(
        '[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"], ' +
        '[data-testid="stFileUploaderDropzone"] > div'
      ).forEach(function (n) {{ paint(n, surface, text); }});

      doc.querySelectorAll(
        '[data-testid="stFileUploader"] button, [data-testid="stFileUploaderDropzone"] button, ' +
        '[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]'
      ).forEach(function (btn) {{
        paint(btn, accent, accentFg);
        btn.style.setProperty('font-weight', '700', 'important');
        btn.querySelectorAll('*').forEach(function (child) {{
          child.style.setProperty('color', accentFg, 'important');
          child.style.setProperty('-webkit-text-fill-color', accentFg, 'important');
        }});
      }});

      /* Kill leftover light cards in dark mode */
      if ({str(is_dark).lower()}) {{
        doc.querySelectorAll(
          '[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stMetric"], ' +
          '[data-testid="stExpander"], .stAlert'
        ).forEach(function (n) {{
          var bg = getComputedStyle(n).backgroundColor;
          if (bg && (bg.indexOf('255, 255, 255') >= 0 || bg.indexOf('248, 249, 251') >= 0 || bg === 'rgb(255, 255, 255)')) {{
            n.style.setProperty('background-color', surface, 'important');
            n.style.setProperty('color', text, 'important');
          }}
        }});
      }}
    }}

    force();
    var win = window.parent;
    if (!win.__sfThemeForceObs) {{
      win.__sfThemeForceObs = new MutationObserver(function () {{ force(); }});
      win.__sfThemeForceObs.observe(doc.body, {{ childList: true, subtree: true }});
    }}
    setInterval(force, 1200);
  }} catch (e) {{
    console && console.warn && console.warn('theme force failed', e);
  }}
}})();
</script>
""",
        height=0,
    )
