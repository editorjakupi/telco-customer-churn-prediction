"""Force Streamlit widget colors via CSS only (no MutationObserver — that caused flash)."""

from __future__ import annotations

from streamlit_parent_inject import inject_parent_css


def inject_theme_force(theme: str, *, accent: str = "#2dd4bf", accent_fg: str = "#042f2e") -> None:
    """Keep selects/inputs/uploader readable in light and dark — CSS only."""
    is_dark = theme == "dark"
    text = "#f1f5f9" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#475569"
    surface = "#1e293b" if is_dark else "#ffffff"
    surface2 = "#0f172a" if is_dark else "#f8fafc"
    border = "rgba(148,163,184,0.45)" if is_dark else "rgba(15,23,42,0.16)"
    track = "#0b1220" if is_dark else "#e2e8f0"
    thumb = "#64748b" if is_dark else "#94a3b8"
    scheme = "dark" if is_dark else "light"

    css = f"""
html, body, .stApp {{ color-scheme: {scheme} !important; }}
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

/* Widget labels — always readable */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
.stSelectbox label, .stSelectbox label *,
.stSlider label, .stSlider label *,
.stRadio label, .stRadio label *,
.stFileUploader label, .stFileUploader label *,
.stMultiSelect label, .stMultiSelect label *,
.stTextInput label, .stNumberInput label, .stTextArea label {{
  color: {text} !important;
  -webkit-text-fill-color: {text} !important;
  opacity: 1 !important;
  visibility: visible !important;
}}

/* Inputs / BaseWeb selects — never leave white fields on dark */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
[data-baseweb="input"] input,
.stApp input,
.stApp textarea,
.stApp select {{
  background-color: {surface} !important;
  background-image: none !important;
  color: {text} !important;
  -webkit-text-fill-color: {text} !important;
  caret-color: {text} !important;
  border-color: {border} !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
  color: {text} !important;
  -webkit-text-fill-color: {text} !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"],
body > div[data-baseweb="popover"],
body > div[data-baseweb="popover"] * {{
  background-color: {surface2} !important;
  color: {text} !important;
  -webkit-text-fill-color: {text} !important;
}}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {{
  background-color: {surface} !important;
  color: {text} !important;
  border-color: {border} !important;
}}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {{
  background-color: {accent} !important;
  color: {accent_fg} !important;
  -webkit-text-fill-color: {accent_fg} !important;
  font-weight: 700 !important;
}}
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploaderDropzone"] button * {{
  color: {accent_fg} !important;
  -webkit-text-fill-color: {accent_fg} !important;
}}

/* Sidebar / caption copy */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
  color: {text} !important;
  -webkit-text-fill-color: {text} !important;
  opacity: 1 !important;
}}
section[data-testid="stSidebar"] .nav-hint,
[data-testid="stCaptionContainer"] {{
  color: {muted} !important;
  -webkit-text-fill-color: {muted} !important;
}}

/* Cards / metrics — no white flash on dark */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stMetric"],
[data-testid="stExpander"] {{
  background-color: {surface} !important;
  color: {text} !important;
}}
"""
    inject_parent_css(css, style_id="sf-theme-force-style")
