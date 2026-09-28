"""Shared aggressive Streamlit light/dark widget CSS (labels, inputs, uploaders)."""

from __future__ import annotations


def streamlit_widget_theme_css(theme: str, *, prefix: str = "app") -> str:
    """Return CSS that wins against Streamlit Emotion defaults for form widgets."""
    is_dark = theme == "dark"
    text = "#f1f5f9" if is_dark else "#0f172a"
    muted = "#94a3b8" if is_dark else "#475569"
    surface = "#1e293b" if is_dark else "#ffffff"
    surface2 = "#0f172a" if is_dark else "#f8fafc"
    border = "rgba(148,163,184,0.45)" if is_dark else "rgba(15,23,42,0.14)"
    btn_bg = "#2dd4bf" if prefix == "telco" else "#d4af37"
    btn_fg = "#042f2e" if prefix == "telco" else "#0c0b0a"
    scheme = "dark" if is_dark else "light"

    # Concatenate (not one huge f-string) so braces in CSS never break Python.
    return (
        f"""
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    section[data-testid="stSidebar"], .main, .block-container {{
      color-scheme: {scheme} !important;
      color: {text} !important;
    }}

    /* Force readable body / widget copy (Emotion often leaves light-mode ink on dark bg) */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] *,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] *,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp label[data-testid="stWidgetLabel"],
    .stApp .stSelectbox label, .stApp .stSelectbox label *,
    .stApp .stMultiSelect label, .stApp .stMultiSelect label *,
    .stApp .stTextInput label, .stApp .stTextInput label *,
    .stApp .stNumberInput label, .stApp .stNumberInput label *,
    .stApp .stTextArea label, .stApp .stTextArea label *,
    .stApp .stSlider label, .stApp .stSlider label *,
    .stApp .stRadio label, .stApp .stRadio label *,
    .stApp .stCheckbox label, .stApp .stCheckbox label *,
    .stApp .stDateInput label, .stApp .stDateInput label *,
    .stApp .stFileUploader label, .stApp .stFileUploader label *,
    .stApp [class*="st-emotion-cache"] label,
    .stApp [class*="st-emotion-cache"] p {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
      opacity: 1 !important;
      visibility: visible !important;
    }}

    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] *,
    .stApp .nav-hint,
    .stApp small {{
      color: {muted} !important;
      -webkit-text-fill-color: {muted} !important;
    }}

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
      opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] .nav-hint,
    section[data-testid="stSidebar"] small {{
      color: {muted} !important;
      -webkit-text-fill-color: {muted} !important;
    }}

    /* Inputs / selects / BaseWeb */
    .stApp .stTextInput input,
    .stApp .stNumberInput input,
    .stApp .stTextArea textarea,
    .stApp div[data-baseweb="select"],
    .stApp div[data-baseweb="select"] > div,
    .stApp div[data-baseweb="select"] > div > div,
    .stApp div[data-baseweb="base-input"],
    .stApp div[data-baseweb="input"],
    .stApp div[data-baseweb="input"] > div,
    .stApp [data-baseweb="input"] input,
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
    .stApp div[data-baseweb="select"] span,
    .stApp div[data-baseweb="select"] div,
    .stApp div[data-baseweb="select"] svg {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
      fill: {text} !important;
    }}
    .stApp [data-baseweb="popover"],
    .stApp [data-baseweb="menu"],
    .stApp [data-baseweb="popover"] ul,
    .stApp [role="listbox"],
    .stApp [role="option"],
    body > div[data-baseweb="popover"],
    body > div[data-baseweb="popover"] * {{
      background-color: {surface2} !important;
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
    }}

    /* Slider value */
    .stApp .stSlider [data-testid="stTickBarMin"],
    .stApp .stSlider [data-testid="stTickBarMax"],
    .stApp [data-testid="stThumbValue"],
    .stApp [data-testid="stTickBar"] * {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
    }}

    /* File uploader — Browse was white-on-white in dark mode */
    .stApp [data-testid="stFileUploader"],
    .stApp [data-testid="stFileUploader"] section,
    .stApp [data-testid="stFileUploaderDropzone"],
    .stApp [data-testid="stFileUploaderDropzone"] > div {{
      background-color: {surface} !important;
      border: 1px solid {border} !important;
      color: {text} !important;
    }}
    .stApp [data-testid="stFileUploader"] *,
    .stApp [data-testid="stFileUploaderDropzone"] *,
    .stApp [data-testid="stFileUploader"] small,
    .stApp [data-testid="stFileUploader"] span,
    .stApp [data-testid="stFileUploader"] p {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
      opacity: 1 !important;
    }}
    .stApp [data-testid="stFileUploader"] button,
    .stApp [data-testid="stFileUploader"] button *,
    .stApp [data-testid="stFileUploaderDropzone"] button,
    .stApp [data-testid="stFileUploaderDropzone"] button *,
    .stApp [data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
    .stApp [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] * {{
      background-color: {btn_bg} !important;
      background-image: none !important;
      color: {btn_fg} !important;
      -webkit-text-fill-color: {btn_fg} !important;
      border: none !important;
      font-weight: 700 !important;
      opacity: 1 !important;
      visibility: visible !important;
    }}

    .stApp .stButton > button {{
      color: {text} !important;
    }}
    .stApp .stButton > button[kind="primary"],
    .stApp .stButton > button[kind="primary"] * {{
      background: {btn_bg} !important;
      color: {btn_fg} !important;
      -webkit-text-fill-color: {btn_fg} !important;
    }}

    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricDelta"] {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
    }}

    .stApp .stAlert, .stApp .stAlert * {{
      color: {text} !important;
      -webkit-text-fill-color: {text} !important;
    }}

    .stApp [data-testid="stVerticalBlockBorderWrapper"],
    .stApp div[data-testid="stMetric"],
    .stApp [data-testid="stExpander"],
    .stApp [data-testid="stExpanderDetails"],
    .stApp .stDataFrame,
    .stApp [data-testid="stDataFrameResizable"] {{
      background-color: {surface} !important;
      color: {text} !important;
      border-color: {border} !important;
    }}

    .stApp * {{
      scrollbar-width: thin !important;
      scrollbar-color: {muted} {surface2} !important;
    }}
    .stApp *::-webkit-scrollbar {{ width: 12px !important; height: 12px !important; }}
    .stApp *::-webkit-scrollbar-track {{ background: {surface2} !important; }}
    .stApp *::-webkit-scrollbar-thumb {{
      background: {muted} !important;
      border-radius: 8px !important;
      border: 2px solid {surface2} !important;
    }}

    .goog-te-banner-frame, .skiptranslate iframe.goog-te-banner-frame {{ display: none !important; }}
    body {{ top: 0 !important; }}
    .goog-logo-link, .goog-te-gadget span {{ display: none !important; }}
    .goog-te-gadget {{ font-size: 0 !important; color: {text} !important; }}
    #google_translate_element select {{
      font-size: 0.85rem !important;
      min-height: 36px !important;
      color: {text} !important;
      background: {surface} !important;
      border: 1px solid {border} !important;
    }}
    """
    )


def inject_widget_theme(theme: str, *, prefix: str = "app") -> None:
    """Inject widget CSS into the parent Streamlit document."""
    from streamlit_parent_inject import inject_parent_css

    css = streamlit_widget_theme_css(theme, prefix=prefix)
    inject_parent_css(css, style_id=f"sf-{prefix}-widgets")
