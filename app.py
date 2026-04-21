"""
StyleMind AI — Streamlit Interface
Modern, minimal white UI with animated pipeline progress and outfit cards.
"""

from __future__ import annotations

import base64
import html
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="StyleMind AI",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;1,14..32,400&display=swap');

/*
  Palette
  ─────────────────────────────────────────
  Background  #F8F5F0  warm linen
  Surface     #FFFEFB  pure warm white
  Sidebar     #FFFFFF
  Accent      #8B6F5E  warm clay / terracotta
  Accent-2    #C4956A  golden caramel highlight
  Border      #E6E1D9  warm stone
  Text-1      #1C1917  near-black
  Text-2      #57534E  medium warm gray
  Text-3      #A8A29E  muted warm gray
  Success     #4A7C59  muted sage green
  ─────────────────────────────────────────
*/

/* ── Reset Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1C1917 !important;
}
.stApp {
    background: #F8F5F0 !important;
}

/* ── Force Streamlit's own "app view" containers to match ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #F8F5F0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF !important;
    border-right: 1px solid #E6E1D9 !important;
}
[data-testid="stSidebar"] > div { padding-top: 28px; }

/* ── Main container ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 920px !important;
    background: transparent !important;
}

/* ════════════════════════════════════
   FORCE ALL INPUTS TO LIGHT THEME
   ════════════════════════════════════ */

/* Textarea wrapper + the actual element */
.stTextArea,
.stTextArea > label,
.stTextArea > div,
div[data-testid="stTextArea"],
div[data-testid="stTextArea"] > div {
    background: transparent !important;
}
.stTextArea textarea,
div[data-baseweb="textarea"] textarea,
textarea {
    background-color: #FFFEFB !important;
    color: #1C1917 !important;
    border: 1.5px solid #E6E1D9 !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    caret-color: #8B6F5E !important;
    -webkit-text-fill-color: #1C1917 !important;
    box-shadow: none !important;
    resize: none !important;
}
.stTextArea textarea:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-color: #8B6F5E !important;
    box-shadow: 0 0 0 3px rgba(139,111,94,0.12) !important;
    outline: none !important;
}
/* Base-web textarea container */
div[data-baseweb="textarea"] {
    background-color: #FFFEFB !important;
    border-radius: 14px !important;
    border: none !important;
}

/* Text input */
.stTextInput input,
div[data-baseweb="input"] input,
input[type="text"],
input[type="search"] {
    background-color: #FFFEFB !important;
    color: #1C1917 !important;
    border: 1.5px solid #E6E1D9 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    caret-color: #8B6F5E !important;
    -webkit-text-fill-color: #1C1917 !important;
    box-shadow: none !important;
}
.stTextInput input:focus,
div[data-baseweb="input"] input:focus {
    border-color: #8B6F5E !important;
    box-shadow: 0 0 0 3px rgba(139,111,94,0.12) !important;
    outline: none !important;
}
div[data-baseweb="input"] {
    background-color: #FFFEFB !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Selectbox */
div[data-baseweb="select"] > div,
.stSelectbox > div > div {
    background-color: #FFFEFB !important;
    border: 1.5px solid #E6E1D9 !important;
    border-radius: 10px !important;
    color: #1C1917 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
div[data-baseweb="select"] span {
    color: #1C1917 !important;
}
/* Selectbox dropdown */
ul[role="listbox"] li,
div[data-baseweb="popover"] {
    background-color: #FFFEFB !important;
    color: #1C1917 !important;
}

/* Streamlit label text */
.stTextArea label,
.stTextInput label,
.stSelectbox label {
    color: #57534E !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ── Header ── */
.sm-header {
    padding: 6px 0 26px;
    border-bottom: 1px solid #E6E1D9;
    margin-bottom: 30px;
    display: flex;
    align-items: flex-end;
    gap: 16px;
}
.sm-wordmark {
    font-size: 21px;
    font-weight: 600;
    letter-spacing: -0.4px;
    color: #1C1917;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sm-wordmark::before {
    content: "";
    display: inline-block;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2.5px solid #8B6F5E;
    background: transparent;
    flex-shrink: 0;
}
.sm-tagline {
    font-size: 13px;
    color: #8B6F5E;
    margin-bottom: 2px;
    font-weight: 400;
}

/* ── Input section ── */
p.input-label {
    font-size: 13px;
    font-weight: 500;
    color: #57534E;
    margin: 0 0 4px 0 !important;
    letter-spacing: 0.1px;
}

/* ── Photo strip ── */
.photo-strip {
    display: flex;
    gap: 10px;
    margin-bottom: 22px;
    overflow-x: auto;
    padding-bottom: 4px;
    scrollbar-width: none;
}
.photo-strip::-webkit-scrollbar { display: none; }
.photo-frame { flex-shrink: 0; width: 84px; text-align: center; }
.photo-frame img {
    width: 84px;
    height: 108px;
    object-fit: cover;
    border-radius: 10px;
    display: block;
    border: 1px solid #EBE6DF;
}
.photo-placeholder {
    width: 84px;
    height: 108px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    border: 1px solid #EBE6DF;
}
.photo-slot-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: #C4B9B3;
    margin-top: 5px;
}

/* ── Pipeline steps ── */
.pipeline-wrap {
    background: #FFFEFB;
    border-radius: 16px;
    border: 1px solid #E6E1D9;
    padding: 20px 24px;
    margin-bottom: 26px;
    animation: fadeInUp 0.4s ease forwards;
    box-shadow: 0 2px 14px rgba(139,111,94,0.06);
}
.pipeline-title {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #A8A29E;
    margin-bottom: 14px;
}
.p-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    font-size: 13.5px;
    color: #C4B9B3;
    transition: color 0.25s;
}
.p-step.done  { color: #4A7C59; }
.p-step.active{ color: #1C1917; font-weight: 500; }
.p-step .icon { width: 20px; text-align: center; font-size: 13px; }
.p-step .detail {
    font-size: 12px;
    color: #A8A29E;
    margin-left: 4px;
    font-weight: 400;
}

/* ── Outfit cards ── */
.outfit-card {
    background: #FFFEFB;
    border-radius: 22px;
    padding: 30px 34px;
    border: 1px solid #EBE6DF;
    box-shadow: 0 4px 28px rgba(139,111,94,0.08);
    margin-bottom: 22px;
    opacity: 0;
    animation: fadeInUp 0.5s ease forwards;
}
.card-1 { animation-delay: 0.05s; }
.card-2 { animation-delay: 0.20s; }
.card-3 { animation-delay: 0.35s; }

.outfit-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 5px;
}
.outfit-name {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: #1C1917;
    line-height: 1.2;
}
.outfit-price {
    font-size: 18px;
    font-weight: 600;
    color: #8B6F5E;
    white-space: nowrap;
    margin-left: 18px;
}
.outfit-concept {
    font-size: 13.5px;
    color: #6B6360;
    line-height: 1.65;
    margin-bottom: 18px;
}

/* ── Palette ── */
.palette-row { display: flex; align-items: center; gap: 7px; margin-bottom: 18px; }
.pdot {
    width: 17px; height: 17px;
    border-radius: 50%;
    border: 1.5px solid rgba(0,0,0,0.09);
    display: inline-block;
    flex-shrink: 0;
}
.palette-label { font-size: 12px; color: #A8A29E; margin-left: 5px; }

/* ── Items table ── */
.items-table { margin-bottom: 18px; }
.item-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #F2EDE7;
    font-size: 13.5px;
}
.item-row:last-child { border-bottom: none; }
.item-slot {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #C4956A;
    width: 72px;
    flex-shrink: 0;
}
.item-name { flex: 1; color: #1C1917; }
.item-color { font-size: 12px; color: #A8A29E; margin-right: 6px; }
.item-price { font-weight: 600; color: #8B6F5E; font-size: 13px; white-space: nowrap; }

/* ── Lookbook prose ── */
.lookbook-prose {
    font-size: 13.5px;
    line-height: 1.9;
    color: #4A4541;
    font-style: italic;
    padding: 18px 22px;
    background: #F8F3EE;
    border-radius: 14px;
    border-left: 3px solid #8B6F5E;
    margin: 18px 0;
}

/* ── Social signal ── */
.social-signal {
    font-size: 13px;
    color: #57534E;
    padding: 10px 14px;
    background: #F3EFE9;
    border-radius: 10px;
    margin-bottom: 18px;
    line-height: 1.55;
}

/* ── Score bars ── */
.scores-row { display: flex; gap: 22px; margin-bottom: 8px; }
.score-item { flex: 1; }
.score-label { font-size: 11px; color: #A8A29E; margin-bottom: 6px; letter-spacing: 0.2px; }
.score-bar-bg {
    height: 3px; border-radius: 2px; background: #EDE8E2;
}
.score-bar-fill { height: 100%; border-radius: 2px; background: #8B6F5E; }

/* ── Profile card (sidebar) ── */
.profile-card {
    background: #F8F5F0;
    border-radius: 13px;
    padding: 14px 16px;
    margin: 10px 0;
    font-size: 13px;
    border: 1px solid #E6E1D9;
}
.profile-stat { display: flex; justify-content: space-between; padding: 4px 0; color: #57534E; }
.profile-stat .val { font-weight: 500; color: #1C1917; }

/* ── Confidence bar ── */
.conf-bar-bg { height: 3px; border-radius: 2px; background: #E6E1D9; margin: 8px 0 4px; }
.conf-bar-fill { height: 100%; border-radius: 2px; background: #4A7C59; }

/* ── Divider ── */
.sm-divider { border: none; border-top: 1px solid #E6E1D9; margin: 22px 0; }

/* ── Section label ── */
.section-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #A8A29E;
    margin: 26px 0 12px;
}

/* ── Feedback banner ── */
.feedback-banner {
    background: #F0F9F4;
    border: 1px solid #B6DEC5;
    border-radius: 12px;
    padding: 13px 17px;
    font-size: 13.5px;
    color: #4A7C59;
    margin-bottom: 18px;
    animation: fadeInUp 0.35s ease forwards;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 64px 20px;
    color: #A8A29E;
    font-size: 14px;
}
.empty-icon { font-size: 38px; margin-bottom: 12px; }

/* ── Streamlit button overrides ── */
.stButton > button {
    background-color: #FFFEFB !important;
    color: #1C1917 !important;
    border: 1.5px solid #E6E1D9 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #8B6F5E !important;
    color: #8B6F5E !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(139,111,94,0.14) !important;
}
/* Primary button (Style me) */
.stButton > button[kind="primary"] {
    background-color: #8B6F5E !important;
    color: #FFFEFB !important;
    border-color: #8B6F5E !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #7A6150 !important;
    border-color: #7A6150 !important;
    color: #FFFEFB !important;
}

/* ── Spinner / progress ── */
.stSpinner > div { border-top-color: #8B6F5E !important; }

/* ── Sidebar: force all text dark ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] div {
    color: #1C1917 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #1C1917 !important;
    font-size: 13px !important;
}

/* Sidebar expanders */
[data-testid="stSidebar"] .stExpander {
    background: #F8F5F0 !important;
    border: 1px solid #E6E1D9 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stExpander summary,
[data-testid="stSidebar"] .stExpander summary p,
[data-testid="stSidebar"] .stExpander summary span {
    color: #57534E !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] {
    background: #F8F5F0 !important;
}
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] p,
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] span {
    color: #57534E !important;
    font-size: 13px !important;
}

/* Sidebar horizontal rule */
[data-testid="stSidebar"] hr {
    border-color: #E6E1D9 !important;
    border-top-width: 1px !important;
}

/* Sidebar section label (bold markdown) */
[data-testid="stSidebar"] strong {
    color: #1C1917 !important;
    font-weight: 600 !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: #FFFEFB !important;
    color: #57534E !important;
    border: 1px solid #E6E1D9 !important;
    font-size: 12.5px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #8B6F5E !important;
    color: #8B6F5E !important;
}

/* ── Alert / notification boxes ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 13.5px !important;
}
/* Warning */
div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
.stAlert[data-type="warning"],
div[class*="stWarning"] {
    background: #FDF4EC !important;
    border-color: #DDB98A !important;
    color: #7A5C2E !important;
}
/* Error */
div[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
.stAlert[data-type="error"],
div[class*="stError"] {
    background: #FDF0EF !important;
    border-color: #E8A09A !important;
    color: #7A2E2E !important;
}
/* Info */
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
.stAlert[data-type="info"],
div[class*="stInfo"] {
    background: #EEF3FD !important;
    border-color: #9BBCE8 !important;
    color: #2E4A7A !important;
}
/* Success */
div[class*="stSuccess"] {
    background: #EDF6F1 !important;
    border-color: #7DC4A0 !important;
    color: #2E6B4A !important;
}
/* Force all alert text dark */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: inherit !important;
}

/* ── Radio buttons ── */
.stRadio label { color: #1C1917 !important; font-size: 13px !important; }
.stRadio > div > label { color: #57534E !important; font-size: 13px !important; }

/* ── Global paragraph text ── */
.stMarkdown p { color: #1C1917 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_retriever():
    from agents.retriever import ProductRetriever
    from config import FAISS_INDEX_PATH
    if not FAISS_INDEX_PATH.exists():
        return None
    return ProductRetriever()


@st.cache_resource(show_spinner=False)
def load_pipeline(_retriever):
    from pipeline import StyleMindPipeline
    return StyleMindPipeline(retriever=_retriever)


# ── Image helpers ──────────────────────────────────────────────────────────

_IMAGES_DIR = Path(__file__).parent / "data" / "images"

@st.cache_data(show_spinner=False)
def _load_img_b64(product_id: str) -> str | None:
    path = _IMAGES_DIR / f"{product_id}.jpg"
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def photo_frame_html(item) -> str:
    """Return a <div class='photo-frame'> for one item — real img or placeholder."""
    b64 = _load_img_b64(str(item.product_id)) if item.product_id else None
    if b64:
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{_e(item.product_name)}">'
    else:
        bg = color_to_css(item.color)
        img_tag = f'<div class="photo-placeholder" style="background:{bg}">◎</div>'
    slot_label = _e(item.slot.upper())
    return f'<div class="photo-frame">{img_tag}<div class="photo-slot-label">{slot_label}</div></div>'


# ── Color palette helper ───────────────────────────────────────────────────

_COLOR_CSS = {
    "black": "#1A1A1A", "white": "#F5F5F5", "navy": "#1F3A5E",
    "charcoal": "#3D4550", "grey": "#9E9E9E", "gray": "#9E9E9E",
    "beige": "#D4C5A9", "brown": "#7B5230", "cream": "#F5EDD6",
    "khaki": "#C3B091", "olive": "#708238", "burgundy": "#6E2233",
    "red": "#C0392B", "blue": "#2471A3", "green": "#1E8449",
    "yellow": "#D4AC0D", "orange": "#CA6F1E", "pink": "#D98880",
    "purple": "#7D3C98", "maroon": "#76252A", "tan": "#C9A96E",
    "camel": "#C19A6B", "ivory": "#FFFFF0", "off-white": "#F5F0E8",
    "stone": "#B2A99B", "teal": "#1ABC9C", "rust": "#A04000",
    "indigo": "#2C3E8C", "mint": "#ABEBC6", "coral": "#F0826D",
}

def color_to_css(name: str) -> str:
    lower = name.lower().strip()
    for key, val in _COLOR_CSS.items():
        if key in lower:
            return val
    return "#CCCCCA"


def palette_html(colors: list[str]) -> str:
    dots = "".join(
        f'<span class="pdot" style="background:{color_to_css(c)}" title="{c}"></span>'
        for c in colors
    )
    labels = " · ".join(colors)
    return f'<div class="palette-row">{dots}<span class="palette-label">{labels}</span></div>'


# ── Outfit card renderer ───────────────────────────────────────────────────

def _e(text: str) -> str:
    """HTML-escape LLM-generated text to prevent tag injection."""
    return html.escape(str(text))


def render_outfit_card(outfit, lookbook, card_idx: int) -> str:
    # Photo strip — no newlines so Markdown parser can't break HTML context
    photos_html = "".join(photo_frame_html(item) for item in outfit.items)

    # Items table with inline thumbnails
    item_rows = []
    for item in outfit.items:
        b64 = _load_img_b64(str(item.product_id)) if item.product_id else None
        if b64:
            thumb = (f'<img src="data:image/jpeg;base64,{b64}" '
                     f'style="width:38px;height:48px;object-fit:cover;'
                     f'border-radius:6px;flex-shrink:0;border:1px solid #EBE6DF">')
        else:
            bg = color_to_css(item.color)
            thumb = (f'<div style="width:38px;height:48px;border-radius:6px;'
                     f'background:{bg};flex-shrink:0;border:1px solid #EBE6DF"></div>')
        item_rows.append(
            f'<div class="item-row">{thumb}'
            f'<span class="item-slot">{_e(item.slot)}</span>'
            f'<span class="item-name">{_e(item.product_name)}</span>'
            f'<span class="item-color">{_e(item.color)}</span>'
            f'<span class="item-price">${item.price:.0f}</span>'
            f'</div>'
        )

    prose_html = ""
    if lookbook and lookbook.prose:
        # Replace newlines with <br> so they don't fragment the Markdown parser
        safe_prose = _e(lookbook.prose).replace("\n", "<br>")
        prose_html = f'<div class="lookbook-prose">{safe_prose}</div>'

    signal_html = f'<div class="social-signal">&#8220;{_e(outfit.social_signal)}&#8221;</div>'

    bscore = int(outfit.budget_score * 100)
    cscore = int(outfit.color_harmony_score * 100)
    oscore = int(outfit.overall_score * 100)

    scores_html = (
        f'<div class="scores-row">'
        f'<div class="score-item"><div class="score-label">Color harmony</div>'
        f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{cscore}%"></div></div></div>'
        f'<div class="score-item"><div class="score-label">Budget fit</div>'
        f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{bscore}%"></div></div></div>'
        f'<div class="score-item"><div class="score-label">Overall</div>'
        f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{oscore}%"></div></div></div>'
        f'</div>'
    )

    pal = palette_html(outfit.color_palette)

    # Build as one contiguous string — no double-newlines that would break
    # Streamlit's hybrid Markdown+HTML renderer out of block-HTML context.
    return (
        f'<div class="outfit-card card-{card_idx + 1}">'
        f'<div class="outfit-header">'
        f'<div class="outfit-name">{_e(outfit.blueprint_name)}</div>'
        f'<div class="outfit-price">${outfit.total_price:.0f}</div>'
        f'</div>'
        f'<div class="outfit-concept">{_e(outfit.blueprint_concept)}</div>'
        f'<div class="photo-strip">{photos_html}</div>'
        f'{pal}'
        f'<div class="items-table">{"".join(item_rows)}</div>'
        f'{prose_html}'
        f'{signal_html}'
        f'{scores_html}'
        f'</div>'
    )


# ── Session state init ─────────────────────────────────────────────────────

def init_state():
    defaults = {
        "profile": None,
        "user_id": "",
        "result": None,
        "query": "",
        "feedback_done": False,
        "pipeline_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar: Profile management ────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0 4px 20px">
        <div style="font-size:18px;font-weight:600;letter-spacing:-0.3px;color:#1C1917">StyleMind</div>
        <div style="font-size:12px;color:#8B6F5E;margin-top:2px">Personal AI Stylist</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Your Profile**")

    uid_input = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        placeholder="e.g. alex_chen",
        label_visibility="collapsed",
    )

    if uid_input != st.session_state.user_id:
        st.session_state.user_id = uid_input
        st.session_state.profile = None
        st.session_state.result = None
        st.session_state.feedback_done = False

    if uid_input:
        if st.session_state.profile is None:
            from agents.taste_profile import TasteProfile
            st.session_state.profile = TasteProfile.load(uid_input)

        profile = st.session_state.profile

        conf_pct = int(profile.profile_confidence * 100)
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-stat"><span>Sessions</span><span class="val">{profile.total_sessions}</span></div>
            <div class="profile-stat"><span>Items rated</span><span class="val">{profile.total_items_rated}</span></div>
            <div class="profile-stat"><span>Confidence</span><span class="val">{conf_pct}%</span></div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gender selection (shown when unset, persists after first run) ──
        gender_display_map = {
            "masculine": "Men's",
            "feminine": "Women's",
            "androgynous": "Unisex / Both",
        }
        gender_options = ["Not specified", "Men's", "Women's", "Unisex / Both"]
        current_display = gender_display_map.get(profile.gender_expression, "Not specified")
        current_idx = gender_options.index(current_display)

        st.markdown("<div style='font-size:12px;font-weight:500;color:#57534E;margin:10px 0 4px'>I shop for</div>", unsafe_allow_html=True)
        gender_choice = st.selectbox(
            "Gender",
            options=gender_options,
            index=current_idx,
            label_visibility="collapsed",
            key=f"gender_select_{uid_input}",
        )
        gender_map = {
            "Men's": "masculine",
            "Women's": "feminine",
            "Unisex / Both": "androgynous",
            "Not specified": "unspecified",
        }
        chosen = gender_map.get(gender_choice, "unspecified")
        if chosen != profile.gender_expression:
            profile.gender_expression = chosen
            profile.save()
            st.session_state.profile = profile
            st.rerun()

        if profile.total_sessions > 0:
            with st.expander("Style identity", expanded=False):
                st.markdown(f"<div style='font-size:13px;color:#4A4541;line-height:1.6'>{_e(profile.style_identity)}</div>", unsafe_allow_html=True)
                if profile.style_archetypes:
                    st.markdown(f"<div style='font-size:12px;color:#8B6F5E;margin-top:6px'>{' · '.join(_e(a) for a in profile.style_archetypes)}</div>", unsafe_allow_html=True)

            if profile.color_preferences.preferred:
                with st.expander("Colors & fits", expanded=False):
                    st.markdown(f"**Preferred:** {', '.join(profile.color_preferences.preferred)}")
                    if profile.color_preferences.avoided:
                        st.markdown(f"**Avoided:** {', '.join(profile.color_preferences.avoided)}")
                    if profile.fit_preferences.preferred_fits:
                        st.markdown(f"**Fits:** {', '.join(profile.fit_preferences.preferred_fits)}")

            if profile.rejections:
                with st.expander(f"Rejection log ({len(profile.rejections)})", expanded=False):
                    for r in sorted(profile.rejections, key=lambda x: x.count, reverse=True)[:6]:
                        st.markdown(f"<div style='font-size:12.5px;color:#57534E;padding:3px 0'>{_e(r.item_type)} <span style='color:#A8A29E'>×{r.count}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13px;color:#A8A29E;padding:8px 0'>New profile — preferences build as you use StyleMind.</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Reset profile", use_container_width=True):
            from agents.taste_profile import TasteProfile
            st.session_state.profile = TasteProfile(user_id=uid_input)
            st.session_state.profile.save()
            st.session_state.result = None
            st.session_state.feedback_done = False
            st.rerun()
    else:
        st.markdown("<div style='font-size:13px;color:#A8A29E;padding:8px 0'>Enter a user ID above to load or create a profile.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#A8A29E;line-height:1.6'>StyleMind AI · CMU 11-766<br>Tanay Mittal</div>", unsafe_allow_html=True)


# ── Main content ───────────────────────────────────────────────────────────

st.markdown("""
<div class="sm-header">
    <div class="sm-wordmark">StyleMind AI</div>
    <div class="sm-tagline">Personal AI Stylist</div>
</div>
""", unsafe_allow_html=True)

# ── Query input card ───────────────────────────────────────────────────────

st.markdown('<p class="input-label">What are you dressing for?</p>', unsafe_allow_html=True)

query = st.text_area(
    "query",
    value=st.session_state.query,
    placeholder='e.g. "Smart casual dinner in Pittsburgh, October evening, budget around $180"',
    height=90,
    label_visibility="collapsed",
    key="query_input",
)

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    run_btn = st.button("Style me", type="primary", use_container_width=False)
with col2:
    num_outfits = st.selectbox("Outfits", [1, 2, 3], index=1, label_visibility="collapsed")
with col3:
    st.markdown("<div style='font-size:12px;color:#A8A29E;padding-top:8px'>outfits</div>", unsafe_allow_html=True)


# ── Run pipeline ───────────────────────────────────────────────────────────

if run_btn and query.strip():
    if not st.session_state.user_id:
        st.markdown("""
        <div style="background:#FDF4EC;border:1.5px solid #DDB98A;border-radius:14px;
                    padding:16px 20px;font-size:14px;color:#7A5C2E;line-height:1.7;margin:8px 0 16px">
            <strong style="display:block;margin-bottom:4px">Sign in to continue</strong>
            Enter a <strong>User ID</strong> in the sidebar on the left to get personalised
            style recommendations and save your preferences.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.query = query.strip()
        st.session_state.result = None
        st.session_state.feedback_done = False
        st.session_state.pipeline_log = []

        # Pipeline progress display
        stages = [
            ("parse",    "Parsing your request",        "🔍"),
            ("plan",     "Planning outfit blueprints",   "🎨"),
            ("retrieve", "Finding matching products",    "🔎"),
            ("assemble", "Assembling & scoring outfits", "⚡"),
            ("lookbook", "Writing your lookbook",        "✍️"),
        ]
        done_stages = set()
        # Use a dict so the nested closure can mutate it without nonlocal
        _state = {"active": None, "detail": ""}

        def make_pipeline_html():
            rows = ""
            for key, label, icon in stages:
                if key in done_stages:
                    cls, indicator = "done", "✓"
                    rows += f'<div class="p-step {cls}"><span class="icon">{indicator}</span>{label}</div>'
                elif key == _state["active"]:
                    cls, indicator = "active", "⟳"
                    detail_span = (
                        f'<span class="detail">— {_e(_state["detail"])}</span>'
                        if _state["detail"] else ""
                    )
                    rows += f'<div class="p-step {cls}"><span class="icon">{indicator}</span>{label}{detail_span}</div>'
                else:
                    rows += f'<div class="p-step"><span class="icon">○</span>{label}</div>'
            return f'<div class="pipeline-wrap"><div class="pipeline-title">Generating your look</div>{rows}</div>'

        pipeline_placeholder = st.empty()
        pipeline_placeholder.markdown(make_pipeline_html(), unsafe_allow_html=True)

        def progress_cb(stage: str, detail: str):
            if _state["active"] is not None:
                done_stages.add(_state["active"])
            _state["active"] = stage
            _state["detail"] = detail
            pipeline_placeholder.markdown(make_pipeline_html(), unsafe_allow_html=True)

        try:
            retriever = load_retriever()
            pipe = load_pipeline(retriever)
            pipe._progress = progress_cb

            result = pipe.run(
                query=st.session_state.query,
                profile=st.session_state.profile,
                top_k_per_slot=8,
                num_outfits=num_outfits,
            )

            # Mark all done
            done_stages.update(s[0] for s in stages)
            _state["active"] = None
            pipeline_placeholder.markdown(make_pipeline_html(), unsafe_allow_html=True)

            st.session_state.result = result

            # Save updated pipeline (closure kept progress_cb)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

elif run_btn and not query.strip():
    st.warning("Please describe what you're dressing for.")


# ── Results ────────────────────────────────────────────────────────────────

result = st.session_state.get("result")

if result is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◎</div>
        <div>Describe your occasion above and hit <strong>Style me</strong></div>
        <div style="font-size:12px;margin-top:8px;color:#C4B9B3">Weekend brunch · Job interview · Date night · Festival · Anything</div>
    </div>
    """, unsafe_allow_html=True)

else:
    outfits = result.outfits
    lookbooks = result.lookbooks
    intent = result.intent

    # Timing note
    total_s = result.timing.get("total", 0)
    st.markdown(f"""
    <div style="font-size:12px;color:#A8A29E;margin-bottom:20px">
        {len(outfits)} outfit{"s" if len(outfits) != 1 else ""} generated in {total_s:.1f}s ·
        {_e(intent.formality_level)} · {_e(intent.occasion)}
        {"· $" + str(int(intent.budget_max)) + " budget" if intent.budget_max else ""}
    </div>
    """, unsafe_allow_html=True)

    if intent.profile_enhancements:
        st.markdown(f"""
        <div style="background:#F5F0EA;border:1px solid #DDD4C8;border-radius:12px;padding:12px 16px;
                    margin-bottom:20px;font-size:13px;color:#57534E;line-height:1.6">
            <strong style="color:#8B6F5E">Profile personalisation applied</strong><br>
            {" · ".join(_e(e) for e in intent.profile_enhancements[:2])}
        </div>
        """, unsafe_allow_html=True)

    if intent.profile_conflicts:
        st.markdown(f"""
        <div style="background:#FDF8F0;border:1px solid #E8D5B0;border-radius:12px;padding:12px 16px;
                    margin-bottom:20px;font-size:13px;color:#7A5C2E;line-height:1.6">
            ⚠️ {" · ".join(_e(c) for c in intent.profile_conflicts)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Your Style Guide</div>', unsafe_allow_html=True)

    for i, (outfit, lb) in enumerate(zip(outfits, lookbooks)):
        st.markdown(render_outfit_card(outfit, lb, i), unsafe_allow_html=True)

    # ── Feedback section ──────────────────────────────────────────────────

    st.markdown('<hr class="sm-divider">', unsafe_allow_html=True)

    if st.session_state.feedback_done:
        st.markdown("""
        <div class="feedback-banner">
            ✓ Profile updated — your preferences have been saved and will shape future recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">Feedback</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;color:#57534E;margin-bottom:16px'>Accept an outfit to save it to your profile.</div>", unsafe_allow_html=True)

        outfit_names = [o.blueprint_name for o in outfits]
        feedback_cols = st.columns(len(outfit_names) + 1)

        accepted_outfit = None
        for i, (col, name) in enumerate(zip(feedback_cols, outfit_names)):
            with col:
                if st.button(f"✓  {name}", key=f"accept_{i}", use_container_width=True):
                    accepted_outfit = (i, name)

        with feedback_cols[-1]:
            skip = st.button("Skip →", key="skip_feedback", use_container_width=True)

        if accepted_outfit is not None or skip:
            profile = st.session_state.profile
            if profile is not None and st.session_state.user_id:
                from agents.profile_updater import ProfileUpdater, SessionFeedback, ItemFeedback

                accepted = accepted_outfit is not None
                outfit_name = outfit_names[accepted_outfit[0]] if accepted else None

                feedback = SessionFeedback(
                    outfit_name=outfit_name,
                    outfit_accepted=accepted,
                    item_feedback=[
                        ItemFeedback(
                            item_name=item.product_name,
                            item_type=item.article_type,
                            action="accepted",
                        )
                        for item in outfits[accepted_outfit[0]].items
                    ] if accepted else [],
                )

                with st.spinner("Updating your profile…"):
                    updater = ProfileUpdater()
                    updated = updater.update(
                        profile=profile,
                        feedback=feedback,
                        request_summary=st.session_state.query,
                    )
                    st.session_state.profile = updated

                st.session_state.feedback_done = True
                st.rerun()
