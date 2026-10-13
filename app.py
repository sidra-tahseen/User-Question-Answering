import html
from pathlib import Path

import streamlit as st
import torch
from transformers import pipeline

st.set_page_config(
    page_title="Context QA | Extractive Question Answering",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_ID = "sidratahseen/User-Questioning-Answer"
MAROON = "#6F1D36"
MAROON_DARK = "#481324"
MAROON_LIGHT = "#A13D5B"
PAPER = "#F5F1ED"
SURFACE = "#FFFCF9"
INK = "#211B1D"
MUTED = "#6D6265"
RULE = "#D8CCCE"

# Keep the CSS deliberately small and target stable Streamlit selectors.
st.markdown(
    f"""
    <style>
    .stApp {{ background: {PAPER}; color: {INK}; }}
    header[data-testid="stHeader"] {{ background: {PAPER}; }}
    [data-testid="stAppViewContainer"] > .main {{ background: {PAPER}; }}
    [data-testid="stMainBlockContainer"] {{
        max-width: 1180px;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }}
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {{ color: {INK}; }}
    h1 {{ font-size: clamp(2.1rem, 4vw, 3.2rem) !important; line-height: 1.08 !important; }}
    .qa-hero {{
        background: {MAROON_DARK}; color: #FFFFFF;
        border-left: 7px solid {MAROON_LIGHT};
        padding: 2rem 2.2rem; margin: 0.3rem 0 1.8rem 0;
    }}
    .qa-hero .eyebrow {{
        color: #E8B9C7; font-size: .74rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: .8rem;
    }}
    .qa-hero .hero-title {{
        display: block; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 3.25rem); font-weight: 750;
        line-height: 1.08; letter-spacing: -.035em; margin: 0 0 .85rem 0;
    }}
    .qa-hero .hero-copy {{ color: #F0E3E7; max-width: 780px; line-height: 1.65; margin: 0; }}
    .section-label {{
        color: {MAROON}; font-size: .74rem; font-weight: 750;
        text-transform: uppercase; letter-spacing: .12em; margin: .25rem 0 .75rem 0;
    }}
    .fact-label {{ color: {MUTED}; font-size: .7rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }}
    .fact-value {{ color: {INK}; font-weight: 650; margin-top: .25rem; }}
    div[data-testid="stTextArea"] textarea {{
        background: {SURFACE} !important; color: {INK} !important;
        border: 1px solid #BFB0B4 !important; border-radius: 0 !important;
        box-shadow: none !important; line-height: 1.55 !important;
    }}
    div[data-testid="stTextArea"] textarea:focus {{ border: 2px solid {MAROON} !important; box-shadow: none !important; }}
    div[data-testid="stTextArea"] label {{ color: {INK} !important; font-weight: 650 !important; }}
    .stButton > button {{
        width: 100%; min-height: 2.9rem; background: {MAROON} !important;
        color: #FFFFFF !important; border: 1px solid {MAROON} !important;
        border-radius: 0 !important; font-weight: 700 !important; box-shadow: none !important;
    }}
    .stButton > button, .stButton > button * {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }}
    .stButton > button:hover {{ background: {MAROON_DARK} !important; border-color: {MAROON_DARK} !important; }}
    .guidance {{ background: #EFE1E5; border-left: 4px solid {MAROON}; padding: 1rem 1.1rem; margin: .5rem 0 1rem 0; }}
    .guidance-title {{ color: {MAROON}; font-size: .72rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5rem; }}
    .guidance-copy {{ color: #514248; line-height: 1.6; font-size: .9rem; }}
    .answer-panel {{ background: {MAROON_DARK}; border-left: 6px solid {MAROON_LIGHT}; padding: 1.4rem 1.5rem; }}
    .answer-label {{ color: #E8B9C7; font-size: .72rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .7rem; }}
    .answer-value {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-size: clamp(1.4rem, 2.8vw, 2rem); font-weight: 650; line-height: 1.3; overflow-wrap: anywhere; }}
    .muted {{ color: {MUTED}; font-size: .84rem; line-height: 1.5; }}
    .metric-label {{ color: {MUTED}; font-size: .72rem; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }}
    .metric-value {{ color: {MAROON}; font-size: 1.4rem; font-weight: 750; }}
    div[data-testid="stExpander"] {{ border: 1px solid {RULE}; border-radius: 0 !important; background: {SURFACE} !important; }}
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{
        color: {INK} !important; -webkit-text-fill-color: {INK} !important;
    }}
    div[data-testid="stExpander"] a {{ color: {MAROON} !important; }}
    hr {{ border-color: {RULE} !important; }}
    [data-testid="stAlert"] {{ border-radius: 0 !important; }}
    @media (max-width: 700px) {{
        [data-testid="stMainBlockContainer"] {{ padding: 1.2rem 1rem 2rem !important; }}
        .qa-hero {{ padding: 1.4rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the question-answering model…")
def load_qa_model(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "question-answering",
        model=model_id,
        tokenizer=model_id,
        device=device,
    )


def update_context_count():
    # Streamlit updates this state when the text area commits a change.
    st.session_state.context_char_count = len(st.session_state.get("context_input", ""))


# Header is rendered as one self-contained HTML block; no separate CSS class is
# needed on Streamlit's native title, avoiding the missing-title issue.
st.markdown(
    """
    <section class="qa-hero">
      <div class="eyebrow">NLP / EXTRACTIVE QUESTION ANSWERING</div>
      <div class="hero-title">Find the answer in the passage.</div>
      <p class="hero-copy">Provide a passage and ask a question about it. The model identifies an answer span in your text; it does not search the web or generate facts beyond the passage.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

facts = st.columns([1, 1, 2], gap="large")
with facts[0]:
    st.markdown('<div class="fact-label">Architecture</div><div class="fact-value">DistilBERT</div>', unsafe_allow_html=True)
with facts[1]:
    st.markdown('<div class="fact-label">Task</div><div class="fact-value">Answer extraction</div>', unsafe_allow_html=True)
with facts[2]:
    st.markdown('<div class="fact-label">Training data</div><div class="fact-value">SQuAD 1.1 · 20,000 training examples</div>', unsafe_allow_html=True)

st.divider()
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="section-label">01 / Source passage</div>', unsafe_allow_html=True)
    if "context_char_count" not in st.session_state:
        st.session_state.context_char_count = len(st.session_state.get("context_input", ""))
    context = st.text_area(
        "Passage",
        height=300,
        key="context_input",
        on_change=update_context_count,
        placeholder="Paste a passage here…\n\nExample: The University of Chicago was founded in 1890. Its first president was William Rainey Harper.",
        label_visibility="collapsed",
    )
    st.caption(f'{st.session_state.context_char_count:,} characters · updates when the passage change is committed')

with right:
    st.markdown('<div class="section-label">02 / Your question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Question",
        height=130,
        key="question_input",
        placeholder="What would you like to know from this passage?",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="guidance"><div class="guidance-title">How to get a useful answer</div><div class="guidance-copy">Ask one clear question. Include the sentence or paragraph containing the information, and keep the question grounded in that passage.</div></div>',
        unsafe_allow_html=True,
    )
    run = st.button("Extract answer", type="primary", use_container_width=True)

if run:
    if not context.strip() or not question.strip():
        st.warning("Enter both a passage and a question before running the model.")
    else:
        try:
            with st.spinner("Reading the passage and locating an answer…"):
                qa_model = load_qa_model(MODEL_ID)
                result = qa_model(question=question.strip(), context=context.strip())
            st.divider()
            st.markdown('<div class="section-label">03 / Model result</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-panel"><div class="answer-label">Predicted answer</div><div class="answer-value">{html.escape(result["answer"])}</div><div class="muted" style="color:#E0CDD3;margin-top:.8rem;">Extracted span · Characters {result["start"]}–{result["end"]}</div></div>',
                unsafe_allow_html=True,
            )
            details = st.columns([1, 1, 2], gap="large")
            with details[0]:
                st.markdown('<div class="metric-label">Model score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{result["score"]:.4f}</div>', unsafe_allow_html=True)
            with details[1]:
                st.markdown('<div class="metric-label">Device</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fact-value">{"GPU" if torch.cuda.is_available() else "CPU"}</div>', unsafe_allow_html=True)
            with details[2]:
                st.markdown('<div class="muted" style="padding-top:1.1rem;">The model score is not a calibrated probability that the answer is correct.</div>', unsafe_allow_html=True)
        except OSError as exc:
            st.error("Could not load the model from Hugging Face. Check that the repository is public and contains the model weights, config, and tokenizer files.")
            st.caption(str(exc))
        except Exception as exc:
            st.error(f"Unable to generate an answer: {exc}")

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
        **Context-based Question Answering**

        This project fine-tunes DistilBERT for extractive question answering using a subset of SQuAD 1.1. Given a passage and question, the model predicts the start and end positions of an answer span.

        **Selected experiment**
        - Training subset: 20,000 SQuAD examples
        - Training duration: 2 epochs
        - Learning rate: 2e-5
        - Validation Exact Match: 62.00%
        - Validation token F1: 72.81%

        These scores reflect the current project evaluation run. The app is a demonstration and does not guarantee factual correctness.
        """
    )

st.markdown(
    f'<div style="border-top:1px solid {RULE};margin-top:2rem;padding-top:1rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:{MUTED};font-size:.78rem;"><span>Context QA · DistilBERT</span><span>Wheeltrix project · Extractive NLP</span></div>',
    unsafe_allow_html=True,
)
