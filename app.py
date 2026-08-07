from __future__ import annotations

import html
from textwrap import dedent
from typing import Any

import streamlit as st

from src.v3_backend import (
    V3Backend,
    extract_articles_from_uploaded_file,
    split_pasted_articles,
)


# =====================================================================
# Page configuration
# =====================================================================
st.set_page_config(
    page_title="Multilingual RAG",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Built-in demonstrations
# =====================================================================
DEMO_PLACEHOLDER = "Choose a built-in demonstration"

DEMOS: dict[str, dict[str, str]] = {
    "Spanish — Tournament final": {
        "description": (
            "A Spanish comparison between the tournament's opening "
            "match in Munich and its final in Berlin."
        ),
        "question": (
            "¿Qué ciudad albergó la final del torneo?"
        ),
        "articles": """
El torneo terminó el domingo en Berlín. La final se disputó en el Estadio Olímpico de Berlín ante más de 70.000 espectadores.

---

Múnich albergó el partido inaugural del torneo. El encuentro se celebró en el Allianz Arena durante la primera semana de la competición.
        """.strip(),
    },
    "Arabic — Protest timeline": {
        "description": (
            "An Arabic event-order question involving protests and a "
            "later Supreme Court decision."
        ),
        "question": (
            "ما الحدث الذي وقع أولاً، الاحتجاجات المطالبة بالإفراج عن "
            "منسقي إصلاح الحصص أم قرار المحكمة العليا بشأن الحصص الوظيفية؟"
        ),
        "articles": """
بدأت الاحتجاجات المطالبة بالإفراج عن منسقي إصلاح الحصص قبل صدور قرار المحكمة العليا بشأن الحصص الوظيفية. واستمرت المظاهرات بعد ذلك في عدة مدن.

---

أعلنت المحكمة العليا لاحقًا قرارها بشأن تعديل حصص الوظائف الحكومية. وناقشت الحكومة كيفية تنفيذ القرار.
        """.strip(),
    },
}


# =====================================================================
# Modern interface styling
# =====================================================================
st.markdown(
    dedent("""
    <style>
        :root {
            --panel: rgba(18, 25, 39, 0.78);
            --panel-soft: rgba(30, 41, 59, 0.58);
            --border: rgba(148, 163, 184, 0.22);
            --text-soft: #94a3b8;
            --accent: #8b5cf6;
            --accent-two: #06b6d4;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 12% 5%,
                    rgba(124, 58, 237, 0.20),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 88% 10%,
                    rgba(6, 182, 212, 0.14),
                    transparent 26%
                ),
                linear-gradient(
                    160deg,
                    #080b12 0%,
                    #0b1120 55%,
                    #080b12 100%
                );
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    rgba(15, 23, 42, 0.98),
                    rgba(10, 15, 27, 0.98)
                );
            border-right: 1px solid var(--border);
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 2.2rem 2.3rem;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(124, 58, 237, 0.22),
                    rgba(6, 182, 212, 0.10)
                ),
                rgba(15, 23, 42, 0.78);
            box-shadow:
                0 22px 60px rgba(0, 0, 0, 0.32),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -90px;
            top: -120px;
            border-radius: 50%;
            background: rgba(139, 92, 246, 0.17);
            filter: blur(12px);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.76rem;
            border-radius: 999px;
            border: 1px solid rgba(167, 139, 250, 0.30);
            background: rgba(124, 58, 237, 0.16);
            color: #ddd6fe;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: clamp(2.25rem, 5vw, 4rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #ddd6fe 50%,
                    #a5f3fc
                );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 790px;
            margin: 1rem 0 0;
            color: #cbd5e1;
            font-size: 1.06rem;
            line-height: 1.75;
        }

        .pipeline-row {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.4rem;
        }

        .pipeline-chip {
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.72);
            color: #dbeafe;
            padding: 0.42rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 650;
        }

        .section-label {
            color: #a5b4fc;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .section-description {
            color: var(--text-soft);
            margin-bottom: 1rem;
        }

        .answer-card {
            position: relative;
            border: 1px solid rgba(139, 92, 246, 0.38);
            border-radius: 20px;
            padding: 1.45rem 1.55rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(124, 58, 237, 0.14),
                    rgba(6, 182, 212, 0.08)
                ),
                rgba(15, 23, 42, 0.82);
            box-shadow:
                0 16px 40px rgba(0, 0, 0, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            font-size: 1.25rem;
            line-height: 1.8;
            margin-bottom: 1.1rem;
        }

        .answer-label {
            color: #a5b4fc;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-bottom: 1rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            background: rgba(16, 185, 129, 0.10);
            border: 1px solid rgba(16, 185, 129, 0.26);
            color: #a7f3d0;
            font-size: 0.78rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            background: rgba(15, 23, 42, 0.62);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 14px;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.68);
        }

        div[data-testid="stFileUploader"] {
            border-radius: 16px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55);
            overflow: hidden;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 750;
            border: 1px solid rgba(139, 92, 246, 0.35);
        }

        div.stButton > button[kind="primary"] {
            background:
                linear-gradient(
                    90deg,
                    #7c3aed,
                    #2563eb
                );
            border: none;
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.24);
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 34px rgba(124, 58, 237, 0.32);
        }

        hr {
            border-color: var(--border);
        }

        @media (max-width: 720px) {
            .hero {
                padding: 1.5rem;
            }

            .hero h1 {
                font-size: 2.4rem;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """),
    unsafe_allow_html=True,
)


# =====================================================================
# Session-state callbacks
# =====================================================================
def clear_result() -> None:
    """Clear a previous answer whenever input changes."""

    st.session_state.pop(
        "v3_result",
        None,
    )


def load_selected_demo() -> None:
    """Load the selected built-in demonstration."""

    selected_demo = st.session_state.get(
        "demo_selector",
        DEMO_PLACEHOLDER,
    )

    demo = DEMOS.get(
        selected_demo
    )

    if demo is None:
        return

    st.session_state["question_input"] = demo[
        "question"
    ]

    st.session_state["pasted_articles_input"] = demo[
        "articles"
    ]

    st.session_state.pop(
        "uploaded_files",
        None,
    )

    clear_result()


def clear_interface() -> None:
    """Reset all frontend inputs and results."""

    st.session_state["question_input"] = ""
    st.session_state["pasted_articles_input"] = ""
    st.session_state["demo_selector"] = DEMO_PLACEHOLDER

    st.session_state.pop(
        "uploaded_files",
        None,
    )

    clear_result()


# =====================================================================
# Cached backend
# =====================================================================
@st.cache_resource(
    show_spinner=False
)
def load_backend() -> V3Backend:
    """
    Load BGE-M3 and the OpenAI client once.
    """

    return V3Backend()


# =====================================================================
# Article processing
# =====================================================================
def collect_articles(
    pasted_text: str,
    uploaded_files: list[Any],
) -> list[str]:
    """
    Combine pasted and uploaded articles while removing exact duplicates.
    """

    articles: list[str] = []

    if pasted_text.strip():
        articles.extend(
            split_pasted_articles(
                pasted_text
            )
        )

    for uploaded_file in uploaded_files:
        articles.extend(
            extract_articles_from_uploaded_file(
                filename=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
            )
        )

    unique_articles: list[str] = []
    seen_articles: set[str] = set()

    for article in articles:
        cleaned_article = article.strip()

        if not cleaned_article:
            continue

        normalized_article = " ".join(
            cleaned_article.split()
        )

        if normalized_article in seen_articles:
            continue

        seen_articles.add(
            normalized_article
        )

        unique_articles.append(
            cleaned_article
        )

    return unique_articles


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## 🌐 Multilingual RAG")

    st.caption(
        "Final V3 research prototype"
    )

    st.markdown(
        dedent("""
        <div class="status-row">
            <span class="status-pill">● Backend connected</span>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Final configuration")

    sidebar_metric_one, sidebar_metric_two = st.columns(
        2
    )

    sidebar_metric_one.metric(
        "Raw pool",
        "20",
    )

    sidebar_metric_two.metric(
        "Final evidence",
        "12",
    )

    st.metric(
        "Context reduction",
        "40%",
    )

    st.divider()

    st.markdown("### Models")

    st.caption("Embedding")
    st.code(
        "BAAI/bge-m3",
        language=None,
    )

    st.caption("Generation")
    st.code(
        "gpt-5.4",
        language=None,
    )

    st.divider()

    st.markdown("### Filtering")

    st.markdown(
        """
        - Semantic threshold: **0.92**
        - Article cap: **4 sentences**
        - MMR lambda: **0.85**
        """
    )

    st.divider()

    st.caption(
        "This research prototype answers questions from candidate "
        "articles supplied by the user. Automatic web retrieval is "
        "outside the evaluated system."
    )


# =====================================================================
# Hero
# =====================================================================
st.markdown(
    dedent("""
    <section class="hero">
        <div class="hero-badge">● Final V3 Research Prototype</div>

        <h1>Multilingual RAG</h1>

        <p>
            Ask a question in Arabic, Spanish, English, or another
            language. The system retrieves sentence-level evidence,
            removes redundancy, balances relevance and diversity,
            and generates a concise multilingual answer.
        </p>

        <div class="pipeline-row">
            <span class="pipeline-chip">Question</span>
            <span class="pipeline-chip">BGE-M3 Top-20</span>
            <span class="pipeline-chip">Duplicate filtering</span>
            <span class="pipeline-chip">MMR selection</span>
            <span class="pipeline-chip">Filtered Top-12</span>
            <span class="pipeline-chip">GPT-5.4 answer</span>
        </div>
    </section>
    """),
    unsafe_allow_html=True,
)


# =====================================================================
# Demonstration selector
# =====================================================================
st.markdown(
    dedent("""
    <div class="section-label">Quick start</div>
    <div class="section-title">Load a demonstration</div>
    <div class="section-description">
        Try the complete pipeline instantly with a prepared Spanish
        or Arabic example.
    </div>
    """),
    unsafe_allow_html=True,
)

demo_column, load_column, clear_column = st.columns(
    [3.4, 1.2, 1.0]
)

with demo_column:
    selected_demo = st.selectbox(
        "Built-in demonstrations",
        options=[
            DEMO_PLACEHOLDER,
            *DEMOS.keys(),
        ],
        key="demo_selector",
        label_visibility="collapsed",
    )

with load_column:
    st.button(
        "Load demo",
        on_click=load_selected_demo,
        disabled=(
            selected_demo == DEMO_PLACEHOLDER
        ),
        use_container_width=True,
    )

with clear_column:
    st.button(
        "Clear",
        on_click=clear_interface,
        use_container_width=True,
    )

if selected_demo in DEMOS:
    st.caption(
        DEMOS[selected_demo][
            "description"
        ]
    )

st.divider()


# =====================================================================
# Question input
# =====================================================================
st.markdown(
    dedent("""
    <div class="section-label">Step 1</div>
    <div class="section-title">Ask a question</div>
    <div class="section-description">
        The answer will be generated in the same language as the question.
    </div>
    """),
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    key="question_input",
    placeholder=(
        "Enter a multilingual question..."
    ),
    height=125,
    on_change=clear_result,
)


# =====================================================================
# Article input
# =====================================================================
st.markdown(
    dedent("""
    <div class="section-label">Step 2</div>
    <div class="section-title">Provide candidate articles</div>
    <div class="section-description">
        Paste text or upload TXT, JSON, or JSONL files. Use three
        dashes on their own line to separate pasted articles.
    </div>
    """),
    unsafe_allow_html=True,
)

paste_tab, upload_tab = st.tabs(
    [
        "✍️ Paste articles",
        "📁 Upload files",
    ]
)

with paste_tab:
    pasted_articles = st.text_area(
        "Candidate article text",
        key="pasted_articles_input",
        placeholder=(
            "Article one...\n\n"
            "---\n\n"
            "Article two..."
        ),
        height=285,
        on_change=clear_result,
        label_visibility="collapsed",
    )

with upload_tab:
    uploaded_files = st.file_uploader(
        "Upload article files",
        type=[
            "txt",
            "json",
            "jsonl",
        ],
        accept_multiple_files=True,
        key="uploaded_files",
        on_change=clear_result,
    )

    if uploaded_files:
        st.success(
            f"{len(uploaded_files)} file(s) ready for analysis."
        )

        with st.expander(
            "Selected files"
        ):
            for uploaded_file in uploaded_files:
                st.write(
                    f"📄 {uploaded_file.name}"
                )


# =====================================================================
# Generate button
# =====================================================================
st.write("")

generate_left, generate_center, generate_right = st.columns(
    [1, 2.4, 1]
)

with generate_center:
    generate_button = st.button(
        "✨ Generate multilingual answer",
        type="primary",
        use_container_width=True,
    )

st.caption(
    "The first run may take longer while BGE-M3 loads into memory."
)


# =====================================================================
# Execute V3
# =====================================================================
if generate_button:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning(
            "Enter a question before generating an answer."
        )

    else:
        try:
            articles = collect_articles(
                pasted_text=pasted_articles,
                uploaded_files=uploaded_files or [],
            )

            if not articles:
                st.warning(
                    "Paste or upload at least one candidate article."
                )

            else:
                progress_message = st.empty()

                progress_message.info(
                    "Loading the final V3 backend..."
                )

                with st.spinner(
                    "Retrieving, filtering, and generating..."
                ):
                    backend = load_backend()

                    progress_message.info(
                        "Running BGE-M3 retrieval and MMR filtering..."
                    )

                    result = backend.answer(
                        question=cleaned_question,
                        articles=articles,
                    )

                progress_message.empty()

                st.session_state[
                    "v3_result"
                ] = result

        except Exception as error:
            st.error(
                "The system could not generate an answer."
            )

            with st.expander(
                "Technical error details"
            ):
                st.exception(
                    error
                )


# =====================================================================
# Results
# =====================================================================
result = st.session_state.get(
    "v3_result"
)

if result:
    st.divider()

    st.markdown(
        dedent("""
        <div class="section-label">Result</div>
        <div class="section-title">Generated answer</div>
        <div class="section-description">
            Generated from the final filtered evidence.
        </div>
        """),
        unsafe_allow_html=True,
    )

    answer_direction = (
        "rtl"
        if result["language"] == "ar"
        else "ltr"
    )

    safe_answer = html.escape(
        result["answer"]
    )

    st.markdown(
        dedent(f"""
        <div class="answer-card" dir="{answer_direction}">
            <div class="answer-label">Final answer</div>
            {safe_answer}
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        dedent("""
        <div class="status-row">
            <span class="status-pill">✓ Generation complete</span>
            <span class="status-pill">✓ Evidence available</span>
        </div>
        """),
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Articles",
        result["article_count"],
    )

    metric_columns[1].metric(
        "Sentences",
        result["sentence_count"],
    )

    metric_columns[2].metric(
        "Raw candidates",
        result["raw_candidate_count"],
    )

    metric_columns[3].metric(
        "Final evidence",
        result["final_evidence_count"],
    )

    metric_columns[4].metric(
        "Language",
        result["language"].upper(),
    )

    st.write("")

    with st.expander(
        "🔎 Explore filtered evidence",
        expanded=False,
    ):
        for evidence in result["evidence"]:
            final_rank = (
                evidence["final_rank"]
                if evidence["final_rank"] is not None
                else "—"
            )

            evidence_title = (
                f"Evidence {final_rank} · "
                f"Article {evidence['article_number']} · "
                f"Sentence {evidence['sentence_number']}"
            )

            with st.container(
                border=True
            ):
                st.markdown(
                    f"#### {evidence_title}"
                )

                evidence_direction = (
                    "rtl"
                    if result["language"] == "ar"
                    else "ltr"
                )

                safe_evidence = html.escape(
                    evidence["text"]
                )

                st.markdown(
                    dedent(f"""
                    <div dir="{evidence_direction}"
                         style="
                            font-size: 1.03rem;
                            line-height: 1.7;
                            margin: 0.45rem 0 0.9rem;
                         ">
                        {safe_evidence}
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

                score_one, score_two, score_three = st.columns(
                    3
                )

                score_one.metric(
                    "Relevance",
                    f"{evidence['relevance_score']:.4f}",
                )

                if evidence["mmr_score"] is not None:
                    score_two.metric(
                        "MMR",
                        f"{evidence['mmr_score']:.4f}",
                    )

                score_three.metric(
                    "Raw rank",
                    evidence["raw_rank"] or "—",
                )

    with st.expander(
        "📊 Technical retrieval details"
    ):
        detail_one, detail_two, detail_three = st.columns(
            3
        )

        detail_one.metric(
            "Exact duplicates removed",
            result["exact_duplicates_removed"],
        )

        detail_two.metric(
            "Semantic duplicates removed",
            result["semantic_duplicates_removed"],
        )

        detail_three.metric(
            "Evidence retained",
            result["final_evidence_count"],
        )

        st.json(
            {
                "embedding_model": result[
                    "embedding_model"
                ],
                "generation_model": result[
                    "generation_model"
                ],
                "detected_language": result[
                    "language"
                ],
                "response_id": result[
                    "response_id"
                ],
            }
        )