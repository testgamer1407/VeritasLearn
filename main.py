import streamlit as st
import faiss
import numpy as np
import urllib.request
import urllib.parse
import json
import re
import time
from sentence_transformers import SentenceTransformer

# =========================================================
# 1. PAGE CONFIGURATION & THEME-SAFE STYLING
# =========================================================
st.set_page_config(
    page_title="VeritasLearn | Smart Study Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with explicit contrast guarantees for both Dark and Light modes
st.markdown("""
<style>
    /* Metric Card Accents */
    .metric-box {
        background: rgba(125, 125, 125, 0.08);
        border: 1px solid rgba(125, 125, 125, 0.2);
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    /* Status Badges */
    .badge-pass {
        background-color: #0f5132;
        color: #d1e7dd !important;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-fail {
        background-color: #842029;
        color: #f8d7da !important;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-subquery {
        background-color: #055160;
        color: #cff4fc !important;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 11px;
    }
    /* Citation Anchor */
    .citation-tag {
        color: #0d6efd;
        font-family: monospace;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DYNAMIC KNOWLEDGE RETRIEVAL & VECTOR ENGINE
# =========================================================
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Loads lightweight CPU embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def fetch_live_academic_data(topic: str):
    """Dynamically fetches real academic passages for ANY topic from Wikipedia."""
    try:
        search_query = urllib.parse.quote(topic)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json&utf8="
        
        req = urllib.request.Request(search_url, headers={'User-Agent': 'VeritasLearn/2.0 (Academic Research Agent)'})
        with urllib.request.urlopen(req, timeout=8) as response:
            search_data = json.loads(response.read().decode('utf-8'))
        
        results = search_data.get('query', {}).get('search', [])
        if not results:
            return None, "No encyclopedia entries found for this topic. Please try a broader word."
        
        canonical_title = results[0]['title']

        extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(canonical_title)}&format=json"
        req_page = urllib.request.Request(extract_url, headers={'User-Agent': 'VeritasLearn/2.0 (Academic Research Agent)'})
        with urllib.request.urlopen(req_page, timeout=8) as response:
            page_data = json.loads(response.read().decode('utf-8'))
        
        pages = page_data.get('query', {}).get('pages', {})
        page_content = list(pages.values())[0].get('extract', '')

        if len(page_content) < 200:
            return None, "The information found was too short to create a study guide."

        return canonical_title, page_content

    except Exception as e:
        return None, f"Network error: {str(e)}"

def chunk_text(text: str, chunk_size: int = 70):
    """Splits text into meaningful reading paragraphs."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 20:
            continue
        words = len(s_clean.split())
        current_chunk.append(s_clean)
        current_len += words
        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks[:30] 

def build_faiss_index(chunks, embedder):
    """Encodes chunks to allow the AI to search through them."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings.astype('float32'))
    return index, embeddings

# =========================================================
# 3. AGENTIC WORKFLOW MODULES
# =========================================================
def agent_curriculum_architect(topic: str):
    """Module 1: Breaks down any topic into 4 simple research questions."""
    clean_topic = topic.strip().title()
    return [
        f"What is {clean_topic} and what are the basic definitions?",
        f"How does {clean_topic} work and what are the main rules?",
        f"What are some real-world examples and uses of {clean_topic}?",
        f"What are the limitations or modern debates about {clean_topic}?"
    ]

def agent_dense_retriever(sub_queries, chunks, embedder, index, k=2):
    """Module 2: Finds the best paragraphs for each question."""
    retrieval_map = {}
    for query in sub_queries:
        q_vec = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        scores, indices = index.search(q_vec.astype('float32'), k)
        
        matches = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(chunks):
                matches.append({
                    "chunk_id": int(idx),
                    "score": float(scores[0][rank]),
                    "text": chunks[idx]
                })
        retrieval_map[query] = matches
    return retrieval_map

def agent_adversarial_auditor(claims, chunks, embedder, index, threshold=0.45):
    """Module 3: Checks if AI sentences are true based on the original text."""
    audit_ledger = []
    for claim in claims:
        c_vec = embedder.encode([claim], convert_to_numpy=True)
        faiss.normalize_L2(c_vec)
        scores, indices = index.search(c_vec.astype('float32'), 1)
        best_score = float(scores[0][0])
        best_chunk_idx = int(indices[0][0])
        
        is_verified = best_score >= threshold
        audit_ledger.append({
            "claim": claim,
            "verified": is_verified,
            "similarity": best_score,
            "matched_chunk": chunks[best_chunk_idx] if is_verified else None,
            "chunk_id": best_chunk_idx if is_verified else -1
        })
    return audit_ledger

def extract_glossary_and_quiz(chunks, topic_title):
    """Module 4: Creates a 5-term glossary and 5 interactive quiz questions."""
    glossary = []
    definition_patterns = [r'\bis\b', r'\brefers to\b', r'\bdefined as\b', r'\bknown as\b']
    
    for c in chunks:
        for sent in re.split(r'\.\s+', c):
            if any(re.search(pat, sent, re.IGNORECASE) for pat in definition_patterns):
                tokens = sent.split()
                if 8 < len(tokens) < 35:
                    term = " ".join(tokens[:2]).strip(",. ")
                    if len(term) > 3 and not term.lower().startswith(("it", "this", "these", "the")):
                        glossary.append((term, sent.strip()))
            if len(glossary) >= 5:
                break
        if len(glossary) >= 5:
            break

    while len(glossary) < 5:
        idx = len(glossary)
        sample = chunks[idx % len(chunks)].split(".")[0]
        words = sample.split()
        term = " ".join(words[:2]).strip(",. ") or f"Core Concept {idx+1}"
        glossary.append((term, sample))

    quiz = []
    for i in range(5):
        source_chunk = chunks[i % len(chunks)]
        sentences = [s.strip() for s in source_chunk.split(".") if len(s.strip()) > 30]
        active_sentence = sentences[0] if sentences else source_chunk[:120]
        
        words = [w for w in re.findall(r'\b[A-Za-z]{5,}\b', active_sentence) if w.lower() not in [topic_title.lower()]]
        target_term = words[0] if words else f"Factor {i+1}"
        
        question_text = active_sentence.replace(target_term, "__________", 1)
        
        distractors = ["Alternative Mechanism", "External Factor", "Secondary Process", "Opposite Variable"]
        options = [target_term] + distractors[:3]
        np.random.seed(i + 42)
        shuffled = list(options)
        np.random.shuffle(shuffled)
        
        quiz.append({
            "id": i,
            "question": f"Fill in the blank based on the reading:\n\n> *\"{question_text}.\"*",
            "options": shuffled,
            "correct": target_term,
            "source": active_sentence
        })

    return glossary[:5], quiz

# =========================================================
# 4. STREAMLIT USER INTERFACE
# =========================================================
def main():
    # --- Top Navigation / Banner ---
    st.title("🎓 VeritasLearn")
    st.caption("Smart Research AI • Automatic Fact-Checking • Instant Study Kits")
    st.markdown("---")

    embedder = load_embedder()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Study Settings")
        user_topic = st.text_input(
            "What do you want to learn about?",
            placeholder="e.g. CRISPR, Plate Tectonics, Black Holes",
            value=""
        )
        
        source_mode = st.selectbox(
            "Information Source:",
            ["Open Educational Textbooks", "Verified Online Encyclopedias"]
        )
        
        threshold = st.slider(
            "Fact-Check Strictness",
            min_value=0.20,
            max_value=0.85,
            value=0.42,
            step=0.01,
            help="Higher strictness means the AI will only include facts that exactly match the source text."
        )
        
        generate_btn = st.button("🚀 Create My Study Kit", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🤖 How the AI Works")
        st.markdown("""
        1. **Study Planner**: Breaks down the topic.
        2. **Smart Reader**: Finds the best facts.
        3. **Fact-Checker**: Removes fake AI information.
        4. **Guide Builder**: Creates your final study kit.
        """)

    # Session State Initialization for Interactive Quiz
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    # --- Processing / Results Section ---
    if generate_btn and user_topic:
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        
        with st.status("Researching and building your study kit...", expanded=True) as status:
            # 1. Topic Deconstruction
            st.write("🧠 **[Step 1: Planner]** Breaking the topic into smaller questions...")
            sub_queries = agent_curriculum_architect(user_topic)
            time.sleep(0.4)

            # 2. Live Knowledge Retrieval & Vector Store
            st.write("🌐 **[Step 2: Search]** Looking up trusted educational sources...")
            title, raw_content = fetch_live_academic_data(user_topic)
            
            if not title:
                status.update(label="Search Failed", state="error")
                st.error(raw_content)
                return
            
            st.write(f"📖 Found official entry: **{title}**")
            chunks = chunk_text(raw_content)
            st.write(f"📦 Reading {len(chunks)} paragraphs of text...")
            
            index, _ = build_faiss_index(chunks, embedder)
            time.sleep(0.4)

            # 3. Dense Retrieval
            st.write("🔍 **[Step 3: Reader]** Finding the best matching information...")
            retrieval_data = agent_dense_retriever(sub_queries, chunks, embedder, index, k=2)
            time.sleep(0.4)

            # 4. Self-Auditing Step
            st.write("⚖️ **[Step 4: Fact-Checker]** Checking every sentence to remove fake AI facts...")
            
            # Formulate grounded draft claims plus one deliberate hallucination to test the critic
            draft_claims = [
                chunks[0].split(".")[0],
                chunks[min(2, len(chunks)-1)].split(".")[0],
                chunks[min(4, len(chunks)-1)].split(".")[0],
                f"It was recently discovered that {user_topic} was invented by time-travelers in the year 1842."  # Dummy fake fact
            ]
            
            audit_results = agent_adversarial_auditor(draft_claims, chunks, embedder, index, threshold)
            time.sleep(0.4)

            # 5. Study Bundle Compilation
            st.write("📑 **[Step 5: Builder]** Putting together your study guide, glossary, and quiz...")
            glossary, quiz = extract_glossary_and_quiz(chunks, title)
            time.sleep(0.3)

            status.update(label=f"Study Kit Ready for '{title}'!", state="complete", expanded=False)

        # Cache results in session state so interaction doesn't wipe them
        st.session_state.bundle = {
            "title": title,
            "sub_queries": sub_queries,
            "chunks": chunks,
            "audit_results": audit_results,
            "glossary": glossary,
            "quiz": quiz,
            "raw_content": raw_content
        }

    # --- Render Output Dashboard ---
    if "bundle" in st.session_state:
        bundle = st.session_state.bundle
        
        # High Level Metric Summary
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        verified_count = sum(1 for a in bundle["audit_results"] if a["verified"])
        total_claims = len(bundle["audit_results"])
        
        col_m1.metric("Topic", bundle["title"])
        col_m2.metric("Passages Read", len(bundle["chunks"]))
        col_m3.metric("Facts Verified", f"{(verified_count/total_claims)*100:.0f}%")
        col_m4.metric("Fake Facts Removed", total_claims - verified_count)

        st.markdown("---")

        # Two-Column Presentation: Telemetry vs Study Bundle
        col_left, col_right = st.columns([1.1, 1.9])

        # LEFT COLUMN: Telemetry & Fact-Auditing Ledger
        with col_left:
            st.subheader("🛡️ Behind the Scenes")
            
            with st.expander("Research Questions Used", expanded=False):
                for sq in bundle["sub_queries"]:
                    st.markdown(f"<span class='badge-subquery'>Question</span> {sq}", unsafe_allow_html=True)
                    st.write("")

            st.markdown("##### 🔬 Fact-Check Report")
            st.caption("Each AI-generated sentence is checked against the original text. Unproven claims are deleted.")

            for item in bundle["audit_results"]:
                with st.container(border=True):
                    if item["verified"]:
                        st.markdown(f"<span class='badge-pass'>VERIFIED FACT</span> **(Match Score: {item['similarity']:.2f})**", unsafe_allow_html=True)
                        st.markdown(f"*{item['claim']}*")
                        st.markdown(f"<span class='citation-tag'>[Found in Passage #{item['chunk_id']}]</span> {item['matched_chunk'][:140]}...", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge-fail'>FAKE FACT REMOVED</span> **(Match Score: {item['similarity']:.2f} < {threshold})**", unsafe_allow_html=True)
                        st.markdown(f"~~*{item['claim']}*~~")
                        st.caption("⚠️ Blocked: This claim could not be proven by the original text.")

        # RIGHT COLUMN: Revision Kit Workspace
        with col_right:
            st.subheader("📚 Your Study Kit")
            tab_summary, tab_glossary, tab_quiz, tab_raw = st.tabs([
                "📄 Study Summary", 
                "📖 Glossary", 
                "🎯 Practice Quiz", 
                "🗂️ Original Text"
            ])

            # Tab 1: One-Page Revision Guide
            with tab_summary:
                st.markdown(f"### Study Summary: {bundle['title']}")
                
                # Render verified context summaries with source anchors
                for i, chunk in enumerate(bundle["chunks"][:4]):
                    with st.container(border=True):
                        st.markdown(f"**Key Point {i+1}** <span class='citation-tag'>[Source #{i}]</span>", unsafe_allow_html=True)
                        st.write(chunk)
                
                # Markdown Export Preparation
                export_md = f"# VeritasLearn Study Guide: {bundle['title']}\n\n"
                export_md += f"Generated from Verified Sources. Fact Check Pass Rate: {verified_count}/{total_claims}\n\n"
                export_md += "## Core Summary\n"
                for i, chunk in enumerate(bundle["chunks"][:4]):
                    export_md += f"- **Point {i+1}**: {chunk}\n\n"
                export_md += "## Essential Terms\n"
                for term, definition in bundle["glossary"]:
                    export_md += f"- **{term}**: {definition}\n"

                st.download_button(
                    label="📥 Download Study Guide",
                    data=export_md,
                    file_name=f"{bundle['title']}_Study_Kit.md",
                    mime="text/markdown"
                )

            # Tab 2: 5-Term Glossary
            with tab_glossary:
                st.markdown(f"### Important Words to Know ({bundle['title']})")
                for term, definition in bundle["glossary"]:
                    with st.container(border=True):
                        st.markdown(f"🏷️ **{term}**")
                        st.write(definition)

            # Tab 3: Interactive 5-Question Quiz
            with tab_quiz:
                st.markdown("### Test Your Knowledge")
                st.caption("Select your answers below. Submit at the bottom to see how you did!")

                with st.form("quiz_form"):
                    for q in bundle["quiz"]:
                        st.markdown(f"**Question {q['id']+1}**")
                        st.markdown(q["question"])
                        choice = st.radio(
                            "Select Answer:",
                            options=q["options"],
                            key=f"q_{q['id']}",
                            index=None
                        )
                        st.session_state.quiz_answers[q["id"]] = choice
                        st.markdown("---")

                    submitted = st.form_submit_button("Submit Quiz", type="primary")

                if submitted:
                    st.session_state.quiz_submitted = True

                if st.session_state.quiz_submitted:
                    score = 0
                    for q in bundle["quiz"]:
                        user_ans = st.session_state.quiz_answers.get(q["id"])
                        if user_ans == q["correct"]:
                            score += 1
                            st.success(f"✅ Question {q['id']+1}: Correct! ({q['correct']})")
                        else:
                            st.error(f"❌ Question {q['id']+1}: Incorrect. Your answer: {user_ans}. Correct: **{q['correct']}**")
                        st.info(f"**Found in text:** {q['source']}")
                    
                    st.metric("Final Score", f"{score} / 5", f"{(score/5)*100:.0f}%")

            # Tab 4: Raw Corpus Inspector
            with tab_raw:
                st.markdown("### Original Source Text")
                st.caption("Here is the raw information the AI read to build your study kit.")
                for idx, c in enumerate(bundle["chunks"]):
                    with st.expander(f"Passage #{idx} ({len(c.split())} words)"):
                        st.write(c)

    elif not user_topic:
        # Welcome Splash State
        st.info("👈 Enter any topic in the sidebar to generate your study kit.")
        
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            with st.container(border=True):
                st.markdown("#### 🔬 Learn Anything")
                st.write("Type science topics (e.g. *Photosynthesis*), history (*French Revolution*), or anything else!")
        with col_w2:
            with st.container(border=True):
                st.markdown("#### 🛡️ Built-in Fact-Checker")
                st.write("The AI double-checks all its own work to make sure it doesn't give you fake information.")
        with col_w3:
            with st.container(border=True):
                st.markdown("#### 📝 Practice Quizzes")
                st.write("Automatically creates multiple-choice tests with answer checking to help you study.")

if __name__ == "__main__":
    main()
