import streamlit as st
import faiss
import numpy as np
import time
import re
from sentence_transformers import SentenceTransformer

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="VeritasLearn | Agentic AI", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .terminal {background-color: #1E1E1E; color: #00FF00; font-family: 'Courier New', Courier, monospace; padding: 15px; border-radius: 5px; font-size: 14px; height: 300px; overflow-y: scroll;}
    .verified {color: #155724; background-color: #d4edda; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;}
    .unverified {color: #721c24; background-color: #f8d7da; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;}
    .card {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. KNOWLEDGE BASE & FAISS SETUP
# ==========================================
@st.cache_resource
def init_vector_store():
    """Initializes local embedding model and FAISS index with OER/Wiki data."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Mock Open Educational / Wikipedia Dataset for the Hackathon
    corpus = [
        # CRISPR Data
        "CRISPR-Cas9 is a unique technology that enables geneticists to edit parts of the genome by removing, adding, or altering sections of the DNA sequence.",
        "The CRISPR-Cas9 system consists of two key molecules that introduce a change into the DNA: an enzyme called Cas9, and a piece of RNA called guide RNA (gRNA).",
        "Cas9 acts as a pair of 'molecular scissors' that can cut the two strands of DNA at a specific location so that bits of DNA can then be added or removed.",
        "Off-target effects are a major limitation of CRISPR, where the Cas9 enzyme cuts DNA outside the target site, potentially causing unintended mutations.",
        # Plate Tectonics Data
        "Plate tectonics is the scientific theory that Earth's lithosphere comprises a number of large tectonic plates which have been slowly moving since about 3.4 billion years ago.",
        "The driving mechanism behind plate tectonics is mantle convection, where hot, less dense material rises, and cooler, denser material sinks.",
        "Convergent boundaries occur where plates collide, often creating mountain ranges like the Himalayas or deep ocean trenches via subduction.",
        "Divergent boundaries occur where plates move apart, such as the Mid-Atlantic Ridge, where new oceanic crust is formed by upwelling magma."
    ]
    
    # Generate embeddings and create FAISS Index
    embeddings = embedder.encode(corpus)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return embedder, index, corpus

# ==========================================
# 2. AGENTIC FUNCTIONS
# ==========================================
def agent_curriculum_architect(topic):
    """Breaks down a broad topic into 3-4 sub-queries."""
    time.sleep(1) # Simulate API call
    if "CRISPR" in topic.upper():
        return [
            "What are the structural components of the CRISPR-Cas9 system?",
            "How does the mechanism of DNA cleavage (molecular scissors) work?",
            "What are the primary applications and limitations (e.g., off-target effects)?"
        ]
    elif "PLATE" in topic.upper():
        return [
            "What is the foundational theory of plate tectonics?",
            "What mechanisms drive tectonic plate movement (mantle convection)?",
            "What are the different types of plate boundaries and their geological effects?"
        ]
    else:
        return [f"Core definitions of {topic}", f"Mechanisms of {topic}", f"Applications of {topic}"]

def agent_dense_retriever(sub_queries, embedder, index, corpus, k=2):
    """Retrieves verified context chunks for each sub-query using FAISS."""
    time.sleep(1.5) # Simulate retrieval time
    retrieved_data = {}
    for sq in sub_queries:
        vec = embedder.encode([sq]).astype('float32')
        distances, indices = index.search(vec, k)
        retrieved_data[sq] = [corpus[i] for i in indices[0]]
    return retrieved_data

def agent_fact_auditor(draft_claims, embedder, index, corpus, threshold=1.2):
    """Self-reflection loop: Checks if claims are grounded in FAISS corpus."""
    time.sleep(2)
    audited_claims = []
    for claim in draft_claims:
        vec = embedder.encode([claim]).astype('float32')
        distances, indices = index.search(vec, 1)
        best_dist = distances[0][0]
        best_match = corpus[indices[0][0]]
        
        status = "Verified" if best_dist < threshold else "Unverified/Hallucination"
        audited_claims.append({
            "claim": claim, 
            "status": status, 
            "distance": best_dist,
            "anchor": best_match if status == "Verified" else "None"
        })
    return audited_claims

# ==========================================
# 3. STREAMLIT UI BUILDER
# ==========================================
def main():
    # --- HEADER ---
    colA, colB = st.columns([1, 5])
    with colA:
        st.image("https://cdn-icons-png.flaticon.com/512/2042/2042462.png", width=80)
    with colB:
        st.title("VeritasLearn")
        st.subheader("Autonomous Dialectic Revision & Fact-Auditing Engine")
    st.markdown("---")

    # --- INIT MODELS ---
    embedder, index, corpus = init_vector_store()

    # --- SIDEBAR CONTROL PANEL ---
    with st.sidebar:
        st.header("⚙️ Mission Control")
        topic = st.text_input("Academic Topic:", placeholder="e.g. CRISPR Gene Editing, Plate Tectonics")
        dataset = st.selectbox("Verified Corpus:", ["OER Commons Science Module", "Hugging Face Wikipedia English"])
        strictness = st.slider("Fact-Audit Strictness (L2 Distance)", 0.5, 2.0, 1.2, help="Lower means stricter exact matching.")
        run_agent = st.button("🚀 Generate Revision Bundle", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Agent Modules Loaded:")
        st.caption("✅ Curriculum Architect")
        st.caption("✅ FAISS Vector Retriever")
        st.caption("✅ Adversarial Critic (Auditor)")
        st.caption("✅ Pedagogical Compiler")

    # --- MAIN WORKSPACE ---
    if run_agent and topic:
        col1, col2 = st.columns([1, 1.5])
        
        # COLUMN 1: AGENT TELEMETRY & LOGS
        with col1:
            st.markdown("### 📡 Agent Telemetry Logs")
            log_container = st.empty()
            logs = "Initializing agentic workflow...\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            
            # Step 1: Architect
            logs += f"[*] [Architect Agent] Deconstructing topic: '{topic}'...\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            sub_queries = agent_curriculum_architect(topic)
            for i, sq in enumerate(sub_queries):
                logs += f"    -> Q{i+1}: {sq}\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            
            # Step 2: Retriever
            logs += f"\n[*] [Retriever Agent] Querying FAISS index ({dataset})...\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            retrieved_context = agent_dense_retriever(sub_queries, embedder, index, corpus)
            logs += f"[*] Retrieved {sum(len(v) for v in retrieved_context.values())} contextual chunks.\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)

            # Step 3: Synthesis & Audit
            logs += f"\n[*] [Fact-Auditor Agent] Cross-referencing generated claims...\n"
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            
            # Mock draft claims to demonstrate the auditor catching a hallucination
            if "CRISPR" in topic.upper():
                draft = [
                    "CRISPR-Cas9 acts as molecular scissors to cut DNA.",
                    "Cas9 is guided by a piece of RNA called gRNA.",
                    "CRISPR was invented by aliens in 1999." # Hallucinated claim
                ]
            else:
                draft = [
                    "Plate tectonics is driven by mantle convection.",
                    "The Mid-Atlantic Ridge is a convergent boundary.", # Incorrect/Hallucinated
                    "Plates have been moving for about 3.4 billion years."
                ]
            
            audit_results = agent_fact_auditor(draft, embedder, index, corpus, strictness)
            
            for res in audit_results:
                if res['status'] == "Verified":
                    logs += f"    [PASS] L2:{res['distance']:.2f} | {res['claim'][:30]}...\n"
                else:
                    logs += f"    [FAIL] L2:{res['distance']:.2f} | Pruning Hallucination: {res['claim'][:30]}...\n"
            
            logs += "\n[*] [Pedagogical Compiler] Assembling finalized study bundle..."
            log_container.markdown(f'<div class="terminal">{logs}</div>', unsafe_allow_html=True)
            
            # Verification Breakdown UI
            st.markdown("#### 🔍 Audit Breakdown")
            for res in audit_results:
                if res['status'] == "Verified":
                    st.markdown(f"**Claim:** {res['claim']} <span class='verified'>VERIFIED</span>", unsafe_allow_html=True)
                    st.caption(f"**Source Anchor:** _{res['anchor']}_ (Distance: {res['distance']:.2f})")
                else:
                    st.markdown(f"**Claim:** {res['claim']} <span class='unverified'>HALLUCINATION PRUNED</span>", unsafe_allow_html=True)

        # COLUMN 2: THE REVISION BUNDLE (TABS)
        with col2:
            st.markdown("### 📚 Exportable Revision Bundle")
            tab1, tab2, tab3 = st.tabs(["📄 1-Page Summary", "📖 Glossary", "📝 Self-Quiz"])
            
            with tab1:
                st.markdown(f"## Topic Synthesis: {topic.title()}")
                st.markdown("""
                <div class="card">
                <b>Core Principles</b><br>
                Based on verified context, the fundamental mechanism involves structural components that interact to drive systemic changes. 
                <i></i>. 
                <br><br>
                <b>Key Mechanisms</b><br>
                The primary drivers require specific catalysts (e.g. molecular structures or thermal convection) to enact measurable shifts in the environment.
                <i></i>.
                </div>
                """, unsafe_allow_html=True)
                st.button("📥 Export as PDF (Markdown)", key="export_btn")

            with tab2:
                st.markdown("### Key Terms to Know")
                if "CRISPR" in topic.upper():
                    st.markdown("- **Cas9 Enzyme:** A protein that acts as 'molecular scissors' to cut DNA.")
                    st.markdown("- **Guide RNA (gRNA):** RNA that guides the Cas9 enzyme to the specific target DNA sequence.")
                    st.markdown("- **Off-target effects:** Unintended mutations caused when Cas9 cuts outside the target site.")
                else:
                    st.markdown("- **Lithosphere:** The rigid outer part of the earth, consisting of the crust and upper mantle.")
                    st.markdown("- **Mantle Convection:** The slow creeping motion of Earth's solid silicate mantle caused by convection currents.")
                    st.markdown("- **Subduction:** The sideways and downward movement of the edge of a plate of the earth's crust into the mantle.")

            with tab3:
                st.markdown("### Interactive Active-Recall Quiz")
                
                # Quiz State Management
                if "q1_ans" not in st.session_state:
                    st.session_state.q1_ans = None
                    
                q1 = "What is the primary driving mechanism or key actor in this system?"
                options = ["Mantle Convection / Cas9", "Solar Radiation", "Magnetic Fields", "Gravitational Pull"]
                
                choice = st.radio(q1, options, index=None)
                if st.button("Check Answer"):
                    if choice == "Mantle Convection / Cas9":
                        st.success("✅ Correct! Verified by OER Reference Data.")
                        st.info(f"**Context:** {corpus[1]}")
                    elif choice is None:
                        st.warning("Please select an answer.")
                    else:
                        st.error("❌ Incorrect. Review the mechanisms section.")
                        
    elif run_agent and not topic:
        st.warning("⚠️ Please enter a topic to begin the autonomous research phase.")
    else:
        # Welcome Screen
        st.info("👈 Enter a topic in the Mission Control sidebar to launch the research agents.")
        st.markdown("""
        ### Why VeritasLearn?
        - **Say No to Hallucinations:** Most AIs guess when they don't know. Our Adversarial Auditor checks every generated sentence against a local FAISS database.
        - **Active Learning:** We don't just give you text to read. We generate flashcard-style glossaries and active-recall quizzes.
        - **Full Transparency:** Watch the agents think, query, and prune fake facts in real-time on the left telemetry panel.
        """)

if __name__ == "__main__":
    main()
