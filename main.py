import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import re
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ClaimPrism | AI Fact Checker", page_icon="🔍", layout="wide")

# --- 1. MODEL LOADING & CACHING ---
@st.cache_resource
def load_models():
    """Load lightweight NLP models for embeddings and NLI."""
    # Sentence embedder for Vector Search (Dense Retrieval)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # NLI Model to check Entailment vs Contradiction
    # We use a RoBERTa model fine-tuned on MNLI (Multi-Genre NLI)
    nli_model = pipeline("text-classification", model="roberta-large-mnli", top_k=None)
    
    return embedder, nli_model

# --- 2. VECTOR DATABASE SETUP ---
@st.cache_resource
def setup_vector_store():
    """Initialize an in-memory ChromaDB and load a mock Ground Truth corpus."""
    client = chromadb.Client()
    collection = client.create_collection(name="verified_ground_truth")
    
    # Mock "Verified/True" articles (e.g., from PolitiFact/Reuters/Kaggle True splits)
    ground_truth_corpus = [
        "NASA's Curiosity rover successfully landed on Mars and drilled into the Gale Crater to analyze soil samples. While it found organic molecules and evidence of ancient water, NASA has not discovered any bacterial or alien life on Mars.",
        "Microwave ovens use non-ionizing electromagnetic radiation to heat food safely. The World Health Organization confirms they do not make food radioactive and do not cause cancer. Claims that Japan is banning microwaves originated from a Russian satirical website in 2019 and are completely false.",
        "The Covid-19 vaccines underwent rigorous clinical trials before public distribution. They do not contain microchips and do not alter human DNA. The mRNA technology teaches cells how to make a protein to trigger an immune response.",
        "Global warming is a documented phenomenon supported by overwhelming scientific consensus. Ice core samples show unprecedented CO2 levels.",
    ]
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(ground_truth_corpus).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=ground_truth_corpus,
        metadatas=[{"source": "NASA Archives"}, {"source": "Snopes & WHO Fact Check"}, {"source": "CDC Verified Data"}, {"source": "NOAA Climate Data"}],
        ids=["doc1", "doc2", "doc3", "doc4"]
    )
    return collection

# --- 3. HELPER FUNCTIONS ---
def decompose_claim(text):
    """Simple regex-based atomic claim extractor (splits compound sentences)."""
    # In a full production app, you would use an LLM or dependency parser here.
    # For this prototype, we split by periods, and basic conjunctions to simulate atomic facts.
    clauses = re.split(r'\.|\b and \b|\b but \b', text)
    return [c.strip() for c in clauses if len(c.strip()) > 10]

def calculate_entropy(probs):
    """Calculates Shannon Entropy to measure AI Uncertainty."""
    return -sum([p * math.log2(p) for p in probs if p > 0])

# --- 4. CORE PIPELINE ---
def process_claim(user_input, embedder, nli_model, collection):
    atomic_claims = decompose_claim(user_input)
    results = []
    
    for claim in atomic_claims:
        # Step A: Vector Retrieval (Find top verified context)
        claim_embedding = embedder.encode([claim]).tolist()
        search_res = collection.query(query_embeddings=claim_embedding, n_results=1)
        
        retrieved_context = search_res['documents'][0][0]
        source = search_res['metadatas'][0][0]['source']
        distance = search_res['distances'][0][0] # Lower distance = higher semantic similarity
        
        # Step B: NLI Verification
        # Format: Premise = Ground Truth, Hypothesis = User's Atomic Claim
        nli_input = f"{retrieved_context} </s></s> {claim}"
        nli_output = nli_model(nli_input)[0]
        
        # Map RoBERTa outputs to probabilities
        probs = {res['label']: res['score'] for res in nli_output}
        # roberta-large-mnli labels: 'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'
        p_entail = probs.get('ENTAILMENT', 0.0)
        p_contra = probs.get('CONTRADICTION', 0.0)
        p_neutral = probs.get('NEUTRAL', 0.0)
        
        # Step C: Status & Uncertainty Logic
        entropy = calculate_entropy([p_entail, p_contra, p_neutral])
        
        if distance > 1.2 or p_neutral > 0.6: # distance threshold for ChromaDB cosine/L2
            status = "Unverified"
            color = "orange"
        elif p_entail > p_contra:
            status = "Verified"
            color = "green"
        else:
            status = "Fabricated / False"
            color = "red"
            
        results.append({
            "claim": claim,
            "status": status,
            "color": color,
            "context": retrieved_context,
            "source": source,
            "p_entail": p_entail,
            "p_contra": p_contra,
            "entropy": entropy
        })
        
    return results

# --- 5. STREAMLIT DASHBOARD UI ---
def main():
    st.title("prism ClaimPrism")
    st.subheader("AI Spectral Truth Analyzer & Deconstruction Dashboard")
    st.markdown("---")
    
    with st.spinner("Initializing Models & Ground Truth Knowledge Base..."):
        embedder, nli_model = load_models()
        collection = setup_vector_store()
        
    # Input Section
    st.markdown("### 1. Ingest Claim")
    
    # Preset examples for easy demo
    example_claims = [
        "NASA's Curiosity rover drilled into Gale crater and discovered alien bacterial life on Mars.",
        "Japan bans microwave ovens by year-end after scientists find lethal radiation levels causing cancer in citizens.",
        "The Covid-19 vaccines contain microchips tracking technology."
    ]
    
    selected_example = st.selectbox("Select a benchmark claim or type your own:", ["-- Type your own --"] + example_claims)
    
    user_text = ""
    if selected_example != "-- Type your own --":
        user_text = st.text_area("Input Text (Headline/Excerpt):", value=selected_example, height=100)
    else:
        user_text = st.text_area("Input Text (Headline/Excerpt):", placeholder="Paste a viral claim or news headline here...", height=100)
        
    if st.button("Analyze Claim via ClaimPrism", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a claim to analyze.")
            return
            
        with st.spinner("Refracting claim into atomic facts & checking vector base..."):
            results = process_claim(user_text, embedder, nli_model, collection)
            
        st.markdown("---")
        st.markdown("### 2. Spectral Breakdown Panel")
        
        # Calculate aggregate scores
        total_entailment = sum([r['p_entail'] for r in results]) / len(results)
        total_contra = sum([r['p_contra'] for r in results]) / len(results)
        avg_entropy = sum([r['entropy'] for r in results]) / len(results)
        
        overall_status = "Potentially Misleading (Mixed Facts)"
        if total_entailment > 0.75 and total_contra < 0.2:
            overall_status = "Highly Verified"
        elif total_contra > 0.5:
            overall_status = "Fabricated / False"
            
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall System Verdict", overall_status)
        col2.metric("Evidentiary Support Score", f"{total_entailment*100:.1f}%")
        col3.metric("System Uncertainty (Entropy)", f"{avg_entropy:.2f} bits", help="High entropy means the AI is unsure due to lack of corpus data.")
        
        st.markdown("---")
        st.markdown("### 3. The Evidence Ledger (Atomic Justification)")
        
        for idx, res in enumerate(results):
            with st.expander(f"Fact {idx+1}: {res['claim']} - [{res['status'].upper()}]", expanded=True):
                st.markdown(f"**Atomic Claim:** `{res['claim']}`")
                
                # Dynamic color bar for probabilities
                st.progress(res['p_entail'], text=f"Entailment (True): {res['p_entail']*100:.1f}%")
                st.progress(res['p_contra'], text=f"Contradiction (False): {res['p_contra']*100:.1f}%")
                
                st.info(f"**Retrieved Ground Truth ({res['source']}):**\n\n> {res['context']}")

if __name__ == "__main__":
    main()
