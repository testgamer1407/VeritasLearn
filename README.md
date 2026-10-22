# VeritasLearn
VeritasLearn 🎓
Link - https://veritas-learn.streamlit.app/
Autonomous Agentic Research & Revision Engine

VeritasLearn is a smart study engine designed to solve the problem of AI hallucinations in education. It uses a multi-agent architecture to research academic topics, fact-check its own findings against verified sources, and instantly generate hallucination-free study guides, glossaries, and interactive quizzes.

1. Project Setup and Installation Instructions

To set up VeritasLearn on your local machine, follow these steps:

Prerequisites:

Python 3.8 or higher installed on your system.

Git (optional, for cloning the repository).

Step-by-step Installation:

Clone or Download the Repository:

git clone <your-repository-url>
cd veritaslearn


Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


Install Dependencies:

pip install -r requirements.txt


2. Description of Each File and its Purpose

app.py: The main application code. This contains the Streamlit user interface, the multi-agent logic (Planner, Reader, Fact-Checker, Builder), dynamic data fetching from Wikipedia, and the FAISS vector database integration.

requirements.txt: Contains the list of all Python libraries and dependencies required to run the application securely (optimized for CPU to prevent memory crashes).

README.md: This documentation file, explaining project setup, architecture, and usage.

3. Instructions for Running the Main Code

Once your dependencies are installed, you can launch the application locally using Streamlit.

Open your terminal or command prompt.

Navigate to the folder containing app.py.

Run the following command:

streamlit run app.py


A new tab will automatically open in your default web browser (usually at http://localhost:8501).

Usage: Enter any academic topic (e.g., "Photosynthesis", "Black Holes") in the sidebar, adjust the fact-check strictness if desired, and click "Create My Study Kit".

4. List of Dependencies or Required Libraries

The project relies on the following core libraries (defined in requirements.txt):

streamlit: Used to build the interactive web dashboard and user interface.

sentence-transformers: Used to load the lightweight NLP models for generating text embeddings.

faiss-cpu: Facebook AI Similarity Search. Used to create the local vector database for fast, semantic similarity searches (Dense Retrieval).

torch: The underlying machine learning framework required by sentence-transformers (configured for CPU usage to ensure lightweight hosting).

numpy: Used for numerical operations and array formatting during vector creation.

(Note: Built-in Python libraries like urllib, json, re, and time are also utilized but do not require external installation).

5. AI Tools Used to Create Models or Components

VeritasLearn leverages a combination of AI techniques and open-source models:

Embedding Model (all-MiniLM-L6-v2): We use this lightweight, open-source Hugging Face model via the sentence-transformers library. It converts academic text into dense vector embeddings for our retrieval system.

Vector Search Engine (FAISS): We utilized Meta's FAISS (Facebook AI Similarity Search) to index the chunked text and perform fast inner-product (Cosine Similarity) searches. This powers the "Dense Retriever" and "Fact-Checker" agents.

Dynamic Knowledge Fetching (Wikipedia REST API): Instead of relying on a pre-trained LLM's static memory, the system fetches live, verified academic passages from Wikipedia to use as its "Ground Truth" corpus.

Multi-Agent Architecture Workflow:

Curriculum Architect: Deconstructs topics using programmatic AI logic.

Adversarial Auditor: A custom AI evaluation step that uses Cosine Similarity thresholds to detect and prune AI hallucinations before they reach the user.
