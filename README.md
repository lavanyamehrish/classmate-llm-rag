CLASSMATE LLM (RAG-Based Document Assistant)

CLASSMATE LLM is an AI-powered document assistant that enables users to upload PDF or CSV files and interact with them through summarization, question generation, and contextual question answering using a Retrieval-Augmented Generation (RAG) pipeline.

Features
Upload and process PDF or CSV files
Automatic document summarization
Bullet point extraction
Question generation from content
Context-aware question answering
Semantic search using embeddings
Tech Stack
Frontend: Streamlit
Models:
BART (summarization)
FLAN-T5 (text generation)
MiniLM (embeddings)
Core Concepts:
Retrieval-Augmented Generation (RAG)
Cosine similarity for retrieval
System Overview
User Input → Text Extraction → Chunking → Embeddings → Retrieval → Generation → Output

Parallel pipeline:

Text → Summarization → Bullet Points → Questions
Installation
1. Clone the repository
git clone https://github.com/your-username/classmate-llm-rag.git
cd classmate-llm-rag
2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install \
streamlit==1.28.0 \
numpy==1.24.4 \
pandas==1.5.3 \
pymupdf \
sentence-transformers \
transformers \
torch \
scikit-learn \
pyarrow==12.0.1 \
altair==4.2.2 \
cachetools==5.3.1 \
importlib-metadata==6.8.0 \
packaging==23.2 \
pillow==10.4.0 \
protobuf==4.25.3 \
rich==13.7.1 \
gitpython \
pydeck==0.8.1b0 \
tenacity==8.2.3 \
tzlocal==5.2 \
validators
Running the Application
streamlit run app_ui.py

Open:

http://localhost:8501
Applications
Study and revision assistance
Research document summarization
Structured data understanding
Knowledge extraction from documents
Future Work
Integration with vector databases
Support for additional file formats
Cloud deployment
User authentication
