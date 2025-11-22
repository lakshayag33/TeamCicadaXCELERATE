# TeamCicadaXCELERATE

## Overview  
StudyMate is a project developed in Python (with HTML/CSS frontend) that provides a RAG-(Retrieval Augmented Generation) aware web app. It lets you query a knowledge base, perform LLM-driven inference, and deploy as a simple web service. It is a platform for user's to upload notes, ask questions and take quizzes. It will help the users in studying and preparing for exams.

## Features  
- Modular architecture: LLM client, RAG engine, web interface.  
- Web app entry point: `app.py`.  
- RAG engine in `rag_engine.py` handles retrieval + LLM query.  
- LLM client wrapper in `llm_client.py`.  
- Test scripts: `test.py`, `test_cuda.py`.  
- Static assets and templates in `static/` & `templates/`.  
- Uploads supported via `uploads/`.  
- Dependencies specified in `requirements.txt`.

## Getting Started

### Prerequisites  
- Python 3.8+ (or later)  
- Virtual environment tool (recommended: `venv` or `conda`)  
- (Optional) GPU / CUDA support for high-performance inference (if you plan to run heavy LLMs)  

### Installation  
# Clone the repo
git clone https://github.com/lakshayag33/TeamCicadaXCELERATE.git
cd TeamCicadaXCELERATE

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
