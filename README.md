# ComplyAI – Compliance Document Assistant

ComplyAI is a learning-focused Retrieval-Augmented Generation (RAG) prototype. It lets a user upload text-based PDF documents, indexes their contents with sentence embeddings and FAISS, retrieves relevant passages, and asks Gemini to answer using those passages with source references.

## Features

- Multiple PDF uploads
- PDF text extraction with page metadata
- Overlapping text chunking
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- FAISS inner-product search over normalized vectors (cosine similarity)
- Gemini-powered grounded question answering
- Retrieved source excerpts with file names and page numbers
- Basic safeguards for questions unsupported by retrieved context

## Architecture

```text
PDF upload
   |
   v
PyMuPDF text extraction (page metadata)
   |
   v
Chunking with overlap
   |
   v
Sentence Transformers embeddings
   |
   v
FAISS vector index
   |
User question -> query embedding -> top-k retrieval
                                   |
                                   v
                         Context + question prompt
                                   |
                                   v
                              Gemini API
                                   |
                                   v
                      Answer + source references
```

## Run locally

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/complyai.git
cd complyai
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your Gemini API key in the app sidebar. Create a key through Google AI Studio: https://aistudio.google.com/app/apikey

## Use

1. Start the Streamlit application.
2. Upload one or more text-based PDF policies or reports.
3. Select **Process documents**.
4. Enter a question in the chat box.
5. Review the answer and expand retrieved sources to inspect the supporting passages.

## Important limitations

- This is a prototype, not a certified compliance product or legal-advice system.
- Scanned/image-only PDFs are not OCR-processed.
- The FAISS index and uploaded document content remain in application memory and are lost when the app restarts.
- The prototype does not implement user authentication, access control, persistent storage, encryption at rest, audit logging, tenant isolation, or production monitoring.
- Retrieval scores are similarity scores, not calibrated confidence probabilities.
- The answer generator may still make mistakes. Verify important decisions against original documents.
- The Gemini model name may need to be changed if it is unavailable for your account or region.

## Security

- Never commit API keys, secrets, private customer documents, or confidential evidence.
- Do not upload real client or sensitive company documents to this demo.
- Use synthetic or public sample documents for demonstrations.
- Review provider data-handling terms before using any external model with sensitive information.

## Suggested next improvements

- Add OCR for scanned PDFs.
- Persist and version the vector index.
- Add document deletion and per-user access control.
- Evaluate retrieval with a labeled question/answer dataset and Recall@K.
- Evaluate answer faithfulness and relevance with human-reviewed test cases.
- Add automated tests, Docker support, and a FastAPI service.
