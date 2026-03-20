# Plan: Project Description (Read-Only Analysis)

## Context
User requested a thorough description of the demo repo. This was a research/documentation task, not an implementation task. No code changes needed.

## Result
Full description delivered in conversation covering all requested aspects:
- Architecture (two-tier: React SPA + FastAPI, file-based persistence)
- Tech stack (React 19, Vite 7, FastAPI, llama-cpp-python, pytesseract, PyMuPDF)
- Folder structure
- Entry points (uvicorn main:app, npm run dev)
- Dependencies (requirements.txt + package.json)
- API surface (7 endpoints documented with request/response shapes)
- Database schema (flat JSON/JSONL files: document registry, chunks JSONL, chat history JSON)
- Environment variables (full table with defaults)
- Test coverage (1 test file, backend only, covers chat history + retrieval modes)
- Build & deployment (local-only, no Docker/CI/CD)

The full description is above. Key takeaways:

  - What it is: A local-first PDF Q&A app — upload a PDF, extract text (with OCR fallback), chunk it for RAG,   and chat with it using a local GGUF model (Gemma-2 9B)                                                    
  - No database: everything is flat JSON/JSONL files on disk                                                 
  - No infrastructure: no Docker, no CI/CD, no reverse proxy — purely local dev                              
  - Test gap: only chat history + retrieval are tested; PDF extraction, chunking, and all frontend code are  
  untested                                                                                                     - The current test subject is Animal Farm by Orwell (the .chunks.jsonl file open in your IDE)              
  
