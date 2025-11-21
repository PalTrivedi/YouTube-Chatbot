# YouTube-Chatbot

YouTube-Chatbot is a small Retrieval-Augmented Generation (RAG) demo that lets you ask questions about YouTube content and get concise, conversational answers. The project includes a simple backend for processing and fetching data and a Streamlit-based frontend for interacting with the chatbot.

**This repository is a minimal example — customize and secure it before using in production.**

**Features**

- **RAG-style answers:** Uses retrieved YouTube content as context for generated responses.
- **Streamlit UI:** Lightweight frontend for quick interaction and testing.
- **Pluggable backends:** `backend.py` holds the data/retrieval logic; adapt it to your APIs or vector store.

**Requirements**

- Python 3.10+ (recommended)
- See `requirements.txt` for Python dependencies

**Quickstart**

1. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your keys (example below).

4. Start the backend (if applicable):

   ```powershell
   python backend.py
   ```

5. Start the frontend UI:

   ```powershell
   streamlit run frontend.py
   ```

6. Open the Streamlit app in your browser (usually `http://localhost:8501`).

**Environment variables (.env example)**
Do not commit secrets to source control. Example `.env` contents:

```
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
# Optional: configure other keys or endpoints used by backend.py
# VECTOR_STORE_PATH=./data/vectors
```

Replace the variable names and values with those expected by your copy of `backend.py` and `frontend.py`.

**Project structure**

- `backend.py` : Retrieval, data fetching, or API logic.
- `frontend.py`: Streamlit UI for the chatbot.
- `requirements.txt`: Python package dependencies.
- `.env`: Local environment variables (gitignored).

**Usage**

- Use the Streamlit UI to enter a YouTube link, paste text, or ask a question (depends on how `backend.py` is implemented).
- The app will retrieve relevant context and generate an answer using the configured LLM.

**Troubleshooting**

- If `streamlit run frontend.py` fails, ensure Streamlit is installed and your virtual environment is active.
- If you see errors about missing API keys, confirm that `.env` contains the expected variables and that your code reads them (e.g., via `python-dotenv`).
- If ports are in use, specify an alternate Streamlit port: `streamlit run frontend.py --server.port 8502`.

**Security & privacy**

- Never commit `.env` or any secret keys to the repository.
- Consider adding rate limits and input sanitization before exposing the app publicly.

**Contributing**

- Fork the repo, make changes on a feature branch, and open a pull request. Describe the change and include testing steps.

**License**

- No license specified

--- Pal Trivedi
