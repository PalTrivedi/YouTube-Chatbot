# 🎥 YouTube-Chatbot — Chat With Any YouTube Video

**Drop in a YouTube link, and ask it questions like it's a document.**

Instead of watching an hour-long video to find the one part you care about, this app downloads the audio, transcribes it with Whisper, embeds it into a vector store, and lets you ask questions that get answered strictly from what was actually said in the video — with each video kept as its own persistent, revisitable chat thread.

🔗 **[Demo Video](https://www.youtube.com/watch?v=KJU-t2OfvoQ)**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Memory-1C3C3C)
![Whisper](https://img.shields.io/badge/Whisper-Transcription-412991?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-3776AB)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash--lite-4285F4?logo=googlegemini&logoColor=white)

---

## Why This Project

Video is a slow way to search for information — there's no Ctrl+F for something someone said 40 minutes into a talk. This project turns any YouTube video into a queryable knowledge base: it strips the audio, transcribes it locally with Whisper, chunks and embeds the transcript, and answers questions using only retrieved context from that transcript — so answers stay grounded in what was actually said, not the model's general knowledge.

Each video gets its own LangGraph thread (keyed by video ID), checkpointed to SQLite, so past transcripts and conversations persist and can be reopened from the sidebar instead of being reprocessed every time.

## How It Works

```
YouTube URL
     │
     ▼
yt-dlp — download audio
     │
     ▼
Whisper — transcribe audio to text
     │
     ▼
RecursiveCharacterTextSplitter — chunk transcript
     │
     ▼
Gemini Embeddings + FAISS — build a per-video vector store
     │
     ▼
User Question
     │
     ▼
FAISS retriever — top-k relevant transcript chunks
     │
     ▼
Gemini 2.5 Flash-Lite — "Answer ONLY from the transcript context"
     │
     ▼
Answer streamed to Streamlit chat UI
```

## Key Features

- **Video → knowledge base pipeline** — automatic audio download (`yt-dlp`), local transcription (`Whisper`), chunking, and embedding into a FAISS vector store, all triggered by pasting a URL
- **Context-grounded answers** — the LLM is explicitly prompted to answer only from retrieved transcript context, reducing hallucinated answers about video content
- **Per-video persistent threads** — each video's transcript, vector store, and conversation history are checkpointed to SQLite (`YouTubeChatbot.db`) under a thread ID derived from the video ID
- **Revisit past videos instantly** — previously processed videos appear in the sidebar; reopening one reloads its transcript, vector store, and full chat history without reprocessing
- **Portable vector stores** — FAISS indexes are serialized to raw bytes and stored directly in the LangGraph state, so no separate vector DB service is required

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph (`StateGraph`) |
| Conversation & vector store memory | LangGraph `SqliteSaver` checkpointer + SQLite |
| Audio download | `yt-dlp` |
| Transcription | OpenAI Whisper (local) |
| Embeddings | Gemini `text-embedding-004` |
| Vector search | FAISS |
| LLM | Google Gemini `2.5-flash-lite` via `langchain-google-genai` |

## Repo Structure

```
YouTube-Chatbot/
├── backend.py         # Video ID extraction, audio download, Whisper transcription,
│                       #  FAISS vector store, LangGraph QA graph, thread management
├── frontend.py         # Streamlit UI: URL input, sidebar of past videos, chat interface
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your system
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed and available on your system
- A [Google Gemini API key](https://makersuite.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/PalTrivedi/YouTube-Chatbot.git
cd YouTube-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure FFmpeg and yt-dlp paths

`backend.py` currently points at hardcoded local paths for `ffmpeg` and `yt-dlp`:

```python
FFMPEG_PATH = r"C:/Users/HP/Downloads/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"
YTDLP_PATH = r"D:/Test/venv/Scripts/yt-dlp.exe"
```

Update these to match where FFmpeg and yt-dlp are installed on your machine (or update the code to resolve them from your `PATH` instead).

### 4. Configure environment variables

Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Run

```bash
streamlit run frontend.py
```

The app opens in your browser (default `http://localhost:8501`). Paste a YouTube URL, click **Fetch Transcript**, and once processing finishes, ask questions about the video in the chat box. Previously processed videos are listed in the sidebar for instant reopening.

## Limitations & Future Work

- `FFMPEG_PATH` and `YTDLP_PATH` are hardcoded to a specific machine's file paths and need to be updated per environment
- Transcription runs synchronously in the Streamlit request, so processing long videos will block the UI
- Whisper's `base` model trades accuracy for speed; larger models would improve transcript quality at the cost of processing time
- No transcript caching keyed by video content changes — reprocessing the same video always redownloads and retranscribes
- Next steps: background/async transcript processing with a progress indicator, configurable Whisper model size, and auto-resolving FFmpeg/yt-dlp paths from the system `PATH`

## Security & Privacy

- Never commit your `.env` file or API keys to source control
- Downloaded audio is deleted after transcription, but transcripts and vector stores persist in `YouTubeChatbot.db` — treat that file as containing the content of every video you've processed

## License

MIT — feel free to fork and adapt.

## Author

Built by **Pal Trivedi**. Feedback and PRs welcome.
