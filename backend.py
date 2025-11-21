# backend.py

import os
import re
import subprocess
import whisper
import sqlite3
import pickle
import tempfile
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ----------------------------
# PATH CONFIG
# ----------------------------
FFMPEG_PATH = r"C:/Users/HP/Downloads/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"
YTDLP_PATH = r"D:/Test/venv/Scripts/yt-dlp.exe"

# ----------------------------
# SQLITE CHECKPOINTING
# ----------------------------
conn = sqlite3.connect("YouTubeChatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)


# ----------------------------
# LANGGRAPH STATE
# ----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    transcript: str
    vector_store: dict  # contains index.faiss + index.pkl bytes


# ----------------------------
# YT VIDEO ID EXTRACT
# ----------------------------
def extract_video_id(url: str):
    # Match normal YT URLs: ?v=
    match = re.search(r"(?:v=)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # Match youtu.be short links
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # Match YouTube Shorts URLs
    match = re.search(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    return None


# ----------------------------
# STEP 1: Download Audio
# ----------------------------
def download_audio(url: str):
    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}")

    if not os.path.isfile(YTDLP_PATH):
        raise FileNotFoundError(f"yt-dlp.exe not found at {YTDLP_PATH}")

    output_file = "audio.m4a"

    subprocess.run(
        [YTDLP_PATH, "-f", "bestaudio", "-o", output_file, url],
        check=True,
    )

    return output_file


# ----------------------------
# STEP 2: Whisper Transcription
# ----------------------------
def transcribe_audio(path: str):
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]


# ----------------------------
# STEP 3: Create vector store safely
# ----------------------------
def create_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector = FAISS.from_documents(chunks, emb)

    temp_dir = tempfile.mkdtemp()
    vector.save_local(temp_dir)

    with open(f"{temp_dir}/index.faiss", "rb") as f:
        idx_bytes = f.read()
    with open(f"{temp_dir}/index.pkl", "rb") as f:
        store_bytes = f.read()

    return {"index": idx_bytes, "store": store_bytes}


# ----------------------------
# STEP 4: Load FAISS safely from bytes
# ----------------------------
def load_vector_store(serialized: dict):
    temp_dir = tempfile.mkdtemp()

    with open(f"{temp_dir}/index.faiss", "wb") as f:
        f.write(serialized["index"])
    with open(f"{temp_dir}/index.pkl", "wb") as f:
        f.write(serialized["store"])

    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector = FAISS.load_local(temp_dir, emb, allow_dangerous_deserialization=True)

    return vector


# ----------------------------
# LLM SETUP
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template=(
        "You are a YouTube transcript assistant.\n"
        "Answer ONLY from the transcript context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"],
)


# ----------------------------
# QA NODE
# ----------------------------
def qa_node(state: ChatState):
    if not state.get("messages"):
        return {}

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return {}

    query = last_msg.content

    vector = load_vector_store(state["vector_store"])
    retriever = vector.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)
    if not docs:
        response = llm.invoke("I cannot find the answer in the transcript.")
        return {"messages": [response]}

    context = "\n\n".join(d.page_content for d in docs)

    final_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(final_prompt)

    return {"messages": [answer]}


# ----------------------------
# BUILD LANGGRAPH APP
# ----------------------------
graph = StateGraph(ChatState)
graph.add_node("qa_node", qa_node)
graph.add_edge(START, "qa_node")
graph.add_edge("qa_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ----------------------------
# CREATE THREAD FOR VIDEO
# ----------------------------
def create_thread_for_video(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    audio = download_audio(url)
    transcript = transcribe_audio(audio)

    if os.path.exists(audio):
        os.remove(audio)

    vector_store = create_vector_store(transcript)

    initial_state = {
        "messages": [],
        "transcript": transcript,
        "vector_store": vector_store,
    }

    chatbot.update_state(
        config={"configurable": {"thread_id": video_id}},
        values=initial_state,
    )

    return video_id


# ----------------------------
# LIST THREADS
# ----------------------------
def list_threads():
    threads = set()
    for cp in checkpointer.list(None):
        cfg = cp.config.get("configurable", {})
        tid = cfg.get("thread_id")
        if tid:
            threads.add(tid)
    return list(threads)


# ----------------------------
# LOAD CONVERSATION
# ----------------------------
def load_conversation(thread_id: str):
    snapshot = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    msgs = snapshot.values.get("messages", [])
    out = []

    for m in msgs:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        else:
            out.append({"role": "assistant", "content": m.content})

    return out
