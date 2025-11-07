from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import re
import traceback

# Load environment variables
load_dotenv()

st.title("🎥 YouTube Chatbot")
# https://youtu.be/UUheH1seQuE?si=0V1oOrjAiPQz0hk_


# --- Helper: Extract Video ID ---
def extract_video_id(url: str):
    """Extracts the video ID from a YouTube URL (standard or short)."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


# --- Input URL ---
video_url = st.text_input("Enter YouTube Video URL:")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# --- Fetch Transcript ---
if st.button("Fetch Transcript"):
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please enter a valid one.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            api = YouTubeTranscriptApi()
            transcript_list = None

            try:
                transcript_list = api.fetch(video_id, languages=["en"])
                st.info("ℹ️ Using direct English transcript (via fetch).")
            except (NoTranscriptFound, TranscriptsDisabled, CouldNotRetrieveTranscript):
                try:
                    transcripts = api.list(video_id)
                    try:
                        transcript_obj = transcripts.find_manually_created_transcript(
                            ["en"]
                        )
                        st.info("ℹ️ Using manually-created English transcript.")
                    except NoTranscriptFound:
                        transcript_obj = transcripts.find_generated_transcript(["en"])
                        st.info("ℹ️ Using auto-generated English transcript.")

                    transcript_list = transcript_obj.fetch()
                except (
                    NoTranscriptFound,
                    TranscriptsDisabled,
                    CouldNotRetrieveTranscript,
                ):
                    st.error("⚠️ No English transcript available for this video.")
                    st.stop()

            if not transcript_list:
                st.warning("⚠️ Transcript is empty or not found.")
                st.stop()

            # Combine transcript text
            transcript = " ".join(chunk.text for chunk in transcript_list)
            st.success(f"✅ Transcript fetched! ({len(transcript)} characters)")

            # Split transcript
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.create_documents([transcript])
            # print(chunks)
            # Embeddings + Vector store
            gemini_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            vector_store = FAISS.from_documents(chunks, gemini_embeddings)
            st.session_state.retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )

            # LLM initialization
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                temperature=0.7,
                max_output_tokens=1024,
            )

            st.success("✅ Ready! Ask your questions below 👇")

        except Exception as e:
            st.error("❌ Error while fetching transcript.")
            st.json(
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "video_id": video_id,
                }
            )
            st.code(traceback.format_exc(), language="python")


# --- Ask Questions ---
if st.session_state.retriever and st.session_state.llm:
    query = st.text_input("💬 Ask a question about the video:")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            retriever = st.session_state.retriever
            llm = st.session_state.llm

            retrieved_docs = retriever.invoke(query)

            if not retrieved_docs:
                st.warning("⚠️ No relevant content found in transcript.")
                st.stop()

            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = PromptTemplate(
                template=(
                    "You are a helpful assistant.\n"
                    "Answer ONLY from the transcript context below.\n"
                    "Context:\n{context}\n\nQuestion: {question}"
                ),
                input_variables=["context", "question"],
            )

            formatted_prompt = prompt.format(context=context, question=query)
            answer = llm.invoke(formatted_prompt)

            st.write("### 💬 Answer:")
            st.write(answer.content)
else:
    st.info("ℹ️ Please fetch a transcript first before asking questions.")
