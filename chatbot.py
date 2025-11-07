from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import re
import traceback
from xml.etree.ElementTree import ParseError

# Load environment variables
load_dotenv()

st.title("🎥 YouTube Chatbot")

# --- Helper: Extract Video ID ---
def extract_video_id(url: str):
    """Extracts the video ID from a YouTube URL (standard or short)."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# --- Streamlit Input ---
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Fetch Transcript"):
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("❌ Invalid YouTube URL. Please enter a valid URL.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            st.write(f"🔍 Video ID: `{video_id}`")

            # --- Attempt to fetch transcript robustly ---
            transcript_list = None

            try:
                # First try to get any available transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except (NoTranscriptFound, TranscriptsDisabled, ParseError, CouldNotRetrieveTranscript):
                try:
                    # Fallback: list transcripts and choose manual or auto-generated
                    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    try:
                        transcript_obj = transcripts.find_manually_created_transcript(['en'])
                        st.info("ℹ️ Using manually created English transcript.")
                    except NoTranscriptFound:
                        transcript_obj = transcripts.find_generated_transcript(['en'])
                        st.info("ℹ️ Using auto-generated English transcript.")

                    transcript_list = transcript_obj.fetch()
                except (NoTranscriptFound, TranscriptsDisabled, ParseError, CouldNotRetrieveTranscript):
                    st.error("⚠️ No English transcript available for this video.")
                    st.info("Try a different video that has English captions enabled.")
                    st.stop()

            if not transcript_list or len(transcript_list) == 0:
                st.warning("⚠️ Transcript exists but is empty.")
                st.stop()

            # --- Combine transcript text ---
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            st.success(f"✅ Transcript fetched successfully! Length: {len(transcript)} characters.")
            st.info("You can now ask questions about the video.")

            # --- Split transcript into chunks ---
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            # --- Create embeddings & vector store ---
            gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = FAISS.from_documents(chunks, gemini_embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            # --- Initialize Gemini Chat Model ---
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                temperature=0.7,
                max_output_tokens=1024
            )

            # --- Ask Questions ---
            query = st.text_input("💬 Ask a question about the video:")
            if query and st.button("Ask"):
                with st.spinner("Thinking..."):
                    retrieved_docs = retriever.get_relevant_documents(query)
                    if not retrieved_docs:
                        st.warning("⚠️ No relevant context found in the transcript.")
                        st.stop()

                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                    prompt = PromptTemplate(
                        template=(
                            "You are a helpful assistant.\n"
                            "Answer ONLY from the transcript context.\n"
                            "If you don’t know, say you don’t know.\n\n"
                            "Context:\n{context}\n\nQuestion: {question}"
                        ),
                        input_variables=["context", "question"]
                    )

                    formatted_prompt = prompt.format(context=context, question=query)
                    answer = llm.invoke(formatted_prompt)

                    st.write("### 💬 Answer:")
                    st.write(answer.content)

        except Exception as e:
            st.error("❌ An unexpected error occurred while fetching the transcript.")
            st.json({
                "error_type": type(e).__name__,
                "error_message": str(e),
                "video_id": video_id
            })
            st.code(traceback.format_exc(), language="python")
