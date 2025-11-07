from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
st.title("YouTube Chatbot")
video = st.text_input("Enter YouTube Video URL:")

if st.button("Input"):
    if video:
        with st.spinner("Fetching transcript..."):
            try:
                transcript_list = YouTubeTranscriptApi.fetch(video, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                if transcript:
                    st.write("Go ahead and ask questions about the video!")
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = splitter.create_documents([transcript])
                    if chunks:
                        gemini_embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/text-embedding-004"
                        )
                        vector_store = FAISS.from_documents(chunks, gemini_embeddings)
                        vector_store.index_to_docstore_id
                        retriever = vector_store.as_retriever(
                            search_type="similarity", search_kwargs={"k": 4}
                        )
                    else:
                        retriever = None
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash-lite",
                            temperature=0.7,
                            max_tokens=1024,
                        )
                    prompt = PromptTemplate(
                        template="""
                         You are a helpful assistant.
                         Answer ONLY from the provided transcript context.
                         If the context is insufficient, just say you don't know.

                         {context}
                         Question: {query}
                    """,
                        input_variables=["context", "query"],
                    )
                    query = st.text_input("Enter your question about the video:")
                    if st.button("Ask"):
                        if query:
                            with st.spinner("Getting answer..."):
                                retrieved_docs = retriever.invoke(query)
                                if retrieved_docs:
                                    context = "\n\n".join(
                                        doc.page_content for doc in retrieved_docs
                                    )
                                    final_prompt = prompt.invoke(
                                        {"context": context, "question": query}
                                    )
                                    answer = llm.invoke(final_prompt)
                                    st.write(answer.content)
                                else:
                                    st.write(
                                        "No relevant context found in the transcript."
                                    )
                        else:
                            st.write("Please enter a question.")

            except:
                st.write("Unable to retrieve transcript for this video!")
