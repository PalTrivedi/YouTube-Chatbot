# frontend.py

import streamlit as st
from langchain_core.messages import HumanMessage
from backend import (
    create_thread_for_video,
    list_threads,
    load_conversation,
    chatbot,
)

st.set_page_config(layout="wide")

# ----------------------------
# INIT SESSION
# ----------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None
if "threads" not in st.session_state:
    st.session_state["threads"] = list_threads()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_url" not in st.session_state:
    st.session_state["current_url"] = ""


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("📁 Previous Videos")

# ➕ NEW TRANSCRIPT BUTTON
if st.sidebar.button("➕ New Transcript"):
    st.session_state["thread_id"] = None
    st.session_state["messages"] = []
    st.session_state["current_url"] = ""
    st.rerun()

# List existing threads
for t in st.session_state["threads"]:
    if st.sidebar.button(t, key=t):
        st.session_state["thread_id"] = t
        st.session_state["messages"] = load_conversation(t)
        st.session_state["current_url"] = f"https://youtu.be/{t}"
        st.rerun()


# ----------------------------
# UI – Main Title
# ----------------------------
st.title("🎥 YouTube Transcript Chatbot")

# If old chat has a URL, show it
url_placeholder = (
    st.session_state["current_url"] if st.session_state["current_url"] else ""
)

url = st.text_input("Enter YouTube URL:", value=url_placeholder)

# ----------------------------
# Fetch Transcript Button
# ----------------------------
if st.button("Fetch Transcript"):
    try:
        tid = create_thread_for_video(url)

        st.session_state["thread_id"] = tid
        st.session_state["threads"] = list_threads()
        st.session_state["messages"] = load_conversation(tid)
        st.session_state["current_url"] = url

        st.success("Transcript processed successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Transcript fetch failed: {e}")


# ----------------------------
# CHAT SECTION
# ----------------------------
if not st.session_state["thread_id"]:
    st.info("Fetch a transcript to start chatting.")
    st.stop()

# Show previous messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
user_msg = st.chat_input("Ask something about the video transcript...")

# ----------------------------
# Chatbot Response
# ----------------------------
if user_msg:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_msg)

    st.session_state["messages"].append({"role": "user", "content": user_msg})

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=CONFIG,
            )
            ai_msg = result["messages"][-1].content
            st.markdown(ai_msg)

    st.session_state["messages"].append({"role": "assistant", "content": ai_msg})
