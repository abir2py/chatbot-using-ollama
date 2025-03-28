import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Chatbot")

# initializing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("Act like an data scientist"))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


prompt = st.chat_input("How are you?")


if prompt:

    
    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))

    

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=2
    )

    result = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))