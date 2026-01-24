"""Streamlit chat interface for TelecomPlus customer support."""

import streamlit as st

from main import answer

st.set_page_config(page_title="TelecomPlus Support", page_icon="📱")

st.title("📱 TelecomPlus - Support Client")
st.markdown("*Assistant intelligent pour répondre à vos questions*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in answer(prompt, stream=True):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # Display final response without cursor
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})