import streamlit as st
from agent import clone, local_clone, invoke

def run_agent():
    # Create a sidebar for inputs
    st.sidebar.title("Add to Codebase")
    project_name = st.sidebar.text_input("Enter Project Name", "")
    repo_url = st.sidebar.text_input("Enter GitHub repository URL", "")
    local_codebase_url = st.sidebar.text_input("Enter Local Codebase Path", "")
    model_name = st.sidebar.selectbox("Models", ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-0125"])

    # Add a Process button
    if st.sidebar.button("Add to Codebase"):
        if repo_url == '' and local_codebase_url != '':
            local_clone(project_name, local_codebase_url)
        elif local_codebase_url == '' and repo_url != '':
            clone(project_name, repo_url)

    # Add a New Chat button
    if st.sidebar.button("New Chat"):
        st.session_state.messages = []
        st.rerun()

    # Create a chatbot interface
    st.title("LP Code Agent")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Pass the entire chat history to the invoke function
            for response in invoke(st.session_state.messages, model_name):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# If this file is run directly, start the Streamlit app
if __name__ == "__main__":
    run_agent()