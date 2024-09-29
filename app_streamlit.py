import streamlit as st
import requests

st.title('PDF Query Application')

# Session state to store access token
if 'access_token' not in st.session_state:
    st.session_state['access_token'] = None

# Embedding model selection
embedding_model = st.selectbox(
    'Select Embedding Model',
    ('voyage-law-2', 'sentence-transformers/all-mpnet-base-v2')
)

# LLM selection
llm_choice = st.selectbox(
    'Select Language Model',
    ('ollama', 'openai')
)

# File upload functionality
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    files = {'file': (uploaded_file.name, uploaded_file, 'application/pdf')}
    response = requests.post(f'http://127.0.0.1:5000/upload-pdf?embedding_model={embedding_model}', files=files)
    
    if response.status_code == 200:
        st.success('File successfully uploaded and processed.')
        response_data = response.json()
        st.write(response_data)
    else:
        st.error(f'Failed to process file: {response.text}')

# Login functionality
st.header('Login')
username = st.text_input("Username")
password = st.text_input("Password", type="password")
if st.button("Login"):
    response = requests.post('http://127.0.0.1:5000/login', json={'username': username, 'password': password})
    data = response.json()
    if response.status_code == 200:
        st.success('Login successful')
        if 'access_token' in data:
            st.session_state['access_token'] = data['access_token']
    else:
        st.error(data.get('error', 'Login failed'))

# Query functionality
query = st.text_input("Enter your query:")
if st.button("Query"):
    headers = {'Authorization': f"Bearer {st.session_state['access_token']}"}
    response = requests.post(f'http://127.0.0.1:5000/query', json={'query': query, 'llm': llm_choice, 'embedding_model': embedding_model}, headers=headers)
    if response.status_code == 200:
        st.write('Query results:')
        response_data = response.json()
        st.json(response_data)
    else:
        st.error(f'Failed to fetch query results: {response.text}')

    