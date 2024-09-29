from flask import Flask, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import os
import pdfplumber
from pymongo import MongoClient, ASCENDING
from flask_cors import CORS
import logging
import numpy as np
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store.simple_chat_store import SimpleChatStore
import torch
import faiss
from tenacity import retry, wait_exponential, stop_after_attempt
import jwt
import datetime
from functools import wraps
import hashlib
from bson import ObjectId

torch.cuda.empty_cache()

# Set up Settings
from llama_index.core import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['SECRET_KEY'] = 'your_secret_key'

# MongoDB setup
mongo_conn_url = os.getenv("MONGO_CONN_URL", "mongodb://localhost:27017/")
client = MongoClient(mongo_conn_url)
db = client['pdf_query_db']
chunks_collection = db['chunks']
discussions_collection = db['discussions']
chats_collection = db['chats']
users_collection = db['users']

# Create an index to prevent duplicate chunks
chunks_collection.create_index([('filename', ASCENDING), ('page_number', ASCENDING), ('text', ASCENDING)], unique=True)

FAISS_INDEX_PATH = './faiss_index.index'

def initialize_embedding_model(model_name, voyage_api_key):
    embed_model = VoyageEmbedding(model_name=model_name, voyage_api_key=voyage_api_key)
    sample_text = "Sample text to determine embedding dimension"
    sample_embedding = embed_model.get_text_embedding(sample_text)
    embedding_dimension = len(sample_embedding)
    logging.info(f"Embedding dimension is {embedding_dimension}")
    return embed_model, embedding_dimension

# Initialize embedding models
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "")
voyage_embed_model, voyage_embedding_dimension = initialize_embedding_model("voyage-law-2", voyage_api_key)
Settings.embed_model = voyage_embed_model

# Initialize LLMs
ollama_llm = Ollama(
    model="llama3.1",
    max_length=4096,
    temperature=0.7,
    top_p=0.9,
    device_map="auto",
    server_url="http://localhost:11434",
    request_timeout=4600.0,
)

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_api_key:
    raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY environment variable.")
openai_llm = OpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
)

# FAISS index setup for fast nearest neighbor search
dimension = voyage_embedding_dimension  # Dimension of the embeddings from VoyageEmbedding
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    logging.info("Loaded existing FAISS index")
    if faiss_index.d != dimension:
        raise ValueError(f"Loaded FAISS index has dimension {faiss_index.d}, but expected {dimension}.")
except (Exception, ValueError) as e:
    logging.info(f"Creating new FAISS index: {e}")
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

def save_faiss_index():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

def embed(text):
    if not text.strip():
        logging.warning("Attempted to embed an empty string")
        return []
    embedding = Settings.embed_model.get_text_embedding(text)
    logging.debug(f"Embedding for text '{text[:50]}...' is {embedding[:5]}...")  # Log first 5 embedding values for brevity
    return embedding

def preprocess_text(text):
    """Normalize and preprocess text for consistent duplicate checking."""
    text = text.replace('\n', ' ').replace('\r', '').strip()
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def split_text_to_chunks(text, chunk_size=1000, chunk_overlap=200, page_number=None):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_chunk) > chunk_size:
            chunks.append({'text': ' '.join(current_chunk), 'page_number': page_number})
            current_chunk = []
            current_length = 0
        
        current_chunk.append(word);
        current_length += len(word);

    if current_chunk:
        chunks.append({'text': ' '.join(current_chunk), 'page_number': page_number})

    return chunks

# Define your custom prompt template
class MongoDBContextPrompt(PromptTemplate):
    def generate_prompt(self, query: str, documents: list, chat_history: str) -> str:
        context = "\n".join([
            f"Document: {doc.get('filename', 'unknown')}, Page: {doc.get('page_number', 'unknown')}, "
            f"Text: {doc.get('text', '')[:500]}..."
            for doc in documents
        ])
        max_context_length = 4096 - len(query) - len(chat_history) - 200  # Allow more space for detailed instructions
        if len(context) > max_context_length:
            context = context[:max_context_length] + '...'
        logging.debug(f"Generated prompt context for query: {query}")
        return (
            f"Please answer the following question using the context provided from multiple documents. "
            f"Use the language of the document where the response is retrieved. "
            f"Give a complete and informative response. "
            f"Chat history:\n{chat_history}\n\n"
            f"This query relates to the following documents: {context}\n\n"
            f"Question: {query}"
        )

class CustomRetrieverQueryEngine(RetrieverQueryEngine):
    def __init__(self, llm, prompt_template=None):
        self.prompt_template = prompt_template
        self.llm = llm

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def query(self, query_text, discussion_id=None, selected_document=None, chat_history=None):
        if chat_history is None:
            chat_history = ""
        # Load documents from MongoDB, filter by selected document if provided
        query_filter = {'filename': selected_document} if selected_document else {}
        documents = list(chunks_collection.find(query_filter))
        if not documents:
            logging.error("No documents found in the database.")
            return {"response": "No documents found in the database.", "sources": []}

        logging.info(f"Loaded {len(documents)} documents from the database")

        # Query embedding
        query_embedding = np.array(embed(query_text)).reshape(1, -1)
        if query_embedding.size == 0:
            logging.error("Failed to generate embedding for the query.")
            return {"response": "Failed to generate embedding for the query.", "sources": []}

        # Perform similarity search using FAISS
        k = min(60, len(documents))  # Ensure k is not larger than the number of documents
        D, I = faiss_index.search(query_embedding, k)
        logging.info(f"Similarity scores for the query: {D[0][:10]}")  # Log the first 10 similarity scores

        # Adjust the similarity threshold and filtering logic
        similarity_threshold = 1.2  # Increased threshold based on observed scores
        sorted_docs = sorted(zip(D[0], I[0]), key=lambda x: x[0])
        response_docs = [documents[i] for d, i in sorted_docs if i != -1 and d < similarity_threshold]

        if not response_docs:
            logging.warning(f"No documents found within the similarity threshold of {similarity_threshold}. Using top 10 documents.")
            response_docs = [documents[i] for d, i in sorted_docs[:10] if i != -1]

        logging.info(f"Retrieved {len(response_docs)} relevant documents")

        # Use only relevant context from top matches
        top_docs = response_docs[:60]  # Limit to top 60 documents

        if self.prompt_template:
            query_with_context = self.prompt_template.generate_prompt(query=query_text, documents=top_docs, chat_history=chat_history)
        else:
            query_with_context = query_text

        response = self.llm.complete(query_with_context)
        response_text = str(response)

        # Logging for better debugging
        logging.info(f"Generated response: {response_text[:200]}")  # Log the first 200 characters of the response for clarity

        return {"response": response_text, "sources": [{"filename": doc['filename'], "page_number": doc['page_number'], "text": doc['text']} for doc in top_docs]}

# Initialize vector store and document store
docstore = MongoDocumentStore.from_uri(mongo_conn_url)
index_store = MongoIndexStore.from_uri(mongo_conn_url)
storage_context = StorageContext.from_defaults(docstore=docstore, index_store=index_store)

# Create the index from documents
documents = list(chunks_collection.find({}))
llama_documents = [Document(text=doc['text'], extra_info={"filename": doc["filename"], "page_number": doc["page_number"]}) for doc in documents]
index = VectorStoreIndex.from_documents(llama_documents, storage_context=storage_context, embed_model=Settings.embed_model)
index.set_index_id("my_index")
index.storage_context.persist(persist_dir="./storage")

# Load or create indices
index = load_index_from_storage(storage_context, index_id="my_index")
retriever = index.as_retriever(similarity_top_k=60)

# Create an instance of your custom engine
custom_prompt_template = MongoDBContextPrompt(template="{context}\n\n{query}")

# Initialize chat memory
chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, token_limit=4096)
query_count = 0
MAX_QUERIES_BEFORE_CLEAR = 10

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    hashed_password = hashlib.sha256(data['password'].encode()).hexdigest()
    user_id = users_collection.insert_one({
        'username': data['username'],
        'email': data['email'],
        'password': hashed_password
    }).inserted_id
    return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password are required'}), 400

    user = users_collection.find_one({'username': data['username']})
    if not user or user['password'] != hashlib.sha256(data['password'].encode()).hexdigest():
        return jsonify({'error': 'Invalid credentials'}), 401

    token = jwt.encode({
        'user_id': str(user['_id']),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token}), 200

@app.route('/new_chat', methods=['POST'])
def new_chat():
    discussion_id = discussions_collection.insert_one({
        'name': "New Discussion",
        'created_at': datetime.datetime.utcnow()
    }).inserted_id
    return jsonify({'message': 'New discussion created', 'discussion_id': str(discussion_id)}), 201

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    discussion_id = request.args.get('discussion_id')

    if discussion_id:
        # Fetch chat history for the selected discussion
        chats = list(chats_collection.find({'discussion_id': ObjectId(discussion_id)}).sort('timestamp', ASCENDING))

        # Convert all ObjectId fields to strings
        for chat in chats:
            chat['_id'] = str(chat['_id'])  # Convert ObjectId to string
            chat['discussion_id'] = str(chat['discussion_id'])  # Convert discussion_id to string
            if 'parent_message_id' in chat:
                chat['parent_message_id'] = str(chat['parent_message_id'])  # Convert parent_message_id to string if present

        return jsonify({'chat_history': chats}), 200

    # If no discussion_id is provided, return the list of all discussions
    discussions = list(discussions_collection.find({}))

    # Convert ObjectId fields to strings in discussions as well
    for discussion in discussions:
        discussion['_id'] = str(discussion['_id'])  # Convert ObjectId to string

    return jsonify({'discussions': [{'id': str(discussion['_id']), 'name': discussion.get('name', 'Unnamed Discussion')} for discussion in discussions]}), 200

@app.route('/edit_message', methods=['POST'])
def edit_message():
    data = request.get_json()
    message_id = data.get('message_id')
    new_query = data.get('new_query')
    llm_choice = data.get('llm', 'ollama')
    discussion_id = data.get('discussion_id')

    if not message_id or not new_query:
        return jsonify({'error': 'Message ID and new query are required'}), 400

    try:
        # Ensure the message_id is an ObjectId
        message_id = ObjectId(message_id)

        # Delete previous assistant response
        result = chats_collection.delete_many({
            'discussion_id': ObjectId(discussion_id),
            'role': 'assistant',
            'parent_message_id': message_id
        })

        # Check if the deletion was successful
        if result.deleted_count == 0:
            logging.info("No previous assistant response found to delete.")

        # Update the user's query
        chats_collection.update_one(
            {'_id': message_id},
            {'$set': {'query': new_query, 'timestamp': datetime.datetime.utcnow()}}
        )

        # Generate a new assistant response
        selected_llm = openai_llm if llm_choice == 'openai' else ollama_llm
        custom_prompt_template = MongoDBContextPrompt(template="{context}\n\n{query}")
        custom_engine = CustomRetrieverQueryEngine(llm=selected_llm, prompt_template=custom_prompt_template)
        result = custom_engine.query(new_query, discussion_id)

        if not result['sources']:
            return jsonify({'error': 'No relevant documents found for the query'}), 404

        # Insert the new assistant response
        assistant_message = chats_collection.insert_one({
            'discussion_id': ObjectId(discussion_id),
            'role': 'assistant',
            'response': result['response'],
            'timestamp': datetime.datetime.utcnow(),
            'parent_message_id': message_id
        })
    
        return jsonify({
            'response': result['response'],
            'assistant_message_id': str(assistant_message.inserted_id)
        })

    except Exception as e:
        logging.error(f"Error editing message: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

    
@app.route('/download_discussion/<discussion_id>', methods=['GET'])
def download_discussion(discussion_id):
    try:
        # Retrieve the discussion
        discussion = discussions_collection.find_one({'_id': ObjectId(discussion_id)})
        if not discussion:
            return jsonify({'error': 'Discussion not found'}), 404

        # Retrieve chat messages for the discussion
        chats = list(chats_collection.find({'discussion_id': ObjectId(discussion_id)}).sort('timestamp', ASCENDING))
        
        if not chats:
            return jsonify({'error': 'No messages found for this discussion'}), 404
        
        # Prepare the text file content
        discussion_text = ""
        for chat in chats:
            role = chat['role'].capitalize()
            content = chat.get('query', '') if chat['role'] == 'user' else chat.get('response', '')
            discussion_text += f"{role}: {content}\n\n"
        
        # Use the discussion name as the file name, or fallback to discussion_id if no name is set
        file_name = discussion.get('name', f'discussion_{discussion_id}').replace(" ", "_")  # Remove spaces for filename safety
        
        # Return the content as a downloadable text file
        response = make_response(discussion_text)
        response.headers['Content-Disposition'] = f'attachment; filename={file_name}.txt'
        response.mimetype = 'text/plain'
        return response

    except Exception as e:
        logging.error(f"Error downloading discussion: {e}", exc_info=True)
        return jsonify({'error': 'Error downloading discussion'}), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                text_chunks = split_text_to_chunks(text, chunk_size=1000, chunk_overlap=200, page_number=page_number)
                for chunk in text_chunks:
                    chunks_collection.insert_one({
                        'filename': filename,
                        'page_number': chunk['page_number'],
                        'text': chunk['text']
                    })
                    # Add the embeddings to the FAISS index
                    embedding = embed(chunk['text'])
                    if len(embedding) != dimension:
                        raise ValueError(f"Embedding dimension {len(embedding)} does not match FAISS index dimension {dimension}.")
                    faiss_index.add_with_ids(np.array([embedding]), np.array([page_number]))

        # Save the FAISS index after processing the PDF
        save_faiss_index()
        return jsonify({'message': 'PDF uploaded and processed', 'filename': filename})
    except Exception as e:
        logging.error(f"Error processing PDF: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_documents', methods=['GET'])
def get_documents():
    try:
        documents = chunks_collection.distinct('filename')
        return jsonify({'documents': documents}), 200
    except Exception as e:
        logging.error(f"Error fetching documents: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    global query_count, memory
    query_count += 1
    data = request.get_json()
    query_text = data.get('query')
    llm_choice = data.get('llm', 'ollama')
    discussion_id = data.get('discussion_id')
    selected_document = data.get('selected_document')

    if query_count >= MAX_QUERIES_BEFORE_CLEAR:
        memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, token_limit=4096)
        query_count = 0
        logging.info("Memory automatically cleared after {} queries".format(MAX_QUERIES_BEFORE_CLEAR))

    if not query_text:
        return jsonify({'error': 'Query parameter is missing'}), 400

    try:
        logging.info(f"Processing query: {query_text} with LLM: {llm_choice} on document: {selected_document}")

        # Set the LLM according to user's choice
        selected_llm = openai_llm if llm_choice == 'openai' else ollama_llm

        # Create an instance of CustomRetrieverQueryEngine
        custom_prompt_template = MongoDBContextPrompt(template="{context}\n\n{query}")
        custom_engine = CustomRetrieverQueryEngine(llm=selected_llm, prompt_template=custom_prompt_template)

        # Get chat history
        chat_history = memory.get()
        chat_history_str = "\n".join([f"{msg.role.value}: {msg.content}" for msg in chat_history])

        # Use the custom engine to process the query
        result = custom_engine.query(query_text, discussion_id, selected_document, chat_history=chat_history_str)

        # Update chat memory with user query and assistant response
        memory.put(ChatMessage(role=MessageRole.USER, content=query_text))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=result['response']))

        if not result['sources']:
            return jsonify({'error': 'No relevant documents found for the query'}), 404
           # Update discussion name to the first query
        
        # Check if assistant response already exists for the parent message (to avoid duplicates)
        user_message_id = chats_collection.insert_one({
            'discussion_id': ObjectId(discussion_id),
            'role': 'user',
            'query': query_text,
            'timestamp': datetime.datetime.utcnow()
        }).inserted_id

        if not chats_collection.find_one({
            'discussion_id': ObjectId(discussion_id),
            'role': 'assistant',
            'parent_message_id': user_message_id
        }):
            chats_collection.insert_one({
                'discussion_id': ObjectId(discussion_id),
                'role': 'assistant',
                'response': result['response'],
                'timestamp': datetime.datetime.utcnow(),
                'parent_message_id': user_message_id
            })
            # Only update the discussion name if it hasn't been set yet (i.e., it's the first message in the discussion)
            discussions_collection.update_one(
              {'_id': ObjectId(discussion_id), 'name': {'$in': [None, 'New Discussion']}},  # Only update if name is 'New Discussion' or None
            {'$set': {'name': query_text[:50]}}  # Set the name to the first 50 characters of the query
)


        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/delete_discussion/<discussion_id>', methods=['DELETE'])
def delete_discussion(discussion_id):
    try:
        # Delete discussion and related chats
        result = discussions_collection.delete_one({'_id': ObjectId(discussion_id)})
        chats_collection.delete_many({'discussion_id': ObjectId(discussion_id)})
        
        if result.deleted_count == 1:
            return jsonify({'message': 'Discussion deleted successfully'}), 200
        else:
            return jsonify({'error': 'Discussion not found'}), 404
    except Exception as e:
        logging.error(f"Error deleting discussion: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    global memory
    memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, token_limit=4096)
    return jsonify({'message': 'Chat memory cleared'}), 200

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False, port=5000)
