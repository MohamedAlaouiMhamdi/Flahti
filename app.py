from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from transformers import pipeline

# Load documents from the "data" directory
documents = SimpleDirectoryReader("data").load_data()

# Set the BGE embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Configure Ollama with the Mistral model and set a request timeout
Settings.llm = Ollama(model="mistral", request_timeout=1000.0)

# Create a vector store index from the loaded documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Initialize the Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for sending messages to the bot
@app.route('/send_message', methods=['POST'])
def send_message():
    # Get the user's message from the request
    user_message = request.form['user_message']
    # Query the bot's response using the query engine
    bot_response = query_engine.query(user_message)
    # Return the bot's response as a JSON object
    return jsonify({'bot_response': str(bot_response)})

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)
