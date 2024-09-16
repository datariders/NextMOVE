# pip3 install -U pymongo PyPDF2 sentence-transformers openai==0.28

import os
from pymongo.mongo_client import MongoClient
import certifi
from pymongo.collection import Collection
import PyPDF2
from sentence_transformers import SentenceTransformer
import openai
from numpy import ndarray



MONGODB_URI = "<PROVIDE_MONGODB_URI_HERE>"
MONGODB_CLUSTER_DATABASE_NAME = "<PROVIDE_MONGODB_CLUSTER_DATABASE_NAME_HERE>"
MONGODB_DATABASE_GAMES_COLLECTION_NAME = "<PROVIDE_MONGODB_DATABASE_GAMES_COLLECTION_NAME_HERE>"
MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME = "<PROVIDE_MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME_HERE>"
OPENAI_API_KEY = "<PROVIDE_OPENAI_API_KEY_HERE>"
GAME_DIRECTORY = "<PROVIDE_DIRECTORY_WHERE_GAMES_ARE_STORED_HERE>"

# Initialize OpenAI API (Replace with your API key)
if openai.api_key is None:
    openai.api_key = OPENAI_API_KEY
assert openai.api_key is not None, "OpenAI API key not found."


mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
games_db = mongodb_client[MONGODB_CLUSTER_DATABASE_NAME]
games_collection = games_db[MONGODB_DATABASE_GAMES_COLLECTION_NAME]
chat_history_collection = games_db[MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME]


# Function to vectorize text using SentenceTransformer('paraphrase-MiniLM-L6-v2') model
def vectorize_text(text: str) -> ndarray:
    embedding = None
    if text:
        try:
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            if model:
                embedding = model.encode(text)
        except Exception as e:
            print(f"{e}")
    return embedding

# Function to save vector to MongoDB
def save_embedding_to_collection(embedding: ndarray, text: str, collection: Collection):
    if embedding is not None and text and collection is not None:
        document = {
            "text": text,
            "vector": embedding.tolist()  # Convert numpy array to list
        }

        existing_document = collection.find_one(document)
        if not existing_document:
            collection.insert_one(document)
            print("\n Game saved into MongoDB collection as embedding!")
        else:
            print("\n This game exists in the MongoDB game collection as embedding!")

# Function to retrieve relevant documents from MongoDB
def retrieve_relevant_docs(query: list, collection: Collection) -> list:
    if query and collection is not None:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        if model:
            query_vector = model.encode(query).tolist()
            docs = list(collection.find())
            if docs and query_vector:
                relevant_docs = sorted(docs, key=lambda doc: cosine_similarity(query_vector, doc["vector"]),
                                       reverse=True)[:5]
                return relevant_docs

# Cosine similarity function
def cosine_similarity(vector_1: list, vector_2: list) -> float:
    if vector_1 and vector_2:
        return sum(a * b for a, b in zip(vector_1, vector_2)) / (sum(a * a for a in vector_1) ** 0.5 * sum(b * b for b in vector_2) ** 0.5)

# Function to generate chatbot response using OpenAI GPT-3.5
def generate_response(query: str, relevant_docs: list) -> str:
    if query and relevant_docs:
        augmented_query = query + " " + " ".join([doc["text"] for doc in relevant_docs])
        if augmented_query:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": augmented_query}
                ],
                max_tokens=500
            )
            if response:
                return response['choices'][0]['message']['content'].strip()

# Function to save chat history to MongoDB
def save_chat_history(user_query: str, nextmove_response: str, collection: Collection):
    if user_query and nextmove_response and collection is not None:
        document = {
            'user_query': user_query,
            'nextmove_response': nextmove_response
        }
        collection.insert_one(document)

def get_text_from_pdf(file_path: str) -> str:
    # Open and read the PDF file
    try:
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get the number of pages
            num_pages = len(pdf_reader.pages)

            # Iterate through each page and extract text
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                return text
    except FileNotFoundError:
        print(f"The file at path {file_path} was not found.")
    except IOError:
        print(f"An error occurred while reading the file at path {file_path}.")

def main():
    """
    For each game pdf file in the GAME_DIRECTORY, extract the text, vectorize,
    and store the game along with its embeddings in the games collection.
    """    
    for root, dirnames, fnames in os.walk(GAME_DIRECTORY):
        for fname in fnames:
            file = os.path.join(root, fname)
            if file:
                game_text = get_text_from_pdf(file)
                print("\n Extracted game_text: \n", game_text)

                # Vectorize text
                game_embedding = vectorize_text(game_text)
                print("\n Vectorized game_embedding: \n", game_embedding)

                # Save vector to MongoDB
                save_embedding_to_collection(game_embedding, game_text, games_collection) 

    user_query = "<PROVIDE_USER_QUERY_HERE>" # get the user query
    print("\n user_query: ", user_query)

    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(user_query, games_collection)
    if relevant_docs:
        # Generate response
        nextmove_response = generate_response(user_query, relevant_docs)
        print("\n nextmove_response: ", nextmove_response)

        # Save chat history to MongoDB
        save_chat_history(user_query, nextmove_response, chat_history_collection)            
        print("\n Chat saved into MongoDB chat history collection as embedding!")

if __name__ == "__main__":
    main()
