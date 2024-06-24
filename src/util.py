from constants import *
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import tempfile
import pymupdf
from sentence_transformers import SentenceTransformer
import openai

# Initialize OpenAI API (Replace with your API key)
openai.api_key = OPENAI_API_KEY


def get_mongodb_cluster_connection_uri(mongodb_username: str, mongodb_user_password: str, mongodb_cluster_hostname: str) -> str: 
    """
    Return the MongoDB cluster connection URI

    Parameters:
    mongodb_username (str): MongoDB username
    mongodb_user_password (str): Password of the MongoDB user
    mongodb_cluster_hostname (str): MongoDB cluster hostname

    Returns:
    str: MongoDB cluster connection uri
    """

    mongodb_cluster_hostname_str = mongodb_cluster_hostname.split(".") 
    mongodb_clustername = mongodb_cluster_hostname_str[0]
    
    uri = ("mongodb+srv://" +
           mongodb_username +
           ":" +
           mongodb_user_password +
           "@" +
           mongodb_cluster_hostname +
           "/?retryWrites=true&w=majority&appName=" +
           mongodb_clustername)
    #print(" uri: ", uri, "\t type(uri): ", type(uri))
    return uri


def get_mongodb_cluster_client(uri: str) -> MongoClient:
    """
    Return the MongoDB cluster connection

    Parameters:
    uri (str): MongoDB connection uri

    Returns:
    pymongo.mongo_client.MongoClient: MongoDB cluster connection`
    """

    client = None
    if uri:
        try:
            # Create a new client and connect to the server
            client = MongoClient(uri, server_api=ServerApi('1'))
            print(" client: ", client, "\t type(client): ", type(client))

            # Send a ping to confirm a successful connection
            client.admin.command('ping')
            print("Pinged the MongoDB cluster deployment. Successfully connected to MongoDB cluster!")
        except Exception as e:
            print(f"{e}")
            raise Exception(
                'Failed to connect to MongoDB database.  Please supply valid MongoDB username, MongoDB user password, MongoDB cluster hostname parameters') from e

    return client


def get_mongodb_database(mongodb_client, database_name: str):
    """ Returns vectorsdb """
    db = None
    if mongodb_client and database_name:
        db = mongodb_client[database_name]
        print(" db: ", db, "\t type(db): ", type(db))
    return db


def get_collection(db, collection_name: str):
    collection = None
    if db is not None and collection_name:
        collection = db[collection_name]
        print(" collection: ", collection, "\t type(collection): ", type(collection))
    return collection


def extract_text_from_pdf(pdf_path):
    text = ""

    if pdf_path:
        try:
            doc = pymupdf.open(pdf_path)  # Open the PDF document
            if doc is not None:
                #print(" doc: ", doc, "\t type(doc): ", type(doc))
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)  # Load a page
                    print(" page: ", page, "\t type(page): ", type(page))
                    text += page.get_text()  # Extract text from the page
        except Exception as e:
            print(f"{e}")
            raise Exception('Error extracting text from PDF: {e}') from e

    return text


"""
# This is a sentence-transformers model.  It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
def get_sentence_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
"""


# Function to vectorize text
def vectorize_text(text):
    embeddings = None

    if text:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            if model:
                embeddings = model.encode(text)
                print(" embeddings: ", embeddings, "\t type(embeddings): ", type(embeddings))
        except Exception as e:
            print(f"{e}")

    return embeddings


# Function to save vector to MongoDB
def save_embeddings_to_collection(embeddings, text, collection):
    print("\n\n save_embeddings_to_collection(embeddings, text, collection)")
    print("\n embeddings: ", embeddings)
    print("\n text: ", text)
    print("\n collection: ", collection)

    if embeddings is not None and text and collection is not None:
        document = {
            'text': text,
            'vector': embeddings.tolist()  # Convert numpy array to list
        }
        collection.insert_one(document)


# Function to retrieve relevant documents from MongoDB
def retrieve_relevant_docs(query, collection):
    if query and collection is not None:
        # model = get_sentence_transformer_model()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        if model:
            query_vector = model.encode(query).tolist()
            docs = list(collection.find())
            if docs and query_vector:
                relevant_docs = sorted(docs, key=lambda doc: cosine_similarity(query_vector, doc['vector']),
                                       reverse=True)[:5]
                return relevant_docs


# Cosine similarity function
def cosine_similarity(vec1, vec2):
    if vec1 and vec2:
        print(" vec1: ", vec1, "\t type(vec1): ", type(vec1))
        print(" vec2: ", vec2, "\t type(vec2): ", type(vec2))

        result = sum(a * b for a, b in zip(vec1, vec2)) / (
                    sum(a * a for a in vec1) ** 0.5 * sum(b * b for b in vec2) ** 0.5)
        print(" result: ", result, "\t type(result): ", type(result))
        return result
        # return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a * a for a in vec1) ** 0.5 * sum(b * b for b in vec2) ** 0.5)


# Function to generate chatbot response using OpenAI GPT
def generate_response(query, relevant_docs):
    if query and relevant_docs:
        augmented_query = query + " " + " ".join([doc['text'] for doc in relevant_docs])
        if augmented_query:
            print(" augmented_query: ", augmented_query, "\t type(augmented_query): ", type(augmented_query))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": augmented_query}
                ],
                max_tokens=150
            )
            if response:
                print(" response: ", response, "\t type(response): ", type(response))
                return response['choices'][0]['message']['content'].strip()


# Function to save chat history to MongoDB
def save_chat_history(user_query, nextmove_response, collection):
    if user_query and nextmove_response and collection is not None:
        document = {
            'user_query': user_query,
            'nextmove_response': nextmove_response
        }
        collection.insert_one(document)


def get_text_from_pdf(file):
    if file:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_pdf_path = tmp_file.name

        # Extract text from PDF
        text = extract_text_from_pdf(tmp_pdf_path)
        print(" text: ", text, "\t type(text): ", type(text))

        return text
    return None


if __name__ == "__main__":
    s = "mongodbcluster0.bjmkbwc.mongodb.net"
    result = s.split(".")
    print(" result[0]: ", result[0])
