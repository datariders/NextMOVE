from pymongo.mongo_client import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import tempfile
import pymupdf
from sentence_transformers import SentenceTransformer
import openai
from streamlit.runtime.uploaded_file_manager import UploadedFile
from numpy import ndarray
import certifi
import streamlit as st


OPENAI_API_KEY = "<ENTER_YOUR_OPENAI_API_KEY>"
MONGODB_CLUSTER_HOSTNAME = "<ENTER_YOUR_MONGODB_CLUSTER_HOSTNAME>"
MONGODB_CLUSTER_DATABASE_NAME = "<ENTER_YOUR_MONGODB_CLUSTER_DATABASE_NAME>"
MONGODB_DATABASE_GAMES_COLLECTION_NAME = "<ENTER_YOUR_MONGODB_DATABASE_GAMES_COLLECTION_NAME>"
MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME = "<ENTER_YOUR_MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME>"

SENTENCE_TRANSFORMER_PARAPHRASE_MINI_LM_L6_v2 = "paraphrase-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_ALL_MINI_LM_L6_v2 = "all-MiniLM-L6-v2"

OPENAI_MODEL_GPT_3_5_TURBO = "gpt-3.5-turbo"
MAX_TOKENS = 150
DOCUMENT_TEXT = "text"
DOCUMENT_VECTOR = "vector"


def init_config_parameters():
    """
    Verify and initialize the config parameters

    Parameters:
    None

    Returns:
    None
    """

    assert MONGODB_USERNAME is not None and len(MONGODB_USERNAME) > 0, "MONGODB_USERNAME not set."
    assert MONGODB_USER_PASSWORD is not None and len(MONGODB_USER_PASSWORD) > 0, "MONGODB_USER_PASSWORD not set."
    assert MONGODB_CLUSTER_HOSTNAME is not None and len(MONGODB_CLUSTER_HOSTNAME) > 0, "MONGODB_CLUSTER_HOSTNAME not set."
    assert MONGODB_CLUSTER_DATABASE_NAME is not None and len(MONGODB_CLUSTER_DATABASE_NAME) > 0, "MONGODB_CLUSTER_DATABASE_NAME not set."
    assert MONGODB_DATABASE_GAMES_COLLECTION_NAME is not None and len(MONGODB_DATABASE_GAMES_COLLECTION_NAME) > 0, "MONGODB_DATABASE_GAMES_COLLECTION_NAME not set."
    assert MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME is not None and len(MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME) > 0, "MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME not set."

    # Initialize OpenAI API (Replace with your API key)
    if openai.api_key is None:
        openai.api_key = OPENAI_API_KEY
    assert openai.api_key is not None, "OpenAI API key not found."


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
    return uri


def get_mongodb_cluster_client(uri: str) -> MongoClient:
    """
    Return the MongoDB cluster connection

    Parameters:
    uri (str): MongoDB connection uri

    Returns:
    pymongo.mongo_client.MongoClient: MongoDB cluster connection`
    """

    mongodb_client = None
    if uri:
        try:
            # Create a new client and connect to the server
            mongodb_client = MongoClient(uri, tlsCAFile=certifi.where())

            # Send a ping to confirm a successful connection
            mongodb_client.admin.command('ping')
            print("\n Pinged the MongoDB cluster deployment. Successfully connected to MongoDB cluster!")
        except Exception as e:
            raise Exception(
                'Failed to connect to MongoDB database.  Please supply valid MongoDB username, MongoDB user password, MongoDB cluster hostname parameters') from e

    return mongodb_client 


def get_mongodb_database(mongodb_client: MongoClient, database_name: str) -> Database:
    """
    Connects to MongoDB cluster and returns the games database

    Parameters:
    mongodb_client (MongoClient): MongoDB connection uri
    database_name (str): Name of the game database

    Returns:
    pymongo.database.Database: Game database
    """

    db = None
    if mongodb_client and database_name:
        db = mongodb_client[database_name]
    return db


def get_collection(db: Database, collection_name: str) -> Collection:
    """
    Connects to MongoDB game database and returns the games collection 

    Parameters:
    db (pymongo.database.Database): MongoDB game database
    collection_name (str): Name of the game collection

    Returns:
    pymongo.collection.Collection: Game collection
    """

    collection = None
    if db is not None and collection_name:
        collection = db[collection_name]
    return collection


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts the text out of the game pdf file

    Parameters:
    pdf_path (str): Path of the game pdf file

    Returns:
    str: Returns text out of the game pdf file
    """

    text = ""
    if pdf_path:
        try:
            doc = pymupdf.open(pdf_path)  # Open the PDF document
            if doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)  # Load a page
                    text += page.get_text()  # Extract text from the page
        except Exception as e:
            print(f"{e}")
            raise Exception('Error extracting text from PDF: {e}') from e
    return text


def get_sentence_transformer_model() -> SentenceTransformer:
    """
    Returns SentenceTransformer model

    This is a sentence-transformers model.  It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.  The model returned has the following config parameters:

        SentenceTransformer(
        (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
        (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)

    Parameters:
    None

    Returns:
    sentence_transformers.SentenceTransformer: Returns the SentenceTransformer model
    """

    return SentenceTransformer(SENTENCE_TRANSFORMER_PARAPHRASE_MINI_LM_L6_v2)


# Function to vectorize text
def vectorize_text(text: str) -> ndarray:
    """
    Converts the game text into vector and returns the numpy array of the vector

    Parameters:
    text (str): Text of the game

    Returns:
    numpy.ndarray: Vectorized game text
    """

    embedding = None
    if text:
        try:
            model = get_sentence_transformer_model()
            if model:
                embedding = model.encode(text)
        except Exception as e:
            print(f"{e}")
    return embedding


# Function to save vector to MongoDB
def save_embedding_to_collection(embedding: ndarray, text: str, collection: Collection):
    """
    Store game embedding to the game collection

    Parameters:
    embedding (ndarray): Vectorized numpy array of the game text
    text (str): Text of the game
    collection (pymongo.collection.Collection): game collection

    Returns:
    None
    """

    if embedding is not None and text and collection is not None:
        document = {
            DOCUMENT_TEXT: text,
            DOCUMENT_VECTOR: embedding.tolist()  # Convert numpy array to list
        }

        existing_document = collection.find_one(document)
        if not existing_document:
            collection.insert_one(document)
            print("\n Game saved into MongoDB collection as embedding!")
        else:
            print("\n This game exists in the MongoDB game collection as embedding!")


# Function to retrieve relevant documents from MongoDB
def retrieve_relevant_docs(query: list, collection: Collection) -> list:
    """
    Retrieve relevant document based on the user query

    Parameters:
    query (list): User query
    collection (pymongo.collection.Collection): game collection

    Returns:
    list: document relevant to the user query
    """

    if query and collection is not None:
        model = get_sentence_transformer_model()
        if model:
            query_vector = model.encode(query).tolist()
            docs = list(collection.find())
            if docs and query_vector:
                relevant_docs = sorted(docs, key=lambda doc: cosine_similarity(query_vector, doc[DOCUMENT_VECTOR]),
                                       reverse=True)[:5]
                return relevant_docs


# Cosine similarity function
def cosine_similarity(vector_1: list, vector_2: list) -> float:
    """
    Returns cosine similarity value

    Parameters:
    vector_1 (list): User query
    vector_2 (list): Game collection

    Returns:
    float: cosine similarity value
    """

    if vector_1 and vector_2:
        return sum(a * b for a, b in zip(vector_1, vector_2)) / (sum(a * a for a in vector_1) ** 0.5 * sum(b * b for b in vector_2) ** 0.5)


# Function to generate chatbot response using OpenAI GPT
def generate_response(query: str, relevant_docs: list) -> str:
    """
    Returns next move response for the user query

    Parameters:
    query (str): User query
    relevant_docs (list): The relevant game collection

    Returns:
    str: Next move response for the user query
    """

    if query and relevant_docs:
        augmented_query = query + " " + " ".join([doc[DOCUMENT_TEXT] for doc in relevant_docs])
        if augmented_query:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL_GPT_3_5_TURBO,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": augmented_query}
                ],
                max_tokens=MAX_TOKENS
            )
            if response:
                return response['choices'][0]['message']['content'].strip()



# Function to save chat history to MongoDB
def save_chat_history(user_query: str, nextmove_response: str, collection: Collection):
    """
    Stores the chat history in the chat collection

    Parameters:
    user_query (str): User query
    nextmove (str): The nextmove response
    collection (pymongo.collection.Collection): The relevant game collection

    Returns:
    None
    """

    if user_query and nextmove_response and collection is not None:
        document = {
            'user_query': user_query,
            'nextmove_response': nextmove_response
        }
        collection.insert_one(document)


def get_text_from_pdf(file: UploadedFile) -> str:
    """
    Stores the chat history in the chat collection

    Parameters:
    file (streamlit.runtime.uploaded_file_manager.UploadedFile): The game pdf file

    Returns:
    str: Extracts text from the game pdf
    """

    if file:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_pdf_path = tmp_file.name

        # Extract text from PDF
        return extract_text_from_pdf(tmp_pdf_path)
    return None


def main():
    init_config_parameters()

    mongodb_uri = get_mongodb_cluster_connection_uri(MONGODB_USERNAME, MONGODB_USER_PASSWORD, MONGODB_CLUSTER_HOSTNAME)
    assert mongodb_uri is not None, "mongodb_uri not set."

    mongodb_client = get_mongodb_cluster_client(mongodb_uri)
    assert mongodb_client is not None, "mongodb_client not set."

    games_db = get_mongodb_database(mongodb_client, MONGODB_CLUSTER_DATABASE_NAME)
    assert games_db is not None, "games_db not set."

    games_collection = get_collection(games_db, MONGODB_DATABASE_GAMES_COLLECTION_NAME)
    assert games_collection is not None, "games_collection not set."

    chat_history_collection = get_collection(games_db, MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME)
    assert chat_history_collection is not None, "chat_history_collection not set."


    # Streamlit interface
    st.image('assets/nextmove_header.png')
    st.title("NextMOVE: Chess training assistant")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Chess games (pdf)", type="pdf")
    if uploaded_file:
        game_text = get_text_from_pdf(uploaded_file)
        assert game_text is not None, "game_text not set."
        print("\n Extracted game_text: \n", game_text)

        # Vectorize text
        game_embedding = vectorize_text(game_text)
        assert game_embedding is not None, "game_embedding not set."
        print("\n Vectorized game_embedding: \n", game_embedding)

        # Save vector to MongoDB
        save_embedding_to_collection(game_embedding, game_text, games_collection) 

        user_query = st.text_input("Enter your query:")
        assert user_query is not None, "user_query not set."
        print("\n user_query: ", user_query)

        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_docs(user_query, games_collection)
        if relevant_docs:
            # Generate response
            nextmove_response = generate_response(user_query, relevant_docs)
            assert nextmove_response is not None, "nextmove_response not set."
            print("\n nextmove_response: ", nextmove_response)

            # Display response
            st.write("NextMOVE response:")
            st.write(nextmove_response)

            # Save chat history to MongoDB
            save_chat_history(user_query, nextmove_response, chat_history_collection)            
            print("\n Chat saved into MongoDB collection as embedding!")


if __name__ == "__main__":
    main()
