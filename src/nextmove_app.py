from constants import *
from util import *
#import tempfile

#from sentence_transformers import SentenceTransformer
#import pymupdf
#import openai
import streamlit as st



def main():
    uri = get_mongodb_cluster_connection_uri(MONGODB_USERNAME, MONGODB_USER_PASSWORD, MONGODB_CLUSTER_HOSTNAME, MONGODB_CLUSTERNAME)
    #print(" uri: ", uri, "\t type(uri): ", type(uri))

    mongodb_client = get_mongodb_cluster_client(uri)
    games_db = get_mongodb_database(mongodb_client, "vectorsdb")
    print(" games_db: ", games_db, "\t type(games_db): ", type(games_db))

    games_collection = get_collection(games_db, "vectors")
    print(" games_collection: ", games_collection, "\t type(games_collection): ", type(games_collection))
    
    chat_history_collection = get_collection(games_db, "chat_history")
    print(" chat_history_collection: ", chat_history_collection, "\t type(chat_history_collection): ", type(chat_history_collection))

    # Streamlit interface
    st.image('assets/header.png')
    st.title("NextMOVE: Chess training assistant")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Chess games (pdf)", type="pdf")
    if uploaded_file is not None:
        text = get_text_from_pdf(uploaded_file)

        if text is not None:
            # Vectorize text
            embeddings = vectorize_text(text)
            if embeddings is not None:
                # Save vector to MongoDB
                #save_vector_to_mongo(embeddings, text, games_collection)
                save_embeddings_to_collection(embeddings, text, games_collection)
 
                st.success("Game embeddings are saved to MongoDB collection successfully!")

                user_query = st.text_input("Enter your move:")
                if user_query is not None:
                    # Retrieve relevant documents
                    relevant_docs = retrieve_relevant_docs(user_query, games_collection)
                    if relevant_docs is not None:
                        # Generate response
                        bot_response = generate_response(user_query, relevant_docs)
                        if bot_response is not None:
                            # Display response
                            st.write("NextMOVE response:")
                            st.write(bot_response)

                            # Save chat history to MongoDB
                            save_chat_history(user_query, bot_response, chat_history_collection)            



if __name__ == "__main__":
    main()
