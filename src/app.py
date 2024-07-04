from constants import *
from util import *
import streamlit as st



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
