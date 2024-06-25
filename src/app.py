from constants import *
from util import *
import streamlit as st



def main():
    assert MONGODB_USERNAME is not None and len(MONGODB_USERNAME) > 0, "MONGODB_USERNAME not set."
    assert MONGODB_USER_PASSWORD is not None and len(MONGODB_USER_PASSWORD) > 0, "MONGODB_USER_PASSWORD not set."
    assert MONGODB_CLUSTER_HOSTNAME is not None and len(MONGODB_CLUSTER_HOSTNAME) > 0, "MONGODB_CLUSTER_HOSTNAME not set."

    uri = get_mongodb_cluster_connection_uri(MONGODB_USERNAME, MONGODB_USER_PASSWORD, MONGODB_CLUSTER_HOSTNAME)
    assert uri is not None, "uri not set."

    if uri:
        mongodb_client = get_mongodb_cluster_client(uri)
        if mongodb_client:
            #games_db = get_mongodb_database(mongodb_client, "vectorsdb")
            games_db = get_mongodb_database(mongodb_client, MONGODB_CLUSTER_DATABASE_NAME)
            if games_db is not None:
                #games_collection = get_collection(games_db, "vectors")
                games_collection = get_collection(games_db, MONGODB_DATABASE_GAMES_COLLECTION_NAME)

                #chat_history_collection = get_collection(games_db, "chat_history")
                chat_history_collection = get_collection(games_db, MONGODB_DATABASE_CHAT_HISTORY_COLLECTION_NAME)


    # Streamlit interface
    st.image('assets/header.png')
    st.title("NextMOVE: Chess training assistant")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Chess games (pdf)", type="pdf")
    if uploaded_file:
        text = get_text_from_pdf(uploaded_file)
        if text:
            # Vectorize text
            embedding = vectorize_text(text)
            if embedding is not None:
                #print(" embedding: ", embedding, "\t type(embedding): ", type(embedding))

                # Save vector to MongoDB
                save_embedding_to_collection(embedding, text, games_collection)
 
                st.success("Game embedding is saved to MongoDB collection successfully!")

                user_query = st.text_input("Enter your move:")
                if user_query and games_collection is not None and chat_history_collection is not None:
                    print(" user_query: ", user_query, "\t type(user_query): ", type(user_query))
                    
                    # Retrieve relevant documents
                    relevant_docs = retrieve_relevant_docs(user_query, games_collection)
                    if relevant_docs:
                        print(" relevant_docs: ", relevant_docs, "\t type(relevant_docs): ", type(relevant_docs))

                        # Generate response
                        bot_response = generate_response(user_query, relevant_docs)
                        if bot_response:
                            print(" bot_response: ", bot_response, "\t type(bot_response): ", type(bot_response))

                            # Display response
                            st.write("NextMOVE response:")
                            st.write(bot_response)

                            # Save chat history to MongoDB
                            save_chat_history(user_query, bot_response, chat_history_collection)            


if __name__ == "__main__":
    main()
