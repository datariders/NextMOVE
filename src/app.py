from constants import *
from util import *
import streamlit as st



def main():
    uri = get_mongodb_cluster_connection_uri(MONGODB_USERNAME, MONGODB_USER_PASSWORD, MONGODB_CLUSTER_HOSTNAME)
    if uri:
        mongodb_client = get_mongodb_cluster_client(uri)
        if mongodb_client:
            games_db = get_mongodb_database(mongodb_client, "vectorsdb")
            #print(" games_db: ", games_db, "\t type(games_db): ", type(games_db))

            if games_db is not None:
                games_collection = get_collection(games_db, "vectors")
                #print(" games_collection: ", games_collection, "\t type(games_collection): ", type(games_collection))
    
                chat_history_collection = get_collection(games_db, "chat_history")
                #print(" chat_history_collection: ", chat_history_collection, "\t type(chat_history_collection): ", type(chat_history_collection))


    # Streamlit interface
    st.image('assets/header.png')
    st.title("NextMOVE: Chess training assistant")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Chess games (pdf)", type="pdf")
    if uploaded_file:
        #print(" uploaded_file: ", uploaded_file, "\t type(uploaded_file): ", type(uploaded_file))
        text = get_text_from_pdf(uploaded_file)
        if text:
            # Vectorize text
            embeddings = vectorize_text(text)
            if embeddings is not None:
                #print(" embeddings: ", embeddings, "\t type(embeddings): ", type(embeddings))

                # Save vector to MongoDB
                save_embeddings_to_collection(embeddings, text, games_collection)
 
                st.success("Game embeddings are saved to MongoDB collection successfully!")

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
