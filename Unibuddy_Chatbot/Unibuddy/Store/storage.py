import firebase_admin
from firebase_admin import credentials, firestore, auth
import csv
import streamlit as st

# Check if the Firebase Admin SDK has been initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK with service account key file
    cred = credentials.Certificate("/Users/saikarthiknaladala/Documents/UNIBUDDY/Store/log.json")
    firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Function to retrieve all collection names from Firestore
def get_collection_names():
    collections = db.collections()
    return [collection.id for collection in collections]

def get_all_docs(collection_names):
    documents_list = []
    for collection_name in collection_names:
        docs = db.collection(collection_name).stream()
        for doc in docs:
            doc_data = doc.to_dict()
            doc_id = doc.id
            prompt = doc_data.get('input_text', '')
            response = doc_data.get('response', '')
            documents_list.append({'Collection': collection_name, 'Id': doc_id, 'Prompt': prompt, 'Response': response})
    return documents_list



# Streamlit app
def main():
    st.title("Firebase Collection Data Export")


    # Get all collection names from Firestore
    collection_names = get_collection_names()

    # Allow the user to select multiple collections
    all_selected = st.checkbox("Select all collections")
    if all_selected:
        selected_collections = collection_names
    else:
        selected_collections = st.multiselect("Select collections:", collection_names)

    # Retrieve all documents from the selected collections
    documents = get_all_docs(selected_collections)

    # Define CSV file path
    csv_file_path = "/Users/saikarthiknaladala/Documents/csvfiles/chathistory.csv"

    # Write data to CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ['Collection', 'Id', 'Prompt', 'Response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for document in documents:
            writer.writerow(document)
        st.success(f"Data successfully saved to '{csv_file_path}'")

if __name__ == '__main__':
    main()
