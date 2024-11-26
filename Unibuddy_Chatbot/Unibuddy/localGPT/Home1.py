import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import speech_recognition as sr
import urllib.parse
import torch
import logging  # Added logging module
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from prompt_template_utils import get_prompt_template
from utils import get_embeddings
from langchain.vectorstores import Chroma
from transformers import GenerationConfig, pipeline
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)

# Initialize callback manager for streaming output
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Set page configuration
st.set_page_config(
    page_title="Unibuddy App",
    page_icon="👋",
)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK with service account key file
    cred = credentials.Certificate("Store/log.json")
    firebase_admin.initialize_app(cred)

# Firebase configuration
config = {
    'apiKey': "AIzaSyAyX1XejPVzi3VZjmHrsrFuubLZzZ-ycoE",
    'authDomain': "login-info-134a1.firebaseapp.com",
    'databaseURL': "https://login-info-134a1-default-rtdb.firebaseio.com",
    'projectId': "login-info-134a1",
    'storageBucket': "login-info-134a1.appspot.com",
    'messagingSenderId': "362592688125",
    'appId': "1:362592688125:web:75e6a093b01366ad95ab94",
    'measurementId': "G-V0F4BQRCL2" 
}
            
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

query_params = st.query_params
if "email" in query_params:
    email = urllib.parse.unquote(query_params["email"])
    st.write(f"Email: {email}")
else:
    st.write("No email found")

if "degree" in query_params:
    degree = urllib.parse.unquote(query_params["degree"])
    st.write(f"Degree: {degree}")

if "major" in query_params:
    major = urllib.parse.unquote(query_params["major"])
    st.write(f"Major: {major}")

if "current_semester" in query_params:
    current_semester = urllib.parse.unquote(query_params["current_semester"])
    st.write(f"Current Semester: {current_semester}")

if "interest1" in query_params:
    interest1 = urllib.parse.unquote(query_params["interest1"])
    st.write(f"Interest 1: {interest1}")

if "interest2" in query_params:
    interest2 = urllib.parse.unquote(query_params["interest2"])
    st.write(f"Interest 2: {interest2}")

if "interest3" in query_params:
    interest3 = urllib.parse.unquote(query_params["interest3"])
    st.write(f"Interest 3: {interest3}")


query_params = st.query_params
if "email" in query_params:
    email = urllib.parse.unquote(query_params["email"])

    st.write(f"Database Name: {email}")
    collection_name = email
else:
    st.write("No email found")
    collection_name = None

# Function to insert data into Firestore
def insert_data(collection_name, data):
    try:
        db = firestore.client()
        db.collection(collection_name).add(data)
        return True
    except Exception as e:
        st.error(f"Failed to insert data: {e}")
        return False

# Function to recognize speech using microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        #st.info("Listening...")
        audio_data = recognizer.listen(source)
        #st.success("Audio captured!")
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""


# Sidebar for chat history
st.sidebar.title("Chat History")

# Initialize chat history and messages
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

# Load the model and initialize the QA system
def load_model(device_type, model_id, model_basename=None):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, logging)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, logging)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, logging)

    generation_config = GenerationConfig.from_pretrained(model_id)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

def retrieval_qa_pipeline(device_type, use_history, promptTemplate_type="llama"):
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt},
        )

    return qa

# Initialize QA system
qa_system = None

def clear_input():
    st.session_state.user_input = ""
# Main function to handle user interactions
# Main function to handle user interactions
def main():
    global qa_system

    # Initialize session state variables if not initialized
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("Unibuddy App")

    # Check if the QA system has been initialized
    if not qa_system:
        qa_system = retrieval_qa_pipeline(device_type="cpu", use_history=False)

    st.write("How can I help you today?")

    # Dropdown menu to select input method
    input_method = st.selectbox("Select Input Method", ["Text", "Voice", "Image"])

    # Display input fields based on selected method
    if input_method == "Text":
        handle_text_input()
    elif input_method == "Voice":
        handle_voice_input()
    elif input_method == "Image":
        handle_image_input()
    #elif input_method == "Logout":
     #   logout()


    # Display chat history in the sidebar
    display_chat_history()

def logout():
    st.markdown('<a href="http://127.0.0.1:5500/localGPT/Demo/login1.html" target="_self"> Logout </a>', unsafe_allow_html=True)
def handle_text_input():
    user_input = st.text_input("", value=st.session_state.user_input, placeholder="Message Unibuddy...")
    if user_input != st.session_state.user_input:
        st.session_state.user_input = user_input
        handle_user_input(user_input)


def handle_voice_input():
    if st.button("🎙️", key="voice_button"):
        voice_input = recognize_speech()
        if voice_input:
            st.session_state.user_input = voice_input
            handle_user_input(voice_input)


def handle_image_input():
    if st.button("📷"):
        uploaded_file = st.file_uploader("", type=["jpg", "png"])
        if uploaded_file is not None:
            # Perform image processing or analysis here
            st.session_state.user_input = "Image uploaded"
            handle_user_input("Image uploaded")


def handle_user_input(input_text):
    # Store user message in chat history
    st.session_state.chat_history.append({"query": input_text, "response": ""})

    # Get response from the QA system
    response = qa_system(input_text)["result"]
    insert_data(collection_name, {"input_text": input_text, "response": response})
    with st.chat_message(name="user"):
        st.markdown(st.session_state.user_input)
    with st.chat_message("assistant"):
        st.markdown(response)

    # Update response for the current query in chat history
    if st.session_state.chat_history:
        st.session_state.chat_history[-1]["response"] = response


def display_chat_history():
    # Display chat history in the sidebar
    for idx, entry in enumerate(reversed(st.session_state.chat_history)):
        with st.sidebar.expander(f"{idx + 1}. Query: {entry['query']}"):
            st.write(f"Response: {entry['response']}")


if __name__ == "__main__":
    main()
