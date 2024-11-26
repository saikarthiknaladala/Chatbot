# UniBuddy - Virtual Personal Assistant

UniBuddy is a versatile virtual personal assistant designed to support various tasks and workflows using state-of-the-art technologies such as Mistral 7B, LangChain, and NLTK. Whether you need assistance with document ingestion, language processing, or interactive question answering, UniBuddy has got you covered.

## Features

### Document Ingestion
UniBuddy's `ingest.py` module provides efficient document ingestion capabilities, allowing you to load and process documents from specified directories. Leveraging multithreading and multiprocessing techniques, UniBuddy can handle large volumes of documents while maintaining optimal performance.

### Language Processing
The `run_localGPT.py` module harnesses the power of LangChain and Mistral 7B to enable advanced language processing functionalities. From question answering to text generation, UniBuddy utilizes pre-trained language models to deliver accurate and context-aware responses.

### Interactive Q&A
With UniBuddy, you can engage in interactive question-and-answer sessions to seek information and insights on various topics. By combining document retrieval with language modeling techniques, UniBuddy provides prompt and informative responses to your queries.

### Customization and Configuration
UniBuddy offers flexibility and customization options to adapt to your specific needs and preferences. From choosing the device type for computation to configuring model parameters and callbacks, UniBuddy empowers you to tailor the virtual assistant according to your requirements.

## Getting Started

To start using UniBuddy, follow these steps:

1. **Installation**: Clone the UniBuddy repository to your local machine.

2. **Dependencies**: Install the required dependencies, including Torch, Click, LangChain, NLTK, and Transformers. Refer to the documentation for detailed installation instructions.

3. **Configuration**: Customize the settings and parameters in the `constants.py` file according to your preferences. Specify the source directories for document ingestion and configure the language models and embeddings to be used.

4. **Execution**: Run the `ingest.py` and `runlocal_GPT.py` scripts to initiate the document ingestion and language processing workflows, respectively. Use the provided command-line options to specify device types, enable/disable features, and interact with UniBuddy.

5. **Interaction**: Interact with UniBuddy through the command-line interface or API endpoints. Pose questions, provide input text, or request specific tasks to be performed by UniBuddy, and observe the intelligent responses generated in real-time.


---

*UniBuddy - Your Trusted Virtual Personal Assistant*

*Empowering You with Intelligent Support and Assistance*
