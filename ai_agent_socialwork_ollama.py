from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import gradio as gr
from pathlib import Path
import logging
import sys
import json
import csv  # Add CSV module for saving evaluations


# Define the available models and allow the user to choose one
available_models = {
    "QwQ-32B-4bit": "mlx-community/QwQ-32B-4bit",
    "Gemma-3-27B-4bit": "mlx-community/gemma-3-27b-it-4bit",
    "Qwen2.5-32B-Instruct-4bit": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "Mistral-Large-2407-4bit": "mlx-community/Mistral-Large-Instruct-2407-4bit",
    "Mistral-Small-2501-4bit": "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "Llama-4-Maverick-17B-4bit": "mlx-community/Llama-4-Maverick-17B-16E-Instruct-4bit",
    "Llama-4-Scout-17B-4bit": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    "Mistral-Nemo-Instruct-2407-4bit": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
}

# Function to select a model
def select_model(model_name):
    if model_name in available_models:
        logging.info(f"Model selected: {model_name}")
        return available_models[model_name]
    else:
        logging.error(f"Invalid model name: {model_name}. Defaulting to Mistral-Small-2501-4bit.")
        return available_models["Mistral-Small-2501-4bit"]

# Example: Set the model to use
selected_model = select_model("Gemma-3-27B-4bit")  # Default model
# log the model that has been selected
logging.info(f"Selected model: {selected_model}")

# Configure logging
logging.basicConfig(
    filename="gradio_assisstant_macos.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Ensure the logging configuration is applied
)

# Add a console handler to debug logging issues
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Test logging setup
logging.info("Logging setup complete. Starting application...")

# Load and process all PDF documents from the subfolder
def load_all_documents(folder_path):
    documents = []
    for file in Path(folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = file.name  # Add document name to metadata
        documents.extend(loaded_docs)
    return documents

# Save only the retrieved chunks to a JSON file
def save_chunks_to_file(retrieved_chunks, file_path="retrieved_chunks.json"):
    try:
        if not retrieved_chunks:
            logging.warning("No retrieved chunks to save. The list is empty.")
            return
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in retrieved_chunks],
                f,
                ensure_ascii=False,
                indent=4,
            )
            with open("retrieved_chunks.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Content"])  # Write header
                for chunk in retrieved_chunks:
                    csv_writer.writerow([chunk.page_content])
                    
        logging.info(f"Retrieved chunks successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save retrieved chunks to file: {e}")

# Function to handle user queries
def ask_question(query):
    response = chain.invoke({"input": query})
    return response["answer"]

# Function to handle user queries and save retrieved chunks
def ask_question_with_chunks(query):
    logging.info(f"User query: {query}")
    response = chain.invoke({"input": query})
    answer = response["answer"]
    # retrieved_chunks = response.get("retrieved_documents", [])  # Assuming retrieved chunks are returned here
    # save_chunks_to_file(retrieved_chunks)  # Save only the retrieved chunks
    logging.info(f"Response: {answer}")
    logging.getLogger().handlers[0].flush()  # Explicitly flush logs to the file
    # Save the chunks to a file
    # query = """{input}"""  # Provide a default query for testing
    retrieved_documents = retriever.get_relevant_documents(query)
    save_chunks_to_file(retrieved_documents, "retrieved_chunks.json")

    return answer

# Load the MLXPipeline model immediately
llm = MLXPipeline.from_model_id(
    selected_model,
    pipeline_kwargs={"max_tokens": 2024, "temp": 0.1},
)
logging.info(f"LLM loaded for RAG: {selected_model}")

# Function to update RAG parameters, reload documents, and rerun all necessary functions
def update_rag_parameters(option):
    global retriever, llm, chain, documents, vectorstore, prompt
    logging.info(f"Updating RAG parameters based on user selection: {option}")
    try:
        # Reload documents
        folder_path = "./documents"
        logging.info("Reloading documents...")
        documents = load_all_documents(folder_path)
        if not documents:
            logging.error("No documents were loaded. Please check the folder path and document files.")
            return
        logging.info(f"{len(documents)} documents reloaded successfully.")

        # Split documents into chunks
        logging.info("Splitting documents into chunks...")
        documents = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        ).split_documents(documents)
        if not documents:
            logging.error("No chunks were created. Please check the document processing logic.")
            return
        logging.info(f"{len(documents)} chunks created successfully.")

        # Append document name and page number to each chunk
        for doc in documents:
            source = doc.metadata.get("source", "Unbekanntes Dokument")
            page = doc.metadata.get("page", "Unbekannte Seite")
            doc.page_content += f" ({source}, Seite {page})"

        # Save the chunks to a file
        # save_chunks_to_file(documents)

        # Recreate vectorstore
        logging.info("Recreating vectorstore...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma.from_documents(documents, embeddings)

        # Update retriever and LLM based on the selected option
        if option == "Antworte möglichst genau":
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})
            logging.info("RAG parameters updated: k=8, temp=0.1. Reloading model...")
            llm = MLXPipeline.from_model_id(
                selected_model,
                pipeline_kwargs={"max_tokens": 2024, "temp": 0.1},
            )
            logging.info(f"LLM reloaded for precise answers: {selected_model}")
            return gr.update(value="Antworte möglichst genau", interactive=True, elem_id="active-button"), gr.update(value="Antworte mit möglichst vielen Hinweisen und Ideen", interactive=True, elem_id="inactive-button")
        elif option == "Antworte mit möglichst vielen Hinweisen und Ideen":
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20})
            logging.info("RAG parameters updated: k=20, temp=0.6. Reloading model...")
            llm = MLXPipeline.from_model_id(
                selected_model,
                pipeline_kwargs={"max_tokens": 2024, "temp": 0.6},
            )
            logging.info(f"LLM reloaded for creative answers: {selected_model}")
            return gr.update(value="Antworte möglichst genau", interactive=True, elem_id="inactive-button"), gr.update(value="Antworte mit möglichst vielen Hinweisen und Ideen", interactive=True, elem_id="active-button")

        # Recreate the chain
        doc_chain = create_stuff_documents_chain(llm, prompt)
        
        chain = create_retrieval_chain(retriever, doc_chain)
        print("bla")
    except Exception as e:
        logging.error(f"Error while updating RAG parameters, reloading documents, or rerunning functions: {e}")

# Function to log and save evaluation, ensuring it can only be done once
evaluation_done = False  # Global flag to track if evaluation has been done

def log_evaluation(evaluation):
    global evaluation_done
    if evaluation_done:
        logging.warning("Evaluation already submitted. Ignoring additional evaluations.")
        return gr.update(interactive=False), gr.update(interactive=False)
    try:
        # Log the evaluation
        logging.info(f"User evaluation: {'+1' if evaluation == 1 else '-1'}")

        # Save the evaluation to a CSV file
        with open("evaluation_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([evaluation])
        logging.info("Evaluation saved successfully.")

        # Mark evaluation as done
        evaluation_done = True

        # Update button colors
        if evaluation == 1:
            return gr.update(interactive=False, elem_id="thumbs-up-active"), gr.update(interactive=False, elem_id="thumbs-down-inactive")
        else:
            return gr.update(interactive=False, elem_id="thumbs-up-inactive"), gr.update(interactive=False, elem_id="thumbs-down-active")
    except Exception as e:
        logging.error(f"Failed to save evaluation: {e}")
        return gr.update(interactive=False), gr.update(interactive=False)

# Directory to store uploaded files
uploaded_files_dir = "./uploaded_documents"
Path(uploaded_files_dir).mkdir(parents=True, exist_ok=True)

# Function to list already loaded files
def list_loaded_files():
    try:
        files = [file.name for file in Path(uploaded_files_dir).glob("*.pdf")]
        if not files:
            logging.info("No files have been uploaded yet.")
            return "No files uploaded yet."
        logging.info(f"Loaded files: {files}")
        return "\n".join(files)
    except Exception as e:
        logging.error(f"Error listing loaded files: {e}")
        return "Error listing files."

# Function to handle file uploads
def upload_files(files):
    try:
        if not files:
            logging.warning("No files were provided for upload.")
            return "No files selected."

        uploaded_file_names = []
        for temp_file in files:
            # Gradio provides a temporary file object with a 'name' attribute holding the path
            temp_file_path = temp_file.name
            # Use the original filename if available, otherwise generate one
            original_filename = Path(temp_file_path).name # Gradio often uses generic temp names, consider extracting original if needed elsewhere
            
            # Define the destination path using the original filename or a default
            # For simplicity, we'll use the temp name's base name, but ideally, Gradio might provide the original name
            # Let's assume the temp file name is sufficient for now or use a fixed name if needed.
            # Using the name attribute directly might give a path, let's extract the base name.
            destination_filename = Path(temp_file_path).name # Or use a more robust way to get original name if available
            destination = Path(uploaded_files_dir) / destination_filename

            # Read from the temporary file path and write to the destination
            with open(temp_file_path, "rb") as infile, open(destination, "wb") as outfile:
                outfile.write(infile.read())
            
            logging.info(f"File saved to: {destination}")
            uploaded_file_names.append(destination_filename)

        return f"Files uploaded successfully: {', '.join(uploaded_file_names)}"
    except AttributeError as e:
        logging.error(f"Error accessing file properties: {e}. Check Gradio version compatibility or file object structure.")
        return "Error processing uploaded files (AttributeError)."
    except Exception as e:
        logging.error(f"Error uploading files: {e}")
        # Provide more specific error feedback if possible
        return f"Error uploading files: {e}"

# Function to load all uploaded files for RAG processing
def load_uploaded_files_for_rag():
    global vectorstore, retriever, chain # Ensure global variables are modified
    try:
        logging.info("Loading uploaded files for RAG processing...")
        # Use the dedicated directory for uploaded files
        documents = load_all_documents(uploaded_files_dir)
        if not documents:
            logging.warning("No documents were loaded from the uploaded files directory.")
            # Keep existing vectorstore if no new files are loaded? Or clear it?
            # For now, return a message indicating no new files processed.
            return "No new documents found in the upload directory to process."

        logging.info(f"Loaded {len(documents)} documents from {uploaded_files_dir}.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        split_docs = text_splitter.split_documents(documents)
        if not split_docs:
            logging.error("No chunks were created from uploaded files.")
            return "Failed to split uploaded documents into chunks."
        logging.info(f"Split uploaded documents into {len(split_docs)} chunks.")

        # Append document name and page number to each chunk
        for doc in split_docs:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "Unknown Page")
            doc.page_content += f" ({source}, Page {page})"

        # Recreate vectorstore with the new documents
        # Consider adding to existing store vs. replacing
        # Current implementation replaces the vectorstore
        logging.info("Recreating vectorstore with uploaded documents...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma.from_documents(split_docs, embeddings)

        # Recreate the retriever and chain with the new vectorstore
        # Use the currently selected RAG parameters (k value)
        current_k = retriever.search_kwargs.get("k", 8) # Get current k or default to 8
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": current_k})
        
        # Recreate the chain with the updated retriever
        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)

        logging.info("Vectorstore and RAG chain updated with uploaded files.")
        return f"Uploaded files ({len(documents)} documents, {len(split_docs)} chunks) successfully processed for RAG."
    except Exception as e:
        logging.error(f"Error loading uploaded files for RAG processing: {e}")
        return f"Error loading uploaded files for RAG processing: {e}"

# Gradio app
if __name__ == "__main__":
    folder_path = "./documents"
    document = load_all_documents(folder_path)
    if not document:
        logging.error("No documents were loaded. Please check the folder path and document files.")
    else:
        logging.info(f"{len(document)} documents loaded successfully.")

        try:
            documents = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            ).split_documents(document)

            if not documents:
                logging.error("No chunks were created. Please check the document processing logic.")
            else:
                logging.info(f"{len(documents)} chunks created successfully.")

                # Append document name and page number to each chunk
                for doc in documents:
                    source = doc.metadata.get("source", "Unbekanntes Dokument")
                    page = doc.metadata.get("page", "Unbekannte Seite")
                    doc.page_content += f" ({source}, Seite {page})"

                

        except Exception as e:
            logging.error(f"Error during document splitting: {e}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vectorstore = Chroma.from_documents(documents, embeddings)
    # Set the default retriever parameters
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    # Update retriever parameters based on user choice
    def update_retriever_parameters(choice, input):
        if choice == "Antworte möglichst genau":
            retriever.search_kwargs["k"] = 8
            logging.info("Retriever parameter updated: k=8 for precise answers")
        elif choice == "Antworte mit möglichst vielen Hinweisen und Ideen":
            retriever.search_kwargs["k"] = 20
            logging.info("Retriever parameter updated: k=20 for more creative answers")

    template = """INSTRUKTIONEN: Du musst nur auf Deutsch antworten.
    Du bist ein hilfreicher KI-Agent. Ich bin ein Sozialarbeiter im Einarbeitungsprozess und arbeite bei der Sozialhilfe in der Schweiz. 
    Bitte Suche in den dir zu zurverfügunggestellen Dokumenten und gibt mir möglichst genaue und hilfreiche Antworten auf meine Frage. 
    Wenn es keine Hinweise in den Dokumenten gibt, sage mir, dass ich die Frage nicht beantworten kann.
    Gib mir immer eine Quellenangabe deiner Antwort (zum Beispiel "Dokument 1, , Seite 3")
    FRAGE: {input} 
    KONTEXT: {context} 
    ANTWORT:"""

    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    
    
    with gr.Blocks(css=".gradio-container { font-size: 6px; }") as app:
        gr.Markdown("# Suchassistent für Sozialarbeitende")
        query_input = gr.Textbox(label="Formuliere deine Frage", placeholder="Was möchtest du wissen?", lines=2)

        response_output = gr.Textbox(label="Antwort", interactive=False)

        # File upload and listing section
        gr.Markdown("### Dateien hochladen und anzeigen:")
        with gr.Row():
            file_upload = gr.File(label="Lade PDF-Dateien hoch", file_types=[".pdf"], file_count="multiple")
            upload_button = gr.Button("Hochladen")
            list_files_button = gr.Button("Geladene Dateien anzeigen")
        uploaded_files_output = gr.Textbox(label="Geladene Dateien", interactive=False)

        # Button to load uploaded files for RAG processing
        load_files_button = gr.Button("Geladene Dateien für RAG verarbeiten")
        load_files_output = gr.Textbox(label="Status", interactive=False)

        # Trigger file upload
        upload_button.click(
            upload_files,
            inputs=[file_upload],
            outputs=[uploaded_files_output],
        )

        # List already loaded files
        list_files_button.click(
            list_loaded_files,
            inputs=[],
            outputs=[uploaded_files_output],
        )

        # Load uploaded files for RAG processing
        load_files_button.click(
            load_uploaded_files_for_rag,
            inputs=[],
            outputs=[load_files_output],
        )

        # Add buttons for RAG parameter selection
        gr.Markdown("### Wähle eine Antwortstrategie:")
        with gr.Row():
            precise_button = gr.Button("Antworte möglichst genau", elem_id="inactive-button")
            creative_button = gr.Button("Antworte mit möglichst vielen Hinweisen und Ideen", elem_id="inactive-button")

        # Place buttons for query execution
        with gr.Row():
            ask_button = gr.Button("Ausführen", elem_id="ask-button")
            clear_button = gr.Button("Lösche die Antwort", elem_id="clear-button")

        # Add evaluation buttons
        gr.Markdown("### Bewerte die Antwort:")
        with gr.Row():
            thumbs_up_button = gr.Button("👍", elem_id="thumbs-up-inactive")
            thumbs_down_button = gr.Button("👎", elem_id="thumbs-down-inactive")

        # Trigger "Ausführen" button click on Enter key press
        query_input.submit(
            ask_question_with_chunks, 
            inputs=[query_input], 
            outputs=[response_output]
        )

        ask_button.click(
            ask_question_with_chunks, 
            inputs=[query_input], 
            outputs=[response_output]
        )
        clear_button.click(
            lambda: logging.info("Response cleared") or logging.getLogger().handlers[0].flush() or "", 
            inputs=[], 
            outputs=[response_output]
        )

        # Update RAG parameters on button click and visually indicate the active button
        precise_button.click(
            lambda: update_rag_parameters("Antworte möglichst genau"),
            inputs=[],
            outputs=[precise_button, creative_button],
        )
        creative_button.click(
            lambda: update_rag_parameters("Antworte mit möglichst vielen Hinweisen und Ideen"),
            inputs=[],
            outputs=[precise_button, creative_button],
        )

        # Log evaluation on thumbs up or down
        thumbs_up_button.click(
            lambda: log_evaluation(1),
            inputs=[],
            outputs=[thumbs_up_button, thumbs_down_button],
        )
        thumbs_down_button.click(
            lambda: log_evaluation(-1),
            inputs=[],
            outputs=[thumbs_up_button, thumbs_down_button],
        )

        # Example queries
        gr.Markdown("### Beispielanfragen:")
        gr.Markdown("Beispiel: \"Welche Unterlagen benötige ich für ein Gesuch, finanzielle Sozialhilfe beantrage?\"")
        gr.Markdown("Beispiel: \"Ist ein Ehepaar mit einer AHV Rente von 4000.- plus Pensionskasse Rente von 2000.- berechtigt Sozialhilfe zu beantragen, grundsätzlich?\"")
        gr.Markdown("Beispiel: \"Was muss ich beachten, wenn ich Sozialhilfe beantrage?\"")
        gr.Markdown("Beispiel: \"Wie lange dauert es, bis ich eine Antwort auf mein Gesuch erhalte?\"")
        

        # Corrected CSS to ensure font size is applied properly
        app.css += """
        #active-button {
            background-color: #4CAF50 !important; /* Green for active */
            color: white !important;
        }
        #inactive-button {
            background-color: #f1f1f1 !important; /* Gray for inactive */
            color: black !important;
        }
        #thumbs-up-active {
            background-color: #4CAF50 !important; /* Green for thumbs up */
            color: white !important;
        }
        #thumbs-down-active {
            background-color: #f44336 !important; /* Red for thumbs down */
            color: white !important;
        }
        #thumbs-up-inactive, #thumbs-down-inactive {
            background-color: #f1f1f1 !important; /* Gray for inactive */
            color: black !important;
        }
        .button-row button, .rag-options-row button {
            font-size: 10px;
            padding: 5px 10px;
        }
        """
    # Launch the app with explicit host, port, and public sharing enabled
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
