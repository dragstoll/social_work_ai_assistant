from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Changed to OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # Changed to Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import gradio as gr
from pathlib import Path
import logging
import sys
import json
import csv  # Add CSV module for saving evaluations
import subprocess
import gc  # Add garbage collection

# Add timeout configuration
OLLAMA_TIMEOUT = 30.0  # seconds


# Define available models for Ollama (local)
def get_ollama_models():
    """
    Returns a list of model names pulled in Ollama (from 'ollama list').
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if parts:
                # Take the full model name (first column) including any colons or tags
                model_name = parts[0]
                models.append(model_name)
        logging.info(f"Found Ollama models: {models}")
        return models
    except subprocess.CalledProcessError as e:
        logging.error(f"Ollama command failed: {e}")
        logging.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else 'No output'}")
        return []
    except FileNotFoundError:
        logging.error("Ollama command not found. Is Ollama installed and in PATH?")
        return []
    except Exception as e:
        logging.error(f"Could not get Ollama models: {e}")
        return []

# Dynamically get available models
ollama_models = get_ollama_models()

# Add fallback models if none found
if not ollama_models:
    logging.warning("No Ollama models found. Adding common fallback models.")
    fallback_models = [
        "llama3.2",
        "llama3.1", 
        "llama2",
        "mistral",
        "codellama",
        "phi3",
        "gemma",
        "qwen2",
        "nomic-embed-text"
    ]
    ollama_models = fallback_models

embedding_models = ollama_models
llm_models = ollama_models

# Fallback defaults if none found
# DEFAULT_LLM_MODEL = llm_models[0] if llm_models else ""
# DEFAULT_EMBED_MODEL = embedding_models[0] if embedding_models else ""

# Store selected models globally
selected_llm_model = None
selected_embed_model = None

# Initialize global variables
retriever = None
llm = None
chain = None
documents = None
vectorstore = None
prompt = None
embeddings = None

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
    global llm, prompt, vectorstore

    if not query.strip():
        return "Bitte gib eine Frage ein."

    if llm is None or prompt is None:
        return "❌ Fehler: Bitte wähle zuerst die Modelle aus und klicke auf 'Modelle übernehmen'."

    if vectorstore is None:
        return "❌ Fehler: Keine Dokumente geladen. Bitte lade Dokumente aus dem ./documents Ordner oder lade neue Dateien hoch."

    try:
        logging.info(f"Processing query: {query}")

        # Always create a fresh retriever and chain for each query
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})  # Lower k for performance
        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)

        # Try to set a timeout if supported by Ollama (some versions support it)
        try:
            response = chain.invoke({"input": query})
        except Exception as e:
            logging.error(f"Ollama or chain invoke failed: {e}", exc_info=True)
            return f"❌ Fehler beim Modellaufruf: {e}"

        answer = response.get("answer", "Keine Antwort erhalten.")

        # Save the chunks to a file
        try:
            retrieved_documents = retriever.get_relevant_documents(query)
            save_chunks_to_file(retrieved_documents, "retrieved_chunks.json")
        except Exception as chunk_error:
            logging.warning(f"Could not save chunks: {chunk_error}")

        logging.info("Query processed successfully")
        # Explicitly delete objects to free memory
        del retriever
        del doc_chain
        del chain
        import gc; gc.collect()

        return answer

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        import gc; gc.collect()
        return f"❌ Fehler bei der Verarbeitung der Anfrage: {e}"

# # Load the Ollama LLM model immediately
# llm = Ollama(model=selected_llm_model)
# logging.info(f"LLM loaded for RAG: {selected_llm_model}")

# Function to update RAG parameters, reload documents, and rerun all necessary functions
def update_rag_parameters(option):
    global retriever, chain, llm, prompt, vectorstore

    # Check if models are selected
    if selected_llm_model is None or selected_embed_model is None:
        logging.error("Models not selected. Cannot update RAG parameters.")
        return gr.update(), gr.update()
    
    logging.info(f"Updating RAG parameters based on user selection: {option}")
    try:
        if vectorstore is None:
            logging.error("Vectorstore not initialized. Please select models first.")
            return gr.update(), gr.update()

        k = 4 if option == "Antworte möglichst genau" else 10  # Lower k for performance
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)

        # Do NOT delete retriever/chain here! They are needed for the next query.

        # Update button colors
        if option == "Antworte möglichst genau":
            return gr.update(value="Antworte möglichst genau", interactive=True, elem_id="active-button"), gr.update(value="Antworte mit möglichst vielen Hinweisen und Ideen", interactive=True, elem_id="inactive-button")
        elif option == "Antworte mit möglichst vielen Hinweisen und Ideen":
            return gr.update(value="Antworte möglichst genau", interactive=True, elem_id="inactive-button"), gr.update(value="Antworte mit möglichst vielen Hinweisen und Ideen", interactive=True, elem_id="active-button")
    except Exception as e:
        logging.error(f"Error while updating RAG parameters: {e}", exc_info=True)
        return gr.update(), gr.update()

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

# Clear response function
def clear_response():
    """Clear the response and clean up resources"""
    global evaluation_done
    logging.info("Clearing response and resetting evaluation state")
    evaluation_done = False
    import gc; gc.collect()
    return ""

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
    global vectorstore, retriever, chain, embeddings, documents
    
    # Check if models are selected
    if selected_llm_model is None or selected_embed_model is None:
        return "❌ Fehler: Bitte wähle zuerst die Modelle aus und klicke auf 'Modelle übernehmen'."
    
    if llm is None or embeddings is None:
        return "❌ Fehler: Modelle nicht initialisiert. Bitte klicke auf 'Modelle übernehmen'."
    
    try:
        logging.info("Loading uploaded files for RAG processing...")
        uploaded_documents = load_all_documents(uploaded_files_dir)
        if not uploaded_documents:
            logging.warning("No documents were loaded from the uploaded files directory.")
            return "⚠️ Keine neuen Dokumente im Upload-Verzeichnis gefunden."

        logging.info(f"Loaded {len(uploaded_documents)} documents from {uploaded_files_dir}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        split_docs = text_splitter.split_documents(uploaded_documents)
        if not split_docs:
            logging.error("No chunks were created from uploaded files.")
            return "❌ Fehler: Keine Chunks aus hochgeladenen Dateien erstellt."
        
        logging.info(f"Split uploaded documents into {len(split_docs)} chunks.")

        for doc in split_docs:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "Unknown Page")
            doc.page_content += f" ({source}, Page {page})"

        # Combine with existing documents if any
        if documents and len(documents) > 0:
            all_documents = documents + split_docs
        else:
            all_documents = split_docs
        
        documents = all_documents

        # Recreate vectorstore with all documents
        import chromadb
        client = chromadb.PersistentClient(path=".chroma_db")
        for collection in client.list_collections():
            client.delete_collection(collection.name)
            
        vectorstore = Chroma.from_documents(all_documents, embeddings)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})

        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)

        logging.info("Vectorstore and RAG chain updated with uploaded files.")
        return f"✅ Hochgeladene Dateien erfolgreich verarbeitet:\n📄 {len(uploaded_documents)} Dokumente, {len(split_docs)} Chunks\n🔄 Gesamt: {len(all_documents)} Chunks bereit für Abfragen."
        
    except Exception as e:
        logging.error(f"Error loading uploaded files for RAG processing: {e}")
        return f"❌ Fehler beim Laden der hochgeladenen Dateien: {e}"

# Gradio app
def set_models(selected_llm, selected_embed):
    global selected_llm_model, selected_embed_model, llm, embeddings, vectorstore, retriever, chain, prompt, documents
    
    if not selected_llm or not selected_embed:
        return "❌ Fehler: Bitte wähle beide Modelle aus."
    
    selected_llm_model = selected_llm
    selected_embed_model = selected_embed
    
    try:
        logging.info(f"Loading models - LLM: {selected_llm_model}, Embedding: {selected_embed_model}")
        
        # Initialize models without timeout to avoid hanging
        llm = Ollama(model=selected_llm_model)
        embeddings = OllamaEmbeddings(model=selected_embed_model)
        
        # Initialize prompt template
        template = """INSTRUKTIONEN: Du musst nur auf Deutsch antworten.
        Du bist ein hilfreicher KI-Agent. Ich bin ein Sozialarbeiter im Einarbeitungsprozess und arbeite bei der Sozialhilfe in der Schweiz. 
        Bitte Suche in den dir zu zurverfügunggestellen Dokumenten und gibt mir möglichst genaue und hilfreiche Antworten auf meine Frage. 
        Wenn es keine Hinweise in den Dokumenten gibt, sage mir, dass ich die Frage nicht beantworten kann.
        Gib mir immer eine Quellenangabe deiner Antwort (zum Beispiel "Dokument 1, , Seite 3")
        FRAGE: {input} 
        KONTEXT: {context} 
        ANTWORT:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Always try to create vectorstore if documents exist
        if documents and len(documents) > 0:
            try:
                # Clear existing vectorstore
                import chromadb
                client = chromadb.PersistentClient(path=".chroma_db")
                try:
                    for collection in client.list_collections():
                        client.delete_collection(collection.name)
                except Exception as cleanup_error:
                    logging.warning(f"Could not clean up old collections: {cleanup_error}")
                
                vectorstore = Chroma.from_documents(documents, embeddings)
                
                # Create initial retriever and chain
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})
                doc_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(retriever, doc_chain)
                
                logging.info("RAG system initialized successfully with existing documents.")
                return f"✅ Modelle erfolgreich geladen - LLM: {selected_llm_model}, Embedding: {selected_embed_model}\n📄 {len(documents)} Dokumente bereit für Abfragen."
            except Exception as e:
                logging.error(f"Error creating vectorstore: {e}")
                return f"⚠️ Modelle geladen, aber Fehler beim Erstellen der Vectorstore: {e}\nBitte versuche Dokumente erneut zu laden."
        else:
            logging.info("Models loaded, but no documents available yet.")
            # Only reset these if no documents
            chain = None
            retriever = None
            vectorstore = None
            return f"✅ Modelle erfolgreich geladen - LLM: {selected_llm_model}, Embedding: {selected_embed_model}\n⚠️ Keine Dokumente gefunden. Bitte lade Dokumente hoch oder prüfe den ./documents Ordner."
        
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return f"❌ Fehler beim Laden der Modelle: {e}\nBitte prüfe die Modellnamen und ob sie mit Ollama gepullt wurden."

if __name__ == "__main__":
    # Load documents at startup
    folder_path = "./documents"
    startup_documents = load_all_documents(folder_path)
    if not startup_documents:
        logging.warning("No documents were loaded from ./documents folder at startup.")
        documents = []
    else:
        logging.info(f"{len(startup_documents)} documents loaded successfully from ./documents folder.")

        try:
            documents = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            ).split_documents(startup_documents)

            if not documents:
                logging.error("No chunks were created from startup documents.")
                documents = []
            else:
                logging.info(f"{len(documents)} chunks created successfully from startup documents.")

                for doc in documents:
                    source = doc.metadata.get("source", "Unbekanntes Dokument")
                    page = doc.metadata.get("page", "Unbekannte Seite")
                    doc.page_content += f" ({source}, Seite {page})"

        except Exception as e:
            logging.error(f"Error during document splitting: {e}")
            documents = []

    template = """INSTRUKTIONEN: Du musst nur auf Deutsch antworten.
    Du bist ein hilfreicher KI-Agent. Ich bin ein Sozialarbeiter im Einarbeitungsprozess und arbeite bei der Sozialhilfe in der Schweiz. 
    Bitte Suche in den dir zu zurverfügunggestellen Dokumenten und gibt mir möglichst genaue und hilfreiche Antworten auf meine Frage. 
    Wenn es keine Hinweise in den Dokumenten gibt, sage mir, dass ich die Frage nicht beantworten kann.
    Gib mir immer eine Quellenangabe deiner Antwort (zum Beispiel "Dokument 1, , Seite 3")
    FRAGE: {input} 
    KONTEXT: {context} 
    ANTWORT:"""

    # Do NOT instantiate llm, embeddings, doc_chain, chain here
    # Wait for user to click "Modelle übernehmen" in Gradio UI

    with gr.Blocks(css=".gradio-container { font-size: 12px; }") as app:
        gr.Markdown("# Suchassistent für Sozialarbeitende")

        # Model selection UI (dynamic from ollama list)
        gr.Markdown("### 1️⃣ Wähle die Modelle für LLM und Embedding:")
        
        # Add status message about available models
        if not get_ollama_models():
            gr.Markdown("⚠️ **Hinweis**: Keine Ollama-Modelle automatisch erkannt. Fallback-Modelle werden angezeigt. Stelle sicher, dass Ollama läuft und Modelle installiert sind.")
        
        with gr.Row():
            llm_dropdown = gr.Dropdown(
                label="LLM Modell", 
                choices=llm_models, 
                value=llm_models[0] if llm_models else None,
                allow_custom_value=True,
                info="Wähle ein Modell oder gib einen eigenen Namen ein"
            )
            embed_dropdown = gr.Dropdown(
                label="Embedding Modell", 
                choices=embedding_models, 
                value=embedding_models[0] if embedding_models else None,
                allow_custom_value=True,
                info="Wähle ein Modell oder gib einen eigenen Namen ein"
            )
            set_model_button = gr.Button("Modelle übernehmen", variant="primary")
        
        model_status = gr.Textbox(
            label="Status", 
            value="⚠️ Bitte wähle Modelle aus und klicke auf 'Modelle übernehmen'", 
            interactive=False
        )

        # Add button to refresh model list
        with gr.Row():
            refresh_models_button = gr.Button("🔄 Modelle neu laden", size="sm")
        
        def refresh_models():
            global ollama_models, llm_models, embedding_models
            ollama_models = get_ollama_models()
            if not ollama_models:
                ollama_models = [
                    "llama3.2", "llama3.1", "llama2", "mistral", "codellama", 
                    "phi3", "gemma", "qwen2", "nomic-embed-text"
                ]
            llm_models = ollama_models
            embedding_models = ollama_models
            logging.info(f"Refreshed models: {ollama_models}")
            return (
                gr.update(choices=llm_models, value=llm_models[0] if llm_models else None),
                gr.update(choices=embedding_models, value=embedding_models[0] if embedding_models else None),
                f"🔄 Modelle neu geladen: {len(ollama_models)} gefunden"
            )
        
        refresh_models_button.click(
            refresh_models,
            inputs=[],
            outputs=[llm_dropdown, embed_dropdown, model_status]
        )

        set_model_button.click(
            set_models,
            inputs=[llm_dropdown, embed_dropdown],
            outputs=[model_status],
        )

        gr.Markdown("### 2️⃣ Stelle deine Frage:")
        query_input = gr.Textbox(label="Formuliere deine Frage", placeholder="Was möchtest du wissen?", lines=2)

        response_output = gr.Textbox(label="Antwort", interactive=False, lines=10)

        # Add buttons for RAG parameter selection
        gr.Markdown("### 3️⃣ Wähle eine Antwortstrategie:")
        with gr.Row():
            precise_button = gr.Button("Antworte möglichst genau", elem_id="active-button")
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
            clear_response,
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
       


