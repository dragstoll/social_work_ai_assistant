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

# Function to handle user queries
def ask_question(query):
    response = chain.invoke({"input": query})
    return response["answer"]

# Function to handle user queries and return only the answer
def ask_question_with_chunks(query):
    response = chain.invoke({"input": query})
    return response["answer"]

# Gradio app
if __name__ == "__main__":
    folder_path = "./documents"
    document = load_all_documents(folder_path)
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    ).split_documents(document)

    # Append document name and page number to each chunk
    for doc in documents:
        source = doc.metadata.get("source", "Unbekanntes Dokument")
        page = doc.metadata.get("page", "Unbekannte Seite")
        doc.page_content += f" ({source}, Seite {page})"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    llm = MLXPipeline.from_model_id(
        # "mlx-community/QwQ-32B-4bit",
        #   "mlx-community/gemma-3-27b-it-4bit",
    #    "mlx-community/Qwen2.5-32B-Instruct-4bit",
    # "mlx-community/Mistral-Large-Instruct-2407-4bit",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        pipeline_kwargs={"max_tokens": 2024, "temp": 0.1},
    )

    template = """INSTRUKTIONEN: Du musst nur auf Deutsch antworten.
    Du bist ein hilfreicher KI-Agent. Ich bin ein Sozialarbeiter im Einarbeitungsprozess und arbeite bei der Pro Senectute Aargau. 
    Bei der Pro Senectute haben wir das Angebot der Individuellen Finanzhilfe. 
    Bitte Suche in den dir zu zurverfügunggestellen Dokumenten und gibt mir möglichst genaue und hilfreiche Antworten auf meine Frage. 
    
    Gib mir immer eine Quellenangabe deiner Antwort (zum Beispiel "Dokument 1, , Seite 3")
    FRAGE: {input} 
    KONTEXT: {context} 
    ANTWORT:"""

    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    with gr.Blocks(css=".gradio-container { font-size: 6px; }") as app:
        gr.Markdown("# Pro Senectute Suchassistent zur individuellen Finanzhilfe für Sozialarbeitende")
        query_input = gr.Textbox(label="Formuliere deine Frage", placeholder="Was möchtest du wissen?", lines=2)

        response_output = gr.Textbox(label="Antwort", interactive=False)

        # Place buttons on the same line
        with gr.Row(elem_id="button-row"):
            ask_button = gr.Button("Ausführen", elem_id="ask-button")
            clear_button = gr.Button("Lösche die Antwort", elem_id="clear-button")

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
        clear_button.click(lambda: "", inputs=[], outputs=[response_output])

        # Example queries
        gr.Markdown("Beispiel: \"Welche Unterlagen benötige ich für ein Gesuch, wenn ich Nebenkosten beantrage?\"", elem_id="example1")
        gr.Markdown("Beispiel: \"Was muss ich beachten, wenn ich ein Hörgerät beantrage?\"", elem_id="example2")
        gr.Markdown("Beispiel: \"Ist ein Ehepaar mit einer AHV Rente von 4000.- plus Pensionskasse Rente von 2000.- berechtigt individuelle Finanzhilfe zu beantragen, grundsätzlich?\"", elem_id="example3")
        gr.Markdown("Beispiel: \"Ein Klient hat eine Rechnung von 3000.- für Nebenkosten Wohnnebenkosten, diese Rechnung wurde vor 6 Monaten bezahlt, kann ich diese Gelder über IF beantragen?\"", elem_id="example4")
        gr.Markdown("Beispiel: \"ESchreibe mir ein Gesuch, für eine Brille, das Gesuch beinhaltet, Ausgangslage, Ziel, Subsidiarität, Partizipation?\"", elem_id="example5")

        # Corrected CSS to ensure font size is applied properly
        app.css += """
        example1, #example2, #example3 {
            font-size: 8px !important;
            line-height: 1.2;
        }
        .button-row button {
            font-size: 10px;
            padding: 5px 10px;
        }
        """
    # Launch the app with explicit host, port, and public sharing enabled
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
