# Social Work AI Assistant

## Overview
The **Social Work AI Assistant** is an AI-powered assistant designed to help social workers (SW) at SW organisation navigate the complex regulations of Individual Financial Assistance (Individuelle Finanzhilfe). The assistant uses Retrieval-Augmented Generation (RAG) to provide accurate and reliable answers to user queries based on the provided documents. It also references the source of the information (e.g., document name, page, and section).

## Features
- **Document Loading**: Automatically loads all PDF documents from a specified folder (`./documents`) at startup.
- **File Upload & Management**:
    - Allows users to upload additional PDF documents directly through the interface.
    - Lists currently uploaded documents.
    - Processes uploaded documents to update the RAG knowledge base on demand.
- **RAG Querying**: Uses a retrieval-augmented generation approach to answer user queries based on the loaded documents (initial and uploaded).
- **Answer Evaluation**: Allows users to evaluate the quality of the answers with thumbs up or down. Evaluations are logged and saved to `evaluation_log.csv` for analysis.
- **Customizable RAG Strategy**:
  - **Precise Answers**: Focuses on the most relevant information by adjusting retriever settings (`k=8`).
  - **Creative Answers**: Provides broader context with more clues by adjusting retriever settings (`k=20`).
- **Model Selection**: Uses local Ollama models for both embeddings and LLM inference (see below for setup).
- **Retrieved Chunk Saving**: Saves the document chunks retrieved for the last query in JSON (`retrieved_chunks.json`) and CSV (`retrieved_chunks.csv`) formats for validation.
- **Session Logging**: Logs all user interactions, system responses, and errors in a log file (`gradio_assisstant_macos.log`).
- **Example Queries**: Displays example questions to guide users in formulating their queries.

## Installation

### 0. Install Python and pip (if not already installed)

If your computer does **not** have Python or pip installed, follow these steps:

- **Windows**:
  1. Download Python from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
  2. Run the installer and ensure you check the box "Add Python to PATH".
  3. After installation, open Command Prompt and verify:
     ```bash
     python --version
     pip --version
     ```

- **macOS**:
  - Python 3 is often pre-installed. If not, install via [Homebrew](https://brew.sh/):
    ```bash
    brew install python
    ```
  - Verify installation:
    ```bash
    python3 --version
    pip3 --version
    ```

- **Linux**:
  - Install Python and pip using your package manager, e.g.:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```
  - Verify installation:
    ```bash
    python3 --version
    pip3 --version
    ```

> If you have both `python` and `python3` on your system, use `python3` and `pip3` for commands.

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (for local LLM and embedding models)
Ollama is required for both language model inference and embeddings.

- **macOS / Linux / Windows**:  
  Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

- **Start Ollama**:  
  After installation, start the Ollama server:
  ```bash
  ollama serve
  ```

### 3. Download the required models
You need to pull the models used for inference and embeddings.  
For CPU-only usage, choose smaller models.

#### Example (as used in `ai_agent_socialwork_ollama.py`):
```bash
# For LLM inference (choose one or more, smaller models recommended for CPU)
ollama pull gemma:4b-it-q4_K_M
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M

# For embeddings (smaller multilingual model)
ollama pull jeffh/intfloat-multilingual-e5-large-instruct:q8_0
```
> **Tip:** For CPU-only systems, use models with `q4`, `q8`, or other quantized versions (smaller, less memory usage).  
> See [Ollama Model Library](https://ollama.com/library) for more options.

### 4. Prepare document folders
- Place initial documents to be processed in the `./documents` folder.
- Create an empty `./uploaded_documents` folder for user uploads.

## Usage

### 1. Start Ollama server (if not already running)
```bash
ollama serve
```

### 2. Run the Gradio app

If you installed Python as above, use the appropriate command for your system:

- **Windows**:
  ```bash
  python ai_agent_socialwork_ollama.py
  ```
- **macOS/Linux**:
  ```bash
  python3 ai_agent_socialwork_ollama.py
  ```

### 3. Access the app in your browser
Go to [http://0.0.0.0:7860](http://0.0.0.0:7860).

### 4. Workflow
- **Optional:** Upload additional PDF documents using the "Lade PDF-Dateien hoch" section.
- **Optional:** Click "Geladene Dateien für RAG verarbeiten" to include the uploaded documents in the knowledge base for subsequent queries.
- Select an answer strategy ("Antworte möglichst genau" or "Antworte mit möglichst vielen Hinweisen und Ideen").
- Enter your query and click "Ausführen" or press Enter.
- Evaluate the answer using the 👍 or 👎 buttons.

## CPU-Only Setup Instructions

If you do **not** have a GPU, you can still run the assistant using smaller quantized models:

1. **Choose small models for Ollama**  
   - Use models with names ending in `q4`, `q8`, or similar (quantized).
   - Example: `gemma:4b-it-q4_K_M`, `jeffh/intfloat-multilingual-e5-large-instruct:q8_0`.

2. **Pull only the required models**  
   ```bash
   ollama pull gemma:4b-it-q4_K_M
   ollama pull jeffh/intfloat-multilingual-e5-large-instruct:q8_0
   ```

3. **Edit `ai_agent_socialwork_ollama.py` if needed**  
   - Make sure the `available_models` and embedding model names match what you have downloaded.

4. **Start Ollama and run the app**  
   ```bash
   ollama serve
   python ai_agent_socialwork_ollama.py
   ```

## Features in Detail

### Document Loading & Management
- **Initial Loading**: The assistant loads all PDF documents from the `./documents` folder upon starting.
- **User Uploads**: Users can upload one or more PDF files via the interface. These files are saved to the `./uploaded_documents` directory.
- **Processing Uploads**: Clicking "Geladene Dateien für RAG verarbeiten" triggers the loading, splitting, embedding, and indexing of documents from the `./uploaded_documents` folder, replacing the existing knowledge base with one based *only* on the uploaded files.

### Querying
Users can ask questions in natural language. The assistant processes the query using the configured RAG strategy and provides an answer along with references to the source documents.

### Evaluation
Users can evaluate the quality of the answers:
- **Thumbs Up (👍)**: Indicates a positive evaluation (`+1`).
- **Thumbs Down (👎)**: Indicates a negative evaluation (`-1`).

Evaluations are logged in `gradio_assisstant_macos.log` and saved to `evaluation_log.csv`. Each answer can only be evaluated once per session.

### Customizable RAG Strategy
Users can choose between two modes which adjust retriever parameters and LLM temperature, and reload the RAG components:
1. **Precise Answers**: `k=8`, `temp=0.1`.
2. **Creative Answers**: `k=20`, `temp=0.6`.
The active strategy button is highlighted.

### Retrieved Chunk Saving
After a query, the assistant saves the document chunks retrieved by the retriever in `retrieved_chunks.json` (with metadata) and `retrieved_chunks.csv` (content only) for validation purposes.

### Logging
All user interactions, system responses, chosen strategies, evaluations, and errors are logged in `gradio_assisstant_macos.log`.

## Example Queries
Here are some example questions you can ask the assistant:
1. **"Welche Unterlagen benötige ich für ein Gesuch, finanzielle Sozialhilfe beantrage?"**
2. **"Ist ein Ehepaar mit einer AHV Rente von 4000.- plus Pensionskasse Rente von 2000.- berechtigt Sozialhilfe zu beantragen, grundsätzlich?"**
3. **"Was muss ich beachten, wenn ich Sozialhilfe beantrage?"**
4. **"Wie lange dauert es, bis ich eine Antwort auf mein Gesuch erhalte?"**

## Development Notes
### To-Do List
- [x] Make Gradio app usable from the web for testing purposes.
- [x] Ensure all documents in the folder are loaded and processed (initial load).
- [x] Create a button for clearing the input.
- [x] Translate everything to German.
- [x] Instruct RAG to reference the document, page, and text section (partially done, needs refinement).
- [x] Show example questions.
- [x] Create options for precise or creative answers (implemented with RAG reload).
- [x] Define a template for the prompt.
- [x] Test example questions and save the output.
- [x] Save the used chunks in a file for validation (implemented as retrieved chunks).
- [x] Log every session in a log file.
- [x] Load additional documents dynamically (implemented via upload and process).
- [ ] Make a version running on a Linux server.
- [ ] Analyze case examples for usable questions to ask.
- [ ] Create a Docker environment for deployment.
- [ ] Add a login feature for SW based on their AD credentials.
- [ ] Improve handling of uploaded documents (e.g., add to existing index instead of replacing).
- [ ] Expose model selection in the UI.

### Nice-to-Have Features
- Save queries/prompts for further use.
- Link all related documents used for answering queries.
- Extend feedback collection with simple frequency statistics.
- Visualize retrieved chunks or highlight relevant sections in source documents.

## Open Questions
1. **Hardware and Software Requirements**:
   - The solution should run on a Microsoft Windows 11 OS without a GPU. (Current macOS version uses MLX which requires Apple Silicon).
   - A browser-based solution is preferred (achieved with Gradio).
2. **Target Audience**:
   - Focus on new social workers with generic questions rather than experienced workers with specialized queries.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.



