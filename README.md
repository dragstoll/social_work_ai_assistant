# Social Work AI Assistant

## Overview
The **Social Work AI Assistant** is an AI-powered assistant designed to help social workers (SW) at SW organisation navigate the complex regulations of Individual Financial Assistance (Individuelle Finanzhilfe). The assistant uses Retrieval-Augmented Generation (RAG) to provide accurate and reliable answers to user queries based on the provided documents. It also references the source of the information (e.g., document name, page, and section).

## Features
- **Document Loading**: Automatically loads all PDF documents from a specified folder for processing.
- **RAG Querying**: Uses a retrieval-augmented generation approach to answer user queries based on the loaded documents.
- **Answer Evaluation**: Allows users to evaluate the quality of the answers with thumbs up or down. Evaluations are logged and saved for analysis.
- **Customizable Query Parameters**:
  - **Precise Answers**: Focuses on the most relevant information (`k=8`, `temp=0.1`).
  - **Creative Answers**: Provides broader context with more clues (`k=20`, `temp=0.6`).
- **Chunk Saving**: Saves the document chunks used for answering queries in a JSON file for validation.
- **Session Logging**: Logs all user interactions and system responses in a log file.
- **Example Queries**: Displays example questions to guide users in formulating their queries.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the documents to be processed in the `./documents` folder.

## Usage
1. Run the Gradio app:
   ```bash
   python ai_agent_proSenectute_macos.py
   ```
2. Access the app in your browser at `http://0.0.0.0:7860`.

## Features in Detail

### Document Loading
The assistant automatically loads all PDF documents from the `./documents` folder. Each document is split into smaller chunks for efficient querying.

### Querying
Users can ask questions in natural language. The assistant processes the query using RAG and provides an answer along with references to the source documents.

### Evaluation
Users can evaluate the quality of the answers:
- **Thumbs Up (👍)**: Indicates a positive evaluation (`+1`).
- **Thumbs Down (👎)**: Indicates a negative evaluation (`-1`).

Evaluations are logged in a CSV file (`evaluation_log.csv`) for further analysis.

### Customizable Query Parameters
Users can choose between two query modes:
1. **Precise Answers**:
   - Focuses on the most relevant information.
   - Parameters: `k=8`, `temp=0.1`.
2. **Creative Answers**:
   - Provides broader context with more clues.
   - Parameters: `k=20`, `temp=0.6`.

### Chunk Saving
The assistant saves the document chunks used for answering queries in a JSON file (`retrieved_chunks.json`) for validation.

### Logging
All user interactions and system responses are logged in `gradio_assisstant_macos.log`.

## Example Queries
Here are some example questions you can ask the assistant:
1. **"Welche Unterlagen benötige ich für ein Gesuch, finanzielle Sozialhilfe beantrage?"**
2. **"Ist ein Ehepaar mit einer AHV Rente von 4000.- plus Pensionskasse Rente von 2000.- berechtigt Sozialhilfe zu beantragen, grundsätzlich?"**

## Development Notes
### To-Do List
- [x] Make Gradio app usable from the web for testing purposes.
- [x] Ensure all documents in the folder are loaded and processed.
- [x] Create a button for clearing the input.
- [x] Translate everything to German.
- [x] Instruct RAG to reference the document, page, and text section.
- [x] Show example questions.
- [x] Create options for precise or creative answers.
- [x] Define a template for the prompt.
- [x] Test example questions and save the output.
- [x] Save the used chunks in a file for validation.
- [x] Log every session in a log file.
- [ ] Make a version running on a Linux server.
- [ ] Analyze case examples for usable questions to ask.
- [ ] Create a Docker environment for deployment.
- [ ] Add a login feature for SW based on their AD credentials.

### Nice-to-Have Features
- Save queries/prompts for further use.
- Load additional documents dynamically.
- Link all related documents used for answering queries.
- Extend feedback collection with simple frequency statistics.

## Open Questions
1. **Hardware and Software Requirements**:
   - The solution should run on a Microsoft Windows 11 OS without a GPU.
   - A browser-based solution is preferred.
2. **Target Audience**:
   - Focus on new social workers with generic questions rather than experienced workers with specialized queries.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.



