import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import PGVector
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import csv
import docx

# Funktion zum Laden von Word-Dokumenten
def load_word_file(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Funktion zum Laden von CSV-Dateien
def load_csv_file(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        csv_data = "\n".join([", ".join(row) for row in reader])
    return csv_data

# Funktion zur Verarbeitung der User-Eingaben
def process_input(llm_type, input_type, files, urls, question):
    model_local = ChatOllama(model=llm_type)
    
    # Auswahl zwischen Datei oder URL Verarbeitung
    docs = []
    if input_type == "Dateien":
        for uploaded_file in files:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(uploaded_file)  # Verwende den Dateipfad
                docs.extend(loader.load())
            elif uploaded_file.name.endswith(".docx"):
                docs.append("\n".join(load_word_file(uploaded_file)))
            elif uploaded_file.name.endswith(".csv"):
                docs.append(load_csv_file(uploaded_file))
    elif input_type == "URLs":
        urls_list = urls.split("\n")
        docs = [WebBaseLoader(url).load() for url in urls_list]
        docs = [item for sublist in docs for item in sublist]

    # Textsplitter Einrichtung (Chunks Config)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # Dateneinleitung in die Postgres Vektordatenbank
    vectorstore = PGVector.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text:latest'),
        connection_string="postgresql+psycopg2://postgres:password@localhost:5432/vector_db",
        use_jsonb=True,
    )
    retriever = vectorstore.as_retriever()

    # Prompt Template und Kette erstellen
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Gradio Interface Definition
def update_inputs(input_type):
    if input_type == "Dateien":
        return gr.update(visible=True), gr.update(visible=False)
    elif input_type == "URLs":
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)  # Falls kein Typ ausgewählt ist

# Setup der Gradio UI
with gr.Blocks() as iface:
    llm_type = gr.Radio(choices=['llama3.1:8b', 'mistral:latest', 'JarvisAI:latest'], label="Wählen Sie eine LLM")
    input_type = gr.Radio(choices=["Dateien", "URLs"], label="Wählen Sie den Eingabetyp")

    # Erweiterung der Dateiauswahl für PDF, DOCX und CSV
    files = gr.Files(label="Laden Sie Ihre Dateien hoch (PDF, DOCX, CSV)", file_types=['.pdf', '.docx', '.csv'], visible=False)
    url_input = gr.Textbox(label="Geben Sie URLs getrennt durch Zeilen ein", visible=False)
    
    question = gr.Textbox(label="Stellen Sie Ihre Frage")
    output = gr.Textbox(label="Antwort")

    # Dynamische Sichtbarkeitsaktualisierung
    input_type.change(fn=update_inputs, inputs=input_type, outputs=[files, url_input])

    # Button zur Verarbeitung der Anfrage
    submit_button = gr.Button("Frage stellen")

    # Eventhandling des Submit Button
    submit_button.click(
        fn=process_input, 
        inputs=[llm_type, input_type, files, url_input, question], 
        outputs=[output]
    )

    # Button zum Leeren der Gradio Komponenten (UI-Refresh)
    clear_button = gr.ClearButton(
        components=[
            llm_type,
            input_type,
            files,
            url_input,
            question,
            output
        ],
        value="Chat leeren"
    )
    
    # Eventhandling des ClearButton 
    clear_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),  # Wir geben zwei Updates zurück
        outputs=[files, url_input]  # Das sind die Ausgaben für files und url_input
    )

iface.launch()
