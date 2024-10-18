import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.pgvector import PGVector

def process_input(llm_type, input_type, pdf_files, urls, question):
    model_local = ChatOllama(model=llm_type)
    
    # Auswahl zwischen PDF oder URL Verarbeitung
    docs = []
    if input_type == "PDFs":
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)  # Verwende direkt den Dateipfad
            docs.extend(loader.load())
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
    if input_type == "PDFs":
        return gr.update(visible=True), gr.update(visible=False)
    elif input_type == "URLs":
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)  # Falls kein Typ ausgewählt ist

# Setup der Gradio UI
with gr.Blocks() as iface:
    llm_type = gr.Radio(choices=['llama3.1:8b','mistral:latest','JarvisAI:latest'], label="Wählen Sie eine LLM")
    input_type = gr.Radio(choices=["PDFs", "URLs"], label="Wählen Sie den Eingabetyp")

    pdf_files = gr.Files(label="Laden Sie Ihre PDF-Dateien hoch", file_types=[".pdf"], visible=False)
    url_input = gr.Textbox(label="Geben Sie URLs getrennt durch Zeilen ein", visible=False)
    
    question = gr.Textbox(label="Stellen Sie Ihre Frage")
    output = gr.Textbox(label="Antwort")

    # Dynamische Sichtbarkeitsaktualisierung
    input_type.change(fn=update_inputs, inputs=input_type, outputs=[pdf_files, url_input])

    # Button zur Verarbeitung der Anfrage
    submit_button = gr.Button("Frage stellen")

    # Eventhandling des Submit Button
    submit_button.click(
        fn=process_input, 
        inputs=[llm_type, input_type, pdf_files, url_input, question], 
        outputs=[output]
    )

    # Button zum Leeren der Gradio Komponenten (UI-Refresh)
    clear_button = gr.ClearButton(
        components=[
            llm_type,
            input_type,
            pdf_files,
            url_input,
            question,
            output
        ],
        value="Chat leeren"
        )
    
    #Eventhandling des ClearButton 
    clear_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),  # Wir geben zwei Updates zurück
        outputs=[pdf_files, url_input]  # Das sind die Ausgaben für pdf_files und url_input
    )

iface.launch()
