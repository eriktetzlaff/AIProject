import gradio as gr
import PyPDF2

# Funktion, um den Text aus dem PDF zu extrahieren
def extract_pdf_text(pdf_file):
    # Öffne das PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    # Extrahiere Text aus jeder Seite
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text() + "\n"
    return text

# Gradio-Interface
with gr.Blocks() as demo:
    # Lade PDF-Datei über File Explorer
    pdf_input = gr.File(label="Lade ein PDF hoch", file_types=[".pdf"])
    
    # Zeige den extrahierten Text an
    output_text = gr.Textbox(label="Extrahierter Text aus PDF")
    
    # Verbindung der PDF-Datei mit der Funktion zur Textausgabe
    pdf_input.change(fn=extract_pdf_text, inputs=pdf_input, outputs=output_text)

# Starte die Gradio App
demo.launch()
