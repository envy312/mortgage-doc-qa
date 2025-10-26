import gradio as gr
from gradio_pdf import PDF
import requests
import os

API_URL = "http://localhost:8000"

def process_pdf(file):
    if file is None:
        return "Please upload a PDF file.", None
    
    try:
        files = {"file": open(file.name, "rb")}
        response = requests.post(f"{API_URL}/api/v1/documents/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            status_msg = f"""Successfully Processed:
- File: {stats['filename']}
- Pages: {stats['total_pages']}
- Documents: {stats['documents_found']}
- Types: {', '.join(stats['document_types'])}
- Time: {stats['processing_time']}
"""
            return status_msg, file.name  
        else:
            return f" Error: {response.json().get('detail', 'Unknown error')}", None
    except Exception as e:
        return f" Error: {str(e)}", None

def ask_question(message, history):
    if not message:
        return history
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json={"query": message, "top_k": 5}
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            
            sources_text = "\n\n**Sources:**\n"
            for src in data.get('sources', []):
                sources_text += f"• {src['doc_type']} (Pages {src['pages']}) - {src['relevance']}\n"
            
            full_response = answer + sources_text
            history.append([message, full_response])
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            history.append([message, f" {error_msg}"])
    except Exception as e:
        history.append([message, f" Error: {str(e)}"])
    
    return history

with gr.Blocks(title="Mortgage Document Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Mortgage Document Q&A System")
    gr.Markdown("### Upload PDFs and ask questions with AI-powered analysis")
    
    with gr.Row():
        with gr.Column(scale=2):
            pdf_viewer = PDF(label="PDF Viewer", height=600)

        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Document")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary", size="lg")
            status = gr.Textbox(label="Status", lines=8)
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Ask a question about your documents...", label="Your Question")
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")
    
    
    process_btn.click(process_pdf, inputs=[pdf_input], outputs=[status, pdf_viewer])
    
    submit.click(ask_question, inputs=[msg, chatbot], outputs=[chatbot]).then(lambda: "", outputs=[msg])
    msg.submit(ask_question, inputs=[msg, chatbot], outputs=[chatbot]).then(lambda: "", outputs=[msg])
    clear.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)