import gradio as gr
import sqlite3
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import json
import os

# تنظیم مدل Ollama
llm = OllamaLLM(model="llama3.2", temperature=0.7)

# ایجاد پایگاه داده SQLite برای ذخیره‌سازی تاریخچه مکالمات
conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
cursor = conn.cursor()

# ایجاد جدول برای ذخیره تاریخچه مکالمات
cursor.execute('''
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT
)
''')

# تابع برای ذخیره تاریخچه مکالمات در پایگاه داده
def save_to_db(user_input, bot_response):
    cursor.execute('''
    INSERT INTO conversation_history (user_input, bot_response) 
    VALUES (?, ?)
    ''', (user_input, bot_response))
    conn.commit()

# تابع برای بازیابی تاریخچه مکالمات از پایگاه داده
def get_conversation_history():
    cursor.execute('SELECT user_input, bot_response FROM conversation_history')
    rows = cursor.fetchall()
    history = ""
    for row in rows:
        history += f"User: {row[0]}\nBot: {row[1]}\n\n"
    return history

# تعریف پرامپت برای مدل
BASE_PROMPT = """
You are a helpful and conversational AI assistant. Provide clear, accurate, and friendly responses to the user's input. If the question is vague, ask for clarification. Respond in the same language as the input. If a file is uploaded, analyze its content and respond accordingly.

Conversation history:
{history}

User input: {user_input}

File content (if any): {file_content}

Response:
"""

# ایجاد پرامپت با LangChain
prompt = PromptTemplate(
    input_variables=["history", "user_input", "file_content"],
    template=BASE_PROMPT
)

# ایجاد Chain برای استفاده از مدل و حافظه
llm_chain = LLMChain(llm=llm, prompt=prompt)

# تابع برای خواندن محتوای فایل
def read_file(file):
    if file is None:
        return ""
    
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".txt" or file_extension == ".py":
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == ".csv":
            df = pd.read_csv(file.name)
            return df.to_string()
        elif file_extension == ".json":
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        else:
            return "❌ Unsupported file format. Please upload a .txt, .py, .csv, or .json file."
    except Exception as e:
        return f"❌ Error reading file: {e}"

# متغیر برای ردیابی وضعیت پردازش
processing_state = {"is_processing": False, "cancel_requested": False}

# تابع برای چت با مدل
def chat_with_bot(user_input, uploaded_file, processing_state):
    try:
        # تنظیم وضعیت پردازش
        processing_state["is_processing"] = True
        processing_state["cancel_requested"] = False
        
        # دریافت تاریخچه مکالمات
        history = get_conversation_history()
        
        # خواندن محتوای فایل آپلود شده
        file_content = read_file(uploaded_file)
        
        # بررسی درخواست لغو
        if processing_state["cancel_requested"]:
            processing_state["is_processing"] = False
            return "❌ Processing canceled by user."
        
        # دریافت پاسخ از مدل
        response = llm_chain.run({
            "history": history,
            "user_input": user_input,
            "file_content": file_content
        })
        
        # بررسی دوباره درخواست لغو
        if processing_state["cancel_requested"]:
            processing_state["is_processing"] = False
            return "❌ Processing canceled by user."
        
        # ذخیره مکالمه در پایگاه داده
        save_to_db(user_input + (f"\n[File uploaded: {uploaded_file.name}]" if uploaded_file else ""), response)
        
        # پایان پردازش
        processing_state["is_processing"] = False
        return response
    except Exception as e:
        processing_state["is_processing"] = False
        return f"❌ Error in processing: {e}"

# تابع برای لغو پردازش
def cancel_processing(processing_state):
    if processing_state["is_processing"]:
        processing_state["cancel_requested"] = True
        return "Cancel requested. Stopping processing..."
    return "No active processing to cancel."

# تعریف رابط کاربری با Gradio
with gr.Blocks(theme="soft") as iface:
    gr.Markdown("# 🤖 Conversational Chatbot")
    gr.Markdown("With this chatbot, you can chat about any topic! Ask questions, propose ideas, or request assistance. You can also upload .txt, .py, .csv, or .json files to analyze their content.")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Enter your message:", lines=2, placeholder="Type your message...")
            uploaded_file = gr.File(label="Upload a .txt, .py, .csv, or .json file", file_types=[".txt", ".py", ".csv", ".json"])
            submit_button = gr.Button("Submit")
            cancel_button = gr.Button("Cancel Processing")
        
        output = gr.Textbox(label="Bot Response:", lines=10, interactive=False)
    
    state = gr.State(value=processing_state)
    
    # اتصال دکمه Submit به تابع چت
    submit_button.click(
        fn=chat_with_bot,
        inputs=[user_input, uploaded_file, state],
        outputs=output
    )
    
    # اتصال دکمه Cancel به تابع لغو
    cancel_button.click(
        fn=cancel_processing,
        inputs=[state],
        outputs=output
    )

# اجرای برنامه Gradio
if __name__ == "__main__":
    iface.launch()