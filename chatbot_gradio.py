import gradio as gr
from langchain_ollama import OllamaLLM

# تنظیم مدل Ollama
llm = OllamaLLM(model="", temperature=0.7)

# تعریف تابع برای پاسخ‌دهی مدل
def chat_with_bot(user_input):
    try:
        # دریافت پاسخ از مدل
        response = llm.invoke(user_input)
        return response
    except Exception as e:
        return f"❌ خطا در پردازش: {e}"

# ایجاد رابط کاربری با Gradio
iface = gr.Interface(
    fn=chat_with_bot, 
    inputs=gr.Textbox(label="پیام خود را وارد کنید:", lines=2, placeholder="پیام خود را بنویسید..."), 
    outputs="text", 
    title="🤖 چت‌بات مکالمه‌ای", 
    description="با این چت‌بات می‌توانید درباره هر موضوعی صحبت کنید! سؤال بپرسید، ایده مطرح کنید یا درخواست کمک کنید.",
    theme="compact"  # این برای طراحی سبک و فشرده است
)

# اجرای برنامه Gradio
if __name__ == "__main__":
    iface.launch()
