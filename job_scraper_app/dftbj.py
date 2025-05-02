import streamlit as st
import json
from agent import extract_job_info

st.set_page_config(page_title="Job Info Extractor", layout="centered")
st.title("🧠 استخراج اطلاعات آگهی شغلی")

st.markdown("""
بارگذاری یک فایل متنی یا وارد کردن مستقیم متن آگهی شغلی به زبان آلمانی، سپس اجرای ایجنت برای استخراج اطلاعات ساختاریافته (JSON).
""")

# --- ورودی متن یا فایل ---
tab1, tab2 = st.tabs(["📄 بارگذاری فایل", "✍️ وارد کردن متن"])

job_text = ""
with tab1:
    uploaded_file = st.file_uploader("فایل متنی آگهی شغلی را آپلود کنید (.txt)", type="txt")
    if uploaded_file is not None:
        job_text = uploaded_file.read().decode("utf-8")

with tab2:
    job_text = st.text_area("یا متن خام آگهی شغلی را اینجا وارد کنید:", height=300)

# --- اجرای ایجنت ---
if job_text.strip():
    if st.button("🚀 استخراج اطلاعات با LangChain"):
        with st.spinner("در حال پردازش با LLaMA 3.3 از طریق Ollama..."):
            try:
                result = extract_job_info(job_text)
                json_data = json.loads(result)
                st.success("✅ اطلاعات با موفقیت استخراج شد")
                st.subheader("🔍 خروجی JSON")
                st.json(json_data)

                st.download_button(
                    label="📥 دانلود خروجی JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name="job_info.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ خطا در پردازش: {e}")
else:
    st.info("لطفاً ابتدا یک فایل آپلود کرده یا متن وارد کنید.")
