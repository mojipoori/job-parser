import streamlit as st
import json
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# تعریف فیلدهای استخراج
EXTRACTION_FIELDS = [
    "id", "title", "employer", "location", "schedule", "publication_date", "link",
    "description", "timestamp", "contact_info", "application_method", "full_description",
    "extracted_urls", "search_date", "phone_number", "email", "source_site",
    "company_size", "application_links", "posted", "application_emails",
    "contact_person", "contact_phone", "job_type", "direct_application_url"
]

# ایجاد قالب JSON برای پرامپت
FIELDS_JSON_EXAMPLE = ",\n".join([f'"{field}": ""' for field in EXTRACTION_FIELDS])

# تعریف پرامپت ساده با نقل‌قول‌های سه‌گانه
BASE_PROMPT = """You are a professional information extractor. Given a raw German job posting text, extract all explicitly stated details and return a JSON object with the following keys:

{fields}

If a field is not mentioned in the text, return an empty string (""). Do not invent or assume any values. Return only the JSON object.

Text to extract:
{{raw_text}}
""".format(fields=', '.join([f'"{f}"' for f in EXTRACTION_FIELDS]))

# تنظیم مدل Ollama
llm = OllamaLLM(model="llama3.3", temperature=0.3)

# تعریف پرامپت با LangChain
prompt = PromptTemplate(
    input_variables=["raw_text"],
    template=BASE_PROMPT
)

# تعریف زنجیره با RunnableSequence
extract_chain = RunnableSequence(prompt | llm)

@st.cache_data
def extract_job_info(text: str) -> str:
    """
    اجرای فرآیند استخراج اطلاعات از آگهی شغلی خام.
    خروجی به‌صورت JSON متنی خواهد بود (string).
    """
    try:
        # بررسی اولیه متن
        if len(text.strip()) < 50:
            return json.dumps({"error": "متن ورودی خیلی کوتاه است."})
        
        result = extract_chain.invoke({"raw_text": text})
        # استخراج بخش JSON با استفاده از regex
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # اعتبارسنجی JSON
            json.loads(json_str)
            return json_str
        else:
            return json.dumps({"error": "خروجی مدل JSON معتبر نیست"})
    except json.JSONDecodeError:
        return json.dumps({"error": "خروجی مدل JSON معتبر نیست"})
    except Exception as e:
        return json.dumps({"error": f"خطا در پردازش: {str(e)}"})

# تنظیمات صفحه Streamlit
st.set_page_config(page_title="Job Info Extractor", layout="centered")
st.title("🧠 استخراج اطلاعات آگهی شغلی")

st.markdown("""
بارگذاری یک فایل متنی یا وارد کردن مستقیم متن آگهی شغلی به زبان آلمانی، سپس اجرای ایجنت برای استخراج اطلاعات ساختاریافته (JSON).
""")

# تب‌های ورودی
tab1, tab2 = st.tabs(["📄 بارگذاری فایل", "✍️ وارد کردن متن"])

job_text = ""
with tab1:
    uploaded_file = st.file_uploader("فایل متنی آگهی شغلی را آپلود کنید (.txt)", type="txt")
    if uploaded_file is not None:
        try:
            job_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("فایل باید با فرمت UTF-8 باشد.")
            job_text = ""

with tab2:
    job_text = st.text_area("یا متن خام آگهی شغلی را اینجا وارد کنید:", height=300)

# اعتبارسنجی محتوای ورودی
if job_text.strip():
    # بررسی اینکه متن به نظر آگهی شغلی باشد
    job_keywords = ["stellenanzeige", "job", "bewerbung", "m/w/d", "karriere", "stelle"]
    if not any(keyword in job_text.lower() for keyword in job_keywords):
        st.warning("متن وارد شده ممکن است یک آگهی شغلی نباشد.")

    if st.button("🚀 استخراج اطلاعات با LangChain"):
        with st.spinner("در حال پردازش با LLaMA 3.3 از طریق Ollama..."):
            try:
                result = extract_job_info(job_text)
                json_data = json.loads(result)
                
                if "error" in json_data:
                    st.error(f"❌ {json_data['error']}")
                else:
                    st.success("✅ اطلاعات با موفقیت استخراج شد")
                    st.subheader("🔍 خروجی JSON")
                    st.json(json_data)

                    # امکان دانلود JSON
                    st.download_button(
                        label="📥 دانلود خروجی JSON",
                        data=json.dumps(json_data, indent=2, ensure_ascii=False),
                        file_name="job_info.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"❌ خطا در پردازش: {e}")
else:
    st.info("لطفاً ابتدا یک فایل آپلود کرده یا متن وارد کنید.")

if __name__ == "__main__":
    st.write("اپلیکیشن در حال اجرا است...")