from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from prompt_template import BASE_PROMPT

# مدل LLaMA 3.3 که در سیستم local با Ollama اجرا می‌شود
llm = Ollama(model="llama3.3")

# تعریف قالب پرامپت با placeholder برای متن خام
prompt = PromptTemplate(
    input_variables=["raw_text"],
    template=BASE_PROMPT
)

# تعریف زنجیره LangChain که پرامپت را با مدل ترکیب می‌کند
extract_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

def extract_job_info(text: str) -> str:
    """
    اجرای فرآیند استخراج اطلاعات از آگهی شغلی خام.
    خروجی به‌صورت JSON متنی خواهد بود (string).
    """
    result = extract_chain.run({"raw_text": text})
    return result.strip()
