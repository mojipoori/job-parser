import pandas as pd
import re
import ollama
import os
from datetime import datetime
import uuid
import json

# ستون‌های خروجی
CSV_COLUMNS = [
    'id', 'title', 'employer', 'location', 'schedule', 'publication_date', 'link',
    'description', 'timestamp', 'contact_info', 'application_method', 'full_description',
    'extracted_urls', 'search_date', 'phone_number', 'email', 'source_site',
    'company_size', 'application_links', 'posted', 'application_emails',
    'contact_person', 'contact_phone', 'job_type', 'direct_application_url'
]

# قالب کامل برای پرامپت با جایگزینی امن
FIELDS_BLOCK = ",\n".join([f'"{col}": ""' for col in CSV_COLUMNS])
PROMPT_TEMPLATE = """
You are an expert data extractor. Given the raw German job posting text below, extract the information and return it as a JSON object with the following keys.
If a field is not found, return an empty string ("\"") for that field. Only include information explicitly present in the text.

Raw text:
{{raw_text}}

Output format:
```json
{{
{fields_block}
}}
```
""".replace("{fields_block}", FIELDS_BLOCK)

# ایجاد پوشه داده خام (در صورت نیاز)
def create_raw_data_folder():
    raw_data_dir = "raw_data"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    return raw_data_dir

# خواندن فایل ورودی
def read_raw_text_from_file(filename):
    raw_data_dir = create_raw_data_folder()
    file_path = os.path.join(raw_data_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""

# پاک‌سازی متن خام
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# استخراج اطلاعات شغلی با مدل زبانی
def extract_job_details(raw_text, model_name="llama3.3"):
    cleaned_text = clean_text(raw_text)
    prompt = PROMPT_TEMPLATE.replace("{{raw_text}}", cleaned_text)
    
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        raw_response = response['response']

        print("=== RAW MODEL RESPONSE ===")
        print(raw_response)

        # حذف backticks و کلمه json در صورت وجود
        clean_response = re.sub(r"```json|```", "", raw_response).strip()
        extracted_data = json.loads(clean_response)

        # اطمینان از وجود همه ستون‌ها
        for col in CSV_COLUMNS:
            if col not in extracted_data:
                extracted_data[col] = ""
        return extracted_data

    except Exception as e:
        print(f"Error extracting details: {e}")
        return {col: "" for col in CSV_COLUMNS}

# تولید شناسه یکتا برای آگهی
def generate_job_id():
    return str(uuid.uuid4())

# پردازش آگهی و تبدیل به فایل CSV
def process_raw_text_to_csv(raw_text, input_file, model_name="llama3.3"):
    job_data = extract_job_details(raw_text, model_name)
    job_data['id'] = generate_job_id()
    job_data['timestamp'] = int(datetime.now().timestamp())
    job_data['search_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    base_name = os.path.splitext(input_file)[0]
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f"{base_name}_{timestamp_str}.csv"
    
    df = pd.DataFrame([job_data], columns=CSV_COLUMNS)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Output saved to {output_csv}")

# اجرای اصلی برنامه
if __name__ == "__main__":
    input_file = "job_posting.txt"
    raw_text = read_raw_text_from_file(input_file)
    
    if raw_text:
        process_raw_text_to_csv(raw_text, input_file, model_name="llama3.3")
    else:
        print("No data read from file.")
