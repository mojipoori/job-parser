import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from datetime import datetime

# Initialize the Ollama model
llm = OllamaLLM(model="llama3.2", temperature=0.7)

# Define the prompt for extracting job details as plain text
EXTRACT_PROMPT = """
You are an expert data extraction assistant. Your task is to analyze the provided job posting text and extract specific information for the following fields: id, title, employer, location, schedule, publication_date, link, description, timestamp, contact_info, application_method, full_description, extracted_urls, search_date, phone_number, email, source_site, company_size, application_links, posted, application_emails, contact_person, contact_phone, job_type, direct_application_url.

Instructions:
- Extract only information explicitly present in the text. Do not add or estimate data unless explicitly stated.
- Salary ranges are allowed if explicitly mentioned (e.g., "40.000 - 65.000 EUR").
- For missing fields, return an empty string ("") or empty list "[]" as appropriate.
- For 'id', generate a unique integer based on the current timestamp.
- For 'timestamp', use the current Unix timestamp.
- For 'search_date', use the current date in format 'YYYY-MM-DD HH:MM:SS'.
- For 'full_description', include the entire job description section from the company introduction to the benefits offered.
- For 'extracted_urls' and 'application_links', extract any URLs found in the text. If none are found, return "[]".
- For 'posted', use the publication date if available.
- Ignore irrelevant sections (e.g., similar job postings from other companies).
- Do not include interpretive comments (e.g., "this job looks promising").
- Include specific technologies (e.g., Vue.js, React) in the 'description' field if mentioned.
- Return the result as plain text with each field in the format "key: value" on a new line.

Example:
Input text:
```
Frontend Developer (m/w/d)
40.000 - 65.000 EUR
Sanitätshaus Aktuell AG
Auf d. Höhe 50, Vettelschoß
Vollzeit
Anforderungen: Vue.js, React, Tailwind CSS...
Beschreibung: Gestalte die Zukunft mit uns!...
Über uns: Willkommen bei joviva, einem corporate Startup mit Sitz in Köln...
```
Output:
id: 1745939000
title: Frontend Developer (m/w/d)
employer: Sanitätshaus Aktuell AG
location: Vettelschoß, Deutschland
schedule: Vollzeit
publication_date: 
link: 
description: Entwicklung mit Vue.js, React, Tailwind CSS; Gehalt: 40.000 - 65.000 EUR
timestamp: 1745939000
contact_info: 
application_method: 
full_description: Gestalte die Zukunft mit uns!... Willkommen bei joviva, einem corporate Startup mit Sitz in Köln...
extracted_urls: []
search_date: 2025-05-02 14:50:00
phone_number: 
email: 
source_site: 
company_size: 50-200 Mitarbeiter
application_links: []
posted: 
application_emails: []
contact_person: 
contact_phone: 
job_type: Vollzeit
direct_application_url: 

Job posting text:
{job_text}

Response (plain text, each field as "key: value" on a new line):
id: 
title: 
employer: 
location: 
schedule: 
publication_date: 
link: 
description: 
timestamp: 
contact_info: 
application_method: 
full_description: 
extracted_urls: []
search_date: 
phone_number: 
email: 
source_site: 
company_size: 
application_links: []
posted: 
application_emails: []
contact_person: 
contact_phone: 
job_type: 
direct_application_url: 
"""

# Create prompt for extraction
extract_prompt = PromptTemplate(
    input_variables=["job_text"],
    template=EXTRACT_PROMPT
)

# Create LLM chain for extraction
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

# Function to read file content
def read_file(file_path):
    if not os.path.exists(file_path):
        return "Error: File not found"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return "Error: File is empty"
            return content
    except Exception as e:
        return f"Error reading file: {e}"

# Function to format extracted data as plain text
def format_as_text(data):
    output = ""
    for line in data.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            output += f"{key.strip()}: {value.strip()}\n"
    return output

# Function to extract job details using the language model
def extract_job_details(job_text):
    if not job_text or not isinstance(job_text, str):
        return "Error: Invalid or empty job text provided"
    try:
        # Run the extraction chain
        response = extract_chain.run({"job_text": job_text})
        if not response:
            return "Error: Model returned empty response"
        
        # Ensure response is in key: value format
        lines = response.split('\n')
        extracted_data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                extracted_data[key.strip()] = value.strip()
        
        # Define required fields
        required_fields = [
            "id", "title", "employer", "location", "schedule", "publication_date",
            "link", "description", "timestamp", "contact_info", "application_method",
            "full_description", "extracted_urls", "search_date", "phone_number",
            "email", "source_site", "company_size", "application_links", "posted",
            "application_emails", "contact_person", "contact_phone", "job_type",
            "direct_application_url"
        ]
        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = "[]" if field in ["extracted_urls", "application_links", "application_emails"] else ""
        
        # Set default values for specific fields
        if not extracted_data["id"]:
            extracted_data["id"] = str(int(datetime.now().timestamp()))
        if not extracted_data["timestamp"]:
            extracted_data["timestamp"] = str(int(datetime.now().timestamp()))
        if not extracted_data["search_date"]:
            extracted_data["search_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Validate key fields
        if not extracted_data.get("title") or not extracted_data.get("employer"):
            return f"Error: Missing key fields (title or employer)\nRaw Response: {response}"
        
        # Format as plain text
        return format_as_text('\n'.join(f"{k}: {v}" for k, v in extracted_data.items()))
    except Exception as e:
        return f"Error extracting details: {e}\nRaw Response: {response if 'response' in locals() else 'N/A'}"

# Main function to process the job posting and display output
def process_job_posting(user_input, uploaded_file=None):
    try:
        # Read the job posting text
        job_text = user_input if user_input else read_file(uploaded_file.name if uploaded_file else "job_posting.txt")
        if "Error" in job_text:
            return job_text
        
        # Debug: Print first 500 chars of job_text
        print(f"Job text: {job_text[:500]}...")
        
        # Extract details using the language model
        output_text = extract_job_details(job_text)
        
        return output_text
    except Exception as e:
        return f"Error processing job posting: {e}"

# Gradio interface
iface = gr.Interface(
    fn=process_job_posting, 
    inputs=[
        gr.Textbox(label="Enter your message:", lines=10, placeholder="Paste the job posting text here..."),
        gr.File(label="Upload a .txt file (e.g., job_posting.txt)", file_types=[".txt"])
    ], 
    outputs="text", 
    title="🤖 Job Posting Extractor", 
    description="Paste or upload a job posting text to extract structured information and display it as text, like a chatbot response.",
    theme="compact"
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()