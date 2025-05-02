import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os
from datetime import datetime

# Initialize the Ollama model
llm = OllamaLLM(model="llama3.3", temperature=0.7)

# Define the prompt for extracting job details
EXTRACT_PROMPT = """
You are an expert data extraction assistant. Your task is to analyze the provided job posting text and extract specific information corresponding to the following fields: id, title, employer, location, schedule, publication_date, link, description, timestamp, contact_info, application_method, full_description, extracted_urls, search_date, phone_number, email, source_site, company_size, application_links, posted, application_emails, contact_person, contact_phone, job_type, direct_application_url.

Instructions:
- For fields not explicitly found in the text, return an empty string ("") unless specified otherwise.
- For 'id', generate a unique integer based on the current timestamp.
- For 'timestamp', use the current Unix timestamp.
- For 'search_date', use the current date in format 'YYYY-MM-DD HH:MM:SS'.
- For 'full_description', include the entire job description section if available.
- For 'extracted_urls' and 'application_links', return a list of URLs found in the text.
- Return the extracted information as a JSON object. Ensure the output is valid JSON.
- If a field is ambiguous, use context to make a logical interpretation.

Job posting text:
{job_text}

Response (must be valid JSON):
{
  "id": "",
  "title": "",
  "employer": "",
  "location": "",
  "schedule": "",
  "publication_date": "",
  "link": "",
  "description": "",
  "timestamp": "",
  "contact_info": "",
  "application_method": "",
  "full_description": "",
  "extracted_urls": [],
  "search_date": "",
  "phone_number": "",
  "email": "",
  "source_site": "",
  "company_size": "",
  "application_links": [],
  "posted": "",
  "application_emails": [],
  "contact_person": "",
  "contact_phone": "",
  "job_type": "",
  "direct_application_url": ""
}
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# Function to format extracted data as plain text
def format_as_text(data):
    output = ""
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value) if value else "[]"
        output += f"{key}: {value}\n"
    return output

# Function to extract job details using the language model
def extract_job_details(job_text):
    try:
        # Run the extraction chain
        response = extract_chain.run({"job_text": job_text})
        
        # Try to parse the JSON response
        try:
            extracted_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Return raw response with error message
            return {
                "error": "Failed to parse model response as JSON",
                "raw_response": response
            }
        
        # Ensure all required fields are present
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
                extracted_data[field] = "" if field not in ["extracted_urls", "application_links", "application_emails"] else []
        
        # Set default values for specific fields
        if not extracted_data["id"]:
            extracted_data["id"] = int(datetime.now().timestamp())
        if not extracted_data["timestamp"]:
            extracted_data["timestamp"] = int(datetime.now().timestamp())
        if not extracted_data["search_date"]:
            extracted_data["search_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return extracted_data
    except Exception as e:
        return {"error": f"Error extracting details: {e}"}

# Main function to process the job posting and display output
def process_job_posting(user_input, uploaded_file):
    try:
        # Read the job posting text
        job_text = read_file(uploaded_file.name if uploaded_file else "job_posting.txt")
        if "Error" in job_text:
            return job_text
        
        # Extract details using the language model
        extracted_data = extract_job_details(job_text)
        
        # Handle errors
        if "error" in extracted_data:
            return f"Error: {extracted_data['error']}\nRaw Response: {extracted_data.get('raw_response', 'N/A')}"
        
        # Format the extracted data as plain text
        output_text = format_as_text(extracted_data)
        
        return output_text
    except Exception as e:
        return f"Error processing job posting: {e}"

# Gradio interface
iface = gr.Interface(
    fn=process_job_posting, 
    inputs=[
        gr.Textbox(label="Enter your message:", lines=2, placeholder="Type your message..."),
        gr.File(label="Upload a .txt file (e.g., job_posting.txt)", file_types=[".txt"])
    ], 
    outputs="text", 
    title="🤖 Job Posting Extractor", 
    description="Upload a job posting text file to extract structured information and display it as text. The system uses a language model to handle diverse and unstructured job posting formats.",
    theme="compact"
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()