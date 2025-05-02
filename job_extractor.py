import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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