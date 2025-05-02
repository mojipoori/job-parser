import pandas as pd
import re
import ollama
import os
from datetime import datetime
import uuid
import json

# ستون‌های مورد نظر
CSV_COLUMNS = [
    'id', 'title', 'employer', 'location', 'schedule', 'publication_date', 'link',
    'description', 'timestamp', 'contact_info', 'application_method', 'full_description',
    'extracted_urls', 'search_date', 'phone_number', 'email', 'source_site',
    'company_size', 'application_links', 'posted', 'application_emails',
    'contact_person', 'contact_phone', 'job_type', 'direct_application_url'
]

# پرامپت ساده‌شده و بهینه
PROMPT = """
You are an expert data extractor. Given the raw German job posting text below, extract the information and return it as a valid JSON object with the following keys: {fields}. If a field is not found, return an empty string (""). Ensure the output is valid JSON and contains only the requested fields. Do not include extra text or explanations.

Raw text:
{{raw_text}}

Output format:
```json
{{
    {fields_json}
}}