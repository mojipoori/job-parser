EXTRACTION_FIELDS = [
    "id", "title", "employer", "location", "schedule", "publication_date", "link",
    "description", "timestamp", "contact_info", "application_method", "full_description",
    "extracted_urls", "search_date", "phone_number", "email", "source_site",
    "company_size", "application_links", "posted", "application_emails",
    "contact_person", "contact_phone", "job_type", "direct_application_url"
]

FIELDS_JSON_EXAMPLE = ",\n".join([f'\"{field}\": \"\"' for field in EXTRACTION_FIELDS])

BASE_PROMPT = f"""
You are a professional information extractor.

Given the raw German job posting text, extract all explicitly stated details and return a JSON object with the following keys:

{', '.join([f'"{f}"' for f in EXTRACTION_FIELDS])}

If a field is missing, return an empty string (""). Do not make up any values.

---

Example Input:
"""
Junior Softwareentwickler (m/w/d) bei TechNova GmbH in Berlin.
Vollzeitstelle, veröffentlicht am 10.04.2024.
Bewerbung per E-Mail an jobs@technova.de oder über unser Onlineportal.
Kontakt: Anna Müller, Tel. +49 30 1234567.

Wir suchen einen motivierten Entwickler für unser wachsendes Team...
"""

Expected Output:
```json
{{
  "id": "",
  "title": "Junior Softwareentwickler",
  "employer": "TechNova GmbH",
  "location": "Berlin",
  "schedule": "Vollzeit",
  "publication_date": "10.04.2024",
  "link": "",
  "description": "Wir suchen einen motivierten Entwickler für unser wachsendes Team...",
  "timestamp": "",
  "contact_info": "Anna Müller, Tel. +49 30 1234567",
  "application_method": "E-Mail, Onlineportal",
  "full_description": "Junior Softwareentwickler (m/w/d)...",
  "extracted_urls": "",
  "search_date": "",
  "phone_number": "+49 30 1234567",
  "email": "jobs@technova.de",
  "source_site": "",
  "company_size": "",
  "application_links": "",
  "posted": "",
  "application_emails": "jobs@technova.de",
  "contact_person": "Anna Müller",
  "contact_phone": "+49 30 1234567",
  "job_type": "",
  "direct_application_url": ""
}}
```

---

Now extract from the following text:
"""
{{raw_text}}
"""

Return JSON only.
"""
