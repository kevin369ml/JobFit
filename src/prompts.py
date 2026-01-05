from langchain_core.prompts import PromptTemplate
import json


# Create the extraction prompt template
RESUME_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resume_text"],
    template="""You are an expert resume parser and data extraction specialist. 
Your task is to extract structured information from the provided resume text and return ONLY valid JSON.

CRITICAL INSTRUCTIONS:
1. Extract ALL relevant information from the resume, do not add new information. Only extract information that is explicitly mentioned.
2. For dates, use ISO format (YYYY-MM-DD) if available, otherwise return null
3. For work experience, capture key achievements as strings
4. All fields are optional - use null if information is not available
5. Only extract information that is explicitly mentioned
6. DO NOT add any text before or after the JSON
7. DO NOT wrap the JSON in markdown code blocks
8. DO NOT include explanations, preamble, or any other text
9. Return ONLY valid JSON that can be parsed by json.loads()

Resume Text:
{resume_text}

Return the JSON object with this exact structure:
{{
  "contact_info": {{
    "name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string or null"
  }},
  "summary": "string or null",
  "work_experience": [
    {{
      "company": "string or null",
      "job_title": "string or null",
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null",
      "is_current": "boolean or null",
      "key_achievements": ["string1", "string2"] or []
    }}
  ],
  "education": [
    {{
      "institution": "string or null",
      "degree": "string or null",
      "field_of_study": "string or null",
      "graduation_date": "YYYY-MM-DD or null"
    }}
  ],
  "skills": ["string1", "string2"] or null,
  "languages": ["string1", "string2"] or null,
  "projects": ["string1", "string2"] or null,
  "raw_text": null
}}"""
)

def get_resume_extraction_prompt():
    return RESUME_EXTRACTION_PROMPT

