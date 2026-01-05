from langchain_core.prompts import PromptTemplate
import json


# Create the extraction prompt template
RESUME_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resume_text"],
    template="""You are an expert resume parser and data extraction specialist. 
Your task is to extract structured information from the provided resume text and return it in the specified JSON format.

IMPORTANT INSTRUCTIONS:
1. Extract ALL relevant information from the resume, do not add new information. Only extract information that is explicitly mentioned.
2. For dates, use ISO format (YYYY-MM-DD) if available, otherwise return null
3. For work experience, capture key achievements as strings
4. Ensure all required fields (name, institution, degree, field_of_study, company, job_title) are populated. If information is not available, use null
5. Only extract information that is explicitly mentioned

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
  "contact_info": {{
    "name": "string (required)",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string or null",
    "github": "string or null"
  }},
  "summary": "string or null",
  "work_experience": [
    {{
      "company": "string (required)",
      "job_title": "string (required)",
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null",
      "is_current": "boolean (default: false)",
      "key_achievements": ["string1", "string2"]
    }}
  ],
  "education": [
    {{
      "institution": "string (required)",
      "degree": "string (required)",
      "field_of_study": "string (required)",
      "graduation_date": "YYYY-MM-DD or null",
      "gpa": "number or null"
    }}
  ],
  "skills": ["string1", "string2"],
  "languages": ["string1", "string2"],
  "projects": ["string1", "string2"] or null,
  "raw_text": "full resume text or null"
}}

Resume Text:
{resume_text}

Return ONLY the JSON object, no additional text or markdown formatting."""
)

def get_resume_extraction_prompt():
    return RESUME_EXTRACTION_PROMPT

