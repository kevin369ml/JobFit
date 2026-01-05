from langchain_core.prompts import PromptTemplate


RESUME_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resume_text"],
    template="""You are jobfit.ai Resume Extractor.

Task:
Extract structured resume information from the provided resume text.

Hard rules:
- Extract ONLY what is explicitly supported by the resume text.
- Do NOT invent employers, titles, dates, degrees, skills, or metrics.
- If a field is missing or unclear, use null (for single values) or [] (for lists).
- Dates:
  - Prefer YYYY-MM-DD
  - If only month is available, use YYYY-MM
  - Otherwise, use null
- Preserve experience order from most recent to oldest when possible.
- Deduplicate skills and normalize obvious variants only when unambiguous
  (e.g., "k8s" → "Kubernetes").
- Ignore repeated headers, footers, and formatting artifacts.

Output rules:
- Return structured data only.
- Do NOT include explanations, comments, markdown, or extra keys.
- The output must strictly match the schema enforced by the caller.

Resume text:
{resume_text}
"""
)

def get_resume_extraction_prompt() -> PromptTemplate:
    return RESUME_EXTRACTION_PROMPT
