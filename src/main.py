import re
import json
from typing import Optional
import fitz
from langchain_ollama import ChatOllama

from schemas import Resume
from prompts import get_resume_extraction_prompt

def clean_resume_text(text: str) -> str:
    # normalize weird bullets / zero-width spaces
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # normalize common bullet characters to "-"
    text = text.replace("●", "-").replace("•", "-").replace("​", " ")
    # collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # keep line breaks but collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()

def read_file(pdf_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")


def extract_resume_data(resume_text) -> Resume:
    # llm = ChatOllama(model="llama3.1", temperature=0)
    # llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    # llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
    llm = ChatOllama(model="deepseek-r1:7b", temperature=0)

    chain = get_resume_extraction_prompt() | llm.with_structured_output(Resume)
    resume = chain.invoke({"resume_text": resume_text})
    resume.raw_text = resume_text
    return resume


def process_pdf(resume_text) -> Resume:
    resume = extract_resume_data(resume_text)
    return resume


def score_match(resume: Resume, job_description: str, api_key: Optional[str] = None, model: str = "gemini-pro") -> dict:
    llm = ChatOllama(model="llama3.1", temperature=0)
    resume_json = resume.model_dump_json(indent=2)
    
    prompt = f"""You are an expert recruiter. Analyze the following resume and job description to provide a match score.

RESUME:
{resume_json}

JOB DESCRIPTION:
{job_description}

Provide your analysis in JSON format with the following structure:
{{
  "match_score": <0-100>,
  "skills_match": <0-100>,
  "experience_match": <0-100>,
  "education_match": <0-100>,
  "key_strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "recommendation": "Strong Match|Good Match|Fair Match|Poor Match",
  "analysis": "Brief analysis of why this candidate is/isn't a good fit"
}}

Return ONLY the JSON object, no additional text."""
    
    response = llm.invoke(prompt)
    
    try:
        match_data = json.loads(response.content.strip())
        return match_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse job match response: {str(e)}")


def main():
    
    # Process resume PDF
    pdf_path = "src/sample_data/small.pdf"
    
    resume_text = read_file(pdf_path)
    resume_text = clean_resume_text(resume_text)
    resume = process_pdf(resume_text)
    print(resume.model_dump_json(indent=2))
    
    # # Example job description
    # job_description = """
    # We are looking for a Senior Software Engineer with:
    # - 5+ years of Python experience
    # - Strong background in web development
    # - Experience with PostgreSQL
    # - Bachelor's degree in Computer Science or related field
    # """
    
    # # Score the match
    # match_score = score_match(resume, job_description)
    
    # print("\nJob Match Score:")
    # print(json.dumps(match_score, indent=2))
        

if __name__ == "__main__":
    main()