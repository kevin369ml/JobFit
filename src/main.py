import json
from typing import Optional
import fitz
from langchain_ollama import ChatOllama

from schemas import Resume
from prompts import get_resume_extraction_prompt


def read_file(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pymupdf.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")


def extract_resume_data(resume_text: str, api_key: Optional[str] = None, model: str = "gemini-pro") -> Resume:
    """
    Extract structured resume data from text using LLM.
    
    Args:
        resume_text: Raw resume text
        api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
        model: LLM model to use (default: gemini-pro)
        
    Returns:
        Resume: Structured resume data
    """
    llm = ChatOllama(model="llama3.1", temperature=0)
    extraction_prompt = get_resume_extraction_prompt()
    
    # Format the prompt with resume text
    formatted_prompt = extraction_prompt.format(resume_text=resume_text)
    
    # Call LLM
    response = llm.invoke(formatted_prompt)
    
    # Extract JSON from response
    try:
        json_str = response.content.strip()
        resume_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response.content}")
    
    # Validate and create Resume object
    resume = Resume(**resume_dict)
    resume.raw_text = resume_text  # Store raw text for RAG
    return resume


def process_pdf(resume_text: str, api_key: Optional[str] = None, model: str = "gemini-pro") -> Resume:
    """
    End-to-end pipeline: Extract text from PDF and parse resume data.
    
    Args:
        resume_text: Raw resume text
        api_key: Google API key
        model: LLM model to use
        
    Returns:
        Resume: Structured resume data
    """
    # Extract structured data
    resume = extract_resume_data(resume_text, api_key=api_key, model=model)
    
    return resume


def score_match(resume: Resume, job_description: str, api_key: Optional[str] = None, model: str = "gemini-pro") -> dict:
    """
    Score how well a resume matches a job description.
    
    Args:
        resume: Structured resume data
        job_description: Job description text
        api_key: Google API key
        model: LLM model to use
        
    Returns:
        dict: Match score and analysis
    """
    llm = ChatGoogleGenerativeAI(api_key=api_key, model=model, temperature=0)
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
    """Example usage of resume extraction and job matching."""
    
    # Process resume PDF
    pdf_path = "src/sample_data/sample_cv.pdf"
    
    try:
        resume_text = read_file(pdf_path)
        resume = process_pdf(resume_text)
        
        print("Extracted Resume Data:")
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
        
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()