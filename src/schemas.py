from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date

# Contact Information
class ContactInfo(BaseModel):
    name: str = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

# Work Experience
class WorkExperience(BaseModel):
    company: str = None
    job_title: str = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_current: bool = False
    key_achievements: List[str] = []

# Education
class Education(BaseModel):
    institution: str = None
    degree: str = None
    field_of_study: str = None
    graduation_date: Optional[date] = None

# Resume Schema
class Resume(BaseModel):
    contact_info: ContactInfo = None
    summary: Optional[str] = None
    work_experience: List[WorkExperience] = []
    education: List[Education] = []
    skills: List[str] = []
    languages: List[str] = []
    projects: Optional[List[str]] = None
    raw_text: Optional[str] = None

