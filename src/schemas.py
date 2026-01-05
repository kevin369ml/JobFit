from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date

# Contact Information
class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

# Work Experience
class WorkExperience(BaseModel):
    company: Optional[str] = None
    job_title: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_current: Optional[bool] = False
    key_achievements: Optional[List[str]] = []

# Education
class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_date: Optional[date] = None

# Resume Schema
class Resume(BaseModel):
    contact_info: Optional[ContactInfo] = None
    summary: Optional[str] = None
    work_experience: Optional[List[WorkExperience]] = []
    education: Optional[List[Education]] = []
    skills: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    projects: Optional[List[str]] = None
    raw_text: Optional[str] = None

