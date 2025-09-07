"""
FastAPI application for HR Talent Matching and Development AI System
Main API endpoints for resume upload, project matching, and skill analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os

from db import db_manager, Employee, Project, Course
from vector_store import vector_store
from utils import resume_parser, skill_analyzer
from match import talent_matcher


# Initialize FastAPI app
app = FastAPI(
    title="HR Talent Matching AI",
    description="Local AI system for talent matching and development",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Pydantic models for API requests/responses
class ResumeUploadRequest(BaseModel):
    name: str
    resume_text: str
    preferences: Optional[List[str]] = []


class ProjectRequest(BaseModel):
    title: str
    required_skills: List[str]
    team_size: int
    description: str


class CourseRequest(BaseModel):
    title: str
    skill_tags: List[str]
    provider: str
    url: str
    description: str


class MatchResponse(BaseModel):
    employee_id: int
    matches: List[Dict]
    total_matches: int


class SkillGapResponse(BaseModel):
    employee_id: int
    skill_gap_analysis: Dict
    recommended_courses: List[Dict]


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    else:
        return HTMLResponse("""
        <html>
            <head><title>HR Talent AI</title></head>
            <body>
                <h1>HR Talent Matching AI</h1>
                <p>Frontend not found. Please create frontend/index.html</p>
                <p>API documentation available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)


@app.post("/upload_resume")
async def upload_resume(request: ResumeUploadRequest):
    """Upload and process employee resume"""
    try:
        # Extract skills from resume
        extracted_skills = resume_parser.extract_skills(request.resume_text)
        
        # Create employee object
        employee = Employee(
            name=request.name,
            skills=extracted_skills,
            preferences=request.preferences,
            resume_text=request.resume_text
        )
        
        # Add to database
        employee_id = db_manager.add_employee(employee)
        
        # Generate embedding
        employee_text = f"{employee.name} Skills: {', '.join(employee.skills)} Resume: {employee.resume_text}"
        embedding = vector_store.generate_embedding(employee_text)
        
        # Update database with embedding
        db_manager.update_employee_embedding(employee_id, embedding)
        
        # Add to vector store
        vector_store.add_employee_embedding(
            employee_id, employee.name, employee.skills, 
            employee.resume_text, embedding
        )
        
        return {
            "message": "Resume uploaded successfully",
            "employee_id": employee_id,
            "extracted_skills": extracted_skills,
            "total_skills": len(extracted_skills)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


@app.post("/add_project")
async def add_project(request: ProjectRequest):
    """Add a new project to the system"""
    try:
        # Create project object
        project = Project(
            title=request.title,
            required_skills=request.required_skills,
            team_size=request.team_size,
            description=request.description
        )
        
        # Add to database
        project_id = db_manager.add_project(project)
        
        # Generate embedding
        project_text = f"{project.title} Required Skills: {', '.join(project.required_skills)} Description: {project.description}"
        embedding = vector_store.generate_embedding(project_text)
        
        # Update database with embedding
        db_manager.update_project_embedding(project_id, embedding)
        
        # Add to vector store
        vector_store.add_project_embedding(
            project_id, project.title, project.required_skills,
            project.description, embedding
        )
        
        return {
            "message": "Project added successfully",
            "project_id": project_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding project: {str(e)}")


@app.put("/projects/{project_id}")
async def update_project(project_id: int, request: ProjectRequest):
    """Update an existing project"""
    try:
        # Check if project exists
        existing_project = db_manager.get_project(project_id)
        if not existing_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create updated project object
        project = Project(
            title=request.title,
            required_skills=request.required_skills,
            team_size=request.team_size,
            description=request.description
        )
        
        # Update in database
        db_manager.update_project(project_id, project)
        
        # Generate new embedding
        project_text = f"{project.title} Required Skills: {', '.join(project.required_skills)} Description: {project.description}"
        embedding = vector_store.generate_embedding(project_text)
        
        # Update embedding in database
        db_manager.update_project_embedding(project_id, embedding)
        
        # Update in vector store
        vector_store.update_project_embedding(
            project_id, project.title, project.required_skills,
            project.description, embedding
        )
        
        return {
            "message": "Project updated successfully",
            "project_id": project_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating project: {str(e)}")


@app.post("/add_course")
async def add_course(request: CourseRequest):
    """Add a new course to the system"""
    try:
        # Create course object
        course = Course(
            title=request.title,
            skill_tags=request.skill_tags,
            provider=request.provider,
            url=request.url,
            description=request.description
        )
        
        # Add to database
        course_id = db_manager.add_course(course)
        
        return {
            "message": "Course added successfully",
            "course_id": course_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding course: {str(e)}")


@app.get("/match_projects/{employee_id}")
async def match_projects(employee_id: int, top_k: int = 5):
    """Get project matches for an employee"""
    try:
        matches = talent_matcher.match_employee_to_projects(employee_id, top_k)
        
        return MatchResponse(
            employee_id=employee_id,
            matches=matches,
            total_matches=len(matches)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching projects: {str(e)}")


@app.get("/match_employees/{project_id}")
async def match_employees(project_id: int, top_k: int = 5):
    """Get employee matches for a project"""
    try:
        matches = talent_matcher.match_project_to_employees(project_id, top_k)
        
        return {
            "project_id": project_id,
            "matches": matches,
            "total_matches": len(matches)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching employees: {str(e)}")


@app.get("/skill_gap/{employee_id}")
async def get_skill_gap(employee_id: int, project_id: Optional[int] = None):
    """Get skill gap analysis for an employee"""
    try:
        skill_gap_analysis = talent_matcher.analyze_skill_gaps(employee_id, project_id)
        
        if not skill_gap_analysis:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return SkillGapResponse(
            employee_id=employee_id,
            skill_gap_analysis=skill_gap_analysis.get('skill_gap_analysis', {}),
            recommended_courses=skill_gap_analysis.get('recommended_courses', [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing skill gaps: {str(e)}")


@app.get("/career_path/{employee_id}")
async def get_career_path(employee_id: int):
    """Get career path suggestions for an employee"""
    try:
        career_suggestions = talent_matcher.get_career_path_suggestions(employee_id)
        
        if not career_suggestions:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return career_suggestions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating career path: {str(e)}")


@app.get("/employees")
async def get_all_employees():
    """Get all employees"""
    try:
        # This would require adding a method to db_manager
        # For now, return a simple response
        return {"message": "Employee list endpoint - to be implemented"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching employees: {str(e)}")


@app.get("/projects")
async def get_all_projects():
    """Get all projects"""
    try:
        projects = db_manager.get_all_projects()
        return {
            "projects": [
                {
                    "id": project.id,
                    "title": project.title,
                    "required_skills": project.required_skills,
                    "team_size": project.team_size,
                    "description": project.description
                }
                for project in projects
            ],
            "total_projects": len(projects)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")


@app.get("/courses")
async def get_all_courses():
    """Get all courses"""
    try:
        courses = db_manager.get_all_courses()
        return {
            "courses": [
                {
                    "id": course.id,
                    "title": course.title,
                    "skill_tags": course.skill_tags,
                    "provider": course.provider,
                    "url": course.url,
                    "description": course.description
                }
                for course in courses
            ],
            "total_courses": len(courses)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching courses: {str(e)}")


@app.get("/employee/{employee_id}")
async def get_employee_profile(employee_id: int):
    """Get employee profile with match history"""
    try:
        # Get employee details
        employee = db_manager.get_employee(employee_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Get match history
        match_history = db_manager.get_employee_match_history(employee_id)
        
        return {
            "employee_profile": {
                "id": employee.id,
                "name": employee.name,
                "skills": employee.skills,
                "preferences": employee.preferences,
                "resume_text": employee.resume_text,
                "total_skills": len(employee.skills)
            },
            "match_history": match_history,
            "total_matches": len(match_history)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching employee profile: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HR Talent AI system is running"}


# Initialize sample data on startup
@app.on_event("startup")
async def startup_event():
    """Initialize sample data when the application starts"""
    try:
        # Check if we already have data
        projects = db_manager.get_all_projects()
        courses = db_manager.get_all_courses()
        
        if not projects and not courses:
            # Load sample data
            await load_sample_data()
    
    except Exception as e:
        print(f"Error during startup: {e}")


async def load_sample_data():
    """Load sample data from JSON files"""
    try:
        # Load sample projects
        if os.path.exists("data/sample_projects.json"):
            with open("data/sample_projects.json", "r") as f:
                sample_projects = json.load(f)
            
            for project_data in sample_projects:
                project = Project(**project_data)
                project_id = db_manager.add_project(project)
                
                # Generate and store embedding
                project_text = f"{project.title} Required Skills: {', '.join(project.required_skills)} Description: {project.description}"
                embedding = vector_store.generate_embedding(project_text)
                db_manager.update_project_embedding(project_id, embedding)
                vector_store.add_project_embedding(
                    project_id, project.title, project.required_skills,
                    project.description, embedding
                )
        
        # Load sample courses
        if os.path.exists("data/sample_courses.json"):
            with open("data/sample_courses.json", "r") as f:
                sample_courses = json.load(f)
            
            for course_data in sample_courses:
                course = Course(**course_data)
                db_manager.add_course(course)
        
        print("Sample data loaded successfully")
    
    except Exception as e:
        print(f"Error loading sample data: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
