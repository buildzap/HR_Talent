"""
Database operations for HR Talent Matching System
Handles SQLite database setup and operations for employees, projects, and courses
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class Employee(BaseModel):
    id: Optional[int] = None
    name: str
    skills: List[str]
    preferences: List[str]
    resume_text: str
    embedding_vector: Optional[List[float]] = None


class Project(BaseModel):
    id: Optional[int] = None
    title: str
    required_skills: List[str]
    team_size: int
    description: str
    embedding_vector: Optional[List[float]] = None


class Course(BaseModel):
    id: Optional[int] = None
    title: str
    skill_tags: List[str]
    provider: str
    url: str
    description: str


class DatabaseManager:
    def __init__(self, db_path: str = "hr_talent.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create employees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                skills TEXT NOT NULL,  -- JSON array of skills
                preferences TEXT NOT NULL,  -- JSON array of preferences
                resume_text TEXT NOT NULL,
                embedding_vector TEXT,  -- JSON array of embedding values
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                required_skills TEXT NOT NULL,  -- JSON array of required skills
                team_size INTEGER NOT NULL,
                description TEXT NOT NULL,
                embedding_vector TEXT,  -- JSON array of embedding values
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create courses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                skill_tags TEXT NOT NULL,  -- JSON array of skill tags
                provider TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create match_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                project_id INTEGER NOT NULL,
                match_score REAL NOT NULL,
                missing_skills TEXT,  -- JSON array of missing skills
                matched_skills TEXT,  -- JSON array of matched skills
                skill_match_percentage REAL,
                overall_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (id),
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_employee(self, employee: Employee) -> int:
        """Add a new employee to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO employees (name, skills, preferences, resume_text, embedding_vector)
            VALUES (?, ?, ?, ?, ?)
        """, (
            employee.name,
            json.dumps(employee.skills),
            json.dumps(employee.preferences),
            employee.resume_text,
            json.dumps(employee.embedding_vector) if employee.embedding_vector else None
        ))
        
        employee_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return employee_id
    
    def add_project(self, project: Project) -> int:
        """Add a new project to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO projects (title, required_skills, team_size, description, embedding_vector)
            VALUES (?, ?, ?, ?, ?)
        """, (
            project.title,
            json.dumps(project.required_skills),
            project.team_size,
            project.description,
            json.dumps(project.embedding_vector) if project.embedding_vector else None
        ))
        
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return project_id
    
    def add_course(self, course: Course) -> int:
        """Add a new course to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO courses (title, skill_tags, provider, url, description)
            VALUES (?, ?, ?, ?, ?)
        """, (
            course.title,
            json.dumps(course.skill_tags),
            course.provider,
            course.url,
            course.description
        ))
        
        course_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return course_id
    
    def get_employee(self, employee_id: int) -> Optional[Employee]:
        """Get employee by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, skills, preferences, resume_text, embedding_vector
            FROM employees WHERE id = ?
        """, (employee_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Employee(
                id=row[0],
                name=row[1],
                skills=json.loads(row[2]),
                preferences=json.loads(row[3]),
                resume_text=row[4],
                embedding_vector=json.loads(row[5]) if row[5] else None
            )
        return None
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, required_skills, team_size, description, embedding_vector
            FROM projects WHERE id = ?
        """, (project_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Project(
                id=row[0],
                title=row[1],
                required_skills=json.loads(row[2]),
                team_size=row[3],
                description=row[4],
                embedding_vector=json.loads(row[5]) if row[5] else None
            )
        return None
    
    def get_all_projects(self) -> List[Project]:
        """Get all projects from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, required_skills, team_size, description, embedding_vector
            FROM projects
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        projects = []
        for row in rows:
            projects.append(Project(
                id=row[0],
                title=row[1],
                required_skills=json.loads(row[2]),
                team_size=row[3],
                description=row[4],
                embedding_vector=json.loads(row[5]) if row[5] else None
            ))
        
        return projects
    
    def get_all_courses(self) -> List[Course]:
        """Get all courses from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, skill_tags, provider, url, description
            FROM courses
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        courses = []
        for row in rows:
            courses.append(Course(
                id=row[0],
                title=row[1],
                skill_tags=json.loads(row[2]),
                provider=row[3],
                url=row[4],
                description=row[5]
            ))
        
        return courses
    
    def update_employee_embedding(self, employee_id: int, embedding: List[float]):
        """Update employee's embedding vector"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE employees SET embedding_vector = ? WHERE id = ?
        """, (json.dumps(embedding), employee_id))
        
        conn.commit()
        conn.close()
    
    def update_project_embedding(self, project_id: int, embedding: List[float]):
        """Update project's embedding vector"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE projects SET embedding_vector = ? WHERE id = ?
        """, (json.dumps(embedding), project_id))
        
        conn.commit()
        conn.close()
    
    def update_project(self, project_id: int, project: Project):
        """Update an existing project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE projects 
            SET title = ?, required_skills = ?, team_size = ?, description = ?
            WHERE id = ?
        """, (
            project.title,
            json.dumps(project.required_skills),
            project.team_size,
            project.description,
            project_id
        ))
        
        conn.commit()
        conn.close()
    
    def add_match_history(self, employee_id: int, project_id: int, match_score: float, 
                         missing_skills: List[str], matched_skills: List[str], 
                         skill_match_percentage: float, overall_score: float) -> int:
        """Add or update a match record in the history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if match already exists
        cursor.execute("""
            SELECT id FROM match_history 
            WHERE employee_id = ? AND project_id = ?
        """, (employee_id, project_id))
        
        existing_match = cursor.fetchone()
        
        if existing_match:
            # Update existing match
            cursor.execute("""
                UPDATE match_history 
                SET match_score = ?, missing_skills = ?, matched_skills = ?, 
                    skill_match_percentage = ?, overall_score = ?, timestamp = CURRENT_TIMESTAMP
                WHERE employee_id = ? AND project_id = ?
            """, (
                match_score,
                json.dumps(missing_skills),
                json.dumps(matched_skills),
                skill_match_percentage,
                overall_score,
                employee_id,
                project_id
            ))
            match_id = existing_match[0]
        else:
            # Insert new match
            cursor.execute("""
                INSERT INTO match_history (employee_id, project_id, match_score, missing_skills, 
                                         matched_skills, skill_match_percentage, overall_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                employee_id,
                project_id,
                match_score,
                json.dumps(missing_skills),
                json.dumps(matched_skills),
                skill_match_percentage,
                overall_score
            ))
            match_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return match_id
    
    def get_employee_match_history(self, employee_id: int) -> List[Dict]:
        """Get match history for an employee with project details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT mh.id, mh.employee_id, mh.project_id, mh.match_score, 
                   mh.missing_skills, mh.matched_skills, mh.skill_match_percentage, 
                   mh.overall_score, mh.timestamp,
                   p.title, p.description, p.required_skills, p.team_size
            FROM match_history mh
            JOIN projects p ON mh.project_id = p.id
            WHERE mh.employee_id = ?
            ORDER BY mh.timestamp DESC
        """, (employee_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        match_history = []
        for row in rows:
            match_history.append({
                'match_id': row[0],
                'employee_id': row[1],
                'project_id': row[2],
                'match_score': row[3],
                'missing_skills': json.loads(row[4]) if row[4] else [],
                'matched_skills': json.loads(row[5]) if row[5] else [],
                'skill_match_percentage': row[6],
                'overall_score': row[7],
                'timestamp': row[8],
                'project_title': row[9],
                'project_description': row[10],
                'project_required_skills': json.loads(row[11]) if row[11] else [],
                'project_team_size': row[12]
            })
        
        return match_history


# Global database manager instance
db_manager = DatabaseManager()
