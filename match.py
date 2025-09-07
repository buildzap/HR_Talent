"""
Skill matching logic for HR Talent Matching System
Handles project-employee matching using embeddings and cosine similarity
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from db import db_manager, Employee, Project, Course
from vector_store import vector_store
from utils import skill_analyzer


class TalentMatcher:
    """Main class for talent matching operations"""
    
    def __init__(self):
        self.db = db_manager
        self.vector_store = vector_store
        self.skill_analyzer = skill_analyzer
    
    def match_employee_to_projects(self, employee_id: int, top_k: int = 5) -> List[Dict]:
        """Find best project matches for an employee"""
        # Get employee data
        employee = self.db.get_employee(employee_id)
        if not employee:
            return []
        
        # Get employee embedding
        if not employee.embedding_vector:
            # Generate embedding if not exists
            employee_text = f"{employee.name} Skills: {', '.join(employee.skills)} Resume: {employee.resume_text}"
            employee_embedding = self.vector_store.generate_embedding(employee_text)
            self.db.update_employee_embedding(employee_id, employee_embedding)
            self.vector_store.update_employee_embedding(
                employee_id, employee.name, employee.skills, 
                employee.resume_text, employee_embedding
            )
        else:
            employee_embedding = employee.embedding_vector
        
        # Find similar projects using vector search
        similar_projects = self.vector_store.find_similar_projects(employee_embedding, top_k)
        
        # Enhance results with skill gap analysis
        enhanced_matches = []
        for project_data in similar_projects:
            project_id = project_data['project_id']
            
            # Get full project details
            projects = self.db.get_all_projects()
            project = next((p for p in projects if p.id == project_id), None)
            
            if project:
                # Analyze skill gaps
                skill_gap_analysis = self.skill_analyzer.find_skill_gaps(
                    employee.skills, project.required_skills
                )
                
                overall_score = self._calculate_overall_score(
                    project_data['similarity_score'],
                    skill_gap_analysis['match_percentage']
                )
                
                enhanced_match = {
                    'project_id': project_id,
                    'title': project.title,
                    'description': project.description,
                    'team_size': project.team_size,
                    'required_skills': project.required_skills,
                    'similarity_score': project_data['similarity_score'],
                    'skill_match_percentage': skill_gap_analysis['match_percentage'],
                    'matched_skills': skill_gap_analysis['matched_skills'],
                    'missing_skills': skill_gap_analysis['missing_skills'],
                    'overall_score': overall_score
                }
                enhanced_matches.append(enhanced_match)
                
                # Store match in history
                self.db.add_match_history(
                    employee_id=employee_id,
                    project_id=project_id,
                    match_score=project_data['similarity_score'],
                    missing_skills=skill_gap_analysis['missing_skills'],
                    matched_skills=skill_gap_analysis['matched_skills'],
                    skill_match_percentage=skill_gap_analysis['match_percentage'],
                    overall_score=overall_score
                )
        
        # Sort by overall score
        enhanced_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        return enhanced_matches
    
    def match_project_to_employees(self, project_id: int, top_k: int = 5) -> List[Dict]:
        """Find best employee matches for a project"""
        # Get project data
        projects = self.db.get_all_projects()
        project = next((p for p in projects if p.id == project_id), None)
        
        if not project:
            return []
        
        # Get project embedding
        if not project.embedding_vector:
            # Generate embedding if not exists
            project_text = f"{project.title} Required Skills: {', '.join(project.required_skills)} Description: {project.description}"
            project_embedding = self.vector_store.generate_embedding(project_text)
            self.db.update_project_embedding(project_id, project_embedding)
            self.vector_store.update_project_embedding(
                project_id, project.title, project.required_skills,
                project.description, project_embedding
            )
        else:
            project_embedding = project.embedding_vector
        
        # Find similar employees using vector search
        similar_employees = self.vector_store.find_similar_employees(project_embedding, top_k)
        
        # Enhance results with skill gap analysis
        enhanced_matches = []
        for employee_data in similar_employees:
            employee_id = employee_data['employee_id']
            employee = self.db.get_employee(employee_id)
            
            if employee:
                # Analyze skill gaps
                skill_gap_analysis = self.skill_analyzer.find_skill_gaps(
                    employee.skills, project.required_skills
                )
                
                enhanced_match = {
                    'employee_id': employee_id,
                    'name': employee.name,
                    'skills': employee.skills,
                    'preferences': employee.preferences,
                    'similarity_score': employee_data['similarity_score'],
                    'skill_match_percentage': skill_gap_analysis['match_percentage'],
                    'matched_skills': skill_gap_analysis['matched_skills'],
                    'missing_skills': skill_gap_analysis['missing_skills'],
                    'overall_score': self._calculate_overall_score(
                        employee_data['similarity_score'],
                        skill_gap_analysis['match_percentage']
                    )
                }
                enhanced_matches.append(enhanced_match)
        
        # Sort by overall score
        enhanced_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        return enhanced_matches
    
    def analyze_skill_gaps(self, employee_id: int, project_id: Optional[int] = None) -> Dict:
        """Analyze skill gaps for an employee"""
        employee = self.db.get_employee(employee_id)
        if not employee:
            return {}
        
        if project_id:
            # Analyze gaps for specific project
            projects = self.db.get_all_projects()
            project = next((p for p in projects if p.id == project_id), None)
            if project:
                skill_gap_analysis = self.skill_analyzer.find_skill_gaps(
                    employee.skills, project.required_skills
                )
                return {
                    'employee_id': employee_id,
                    'project_id': project_id,
                    'project_title': project.title,
                    'skill_gap_analysis': skill_gap_analysis,
                    'recommended_courses': self._get_recommended_courses(skill_gap_analysis['missing_skills'])
                }
        else:
            # Analyze general skill gaps based on all projects
            all_projects = self.db.get_all_projects()
            all_required_skills = set()
            
            for project in all_projects:
                all_required_skills.update(project.required_skills)
            
            skill_gap_analysis = self.skill_analyzer.find_skill_gaps(
                employee.skills, list(all_required_skills)
            )
            
            return {
                'employee_id': employee_id,
                'skill_gap_analysis': skill_gap_analysis,
                'recommended_courses': self._get_recommended_courses(skill_gap_analysis['missing_skills']),
                'skill_categories': self.skill_analyzer.categorize_skills(employee.skills)
            }
    
    def get_career_path_suggestions(self, employee_id: int) -> Dict:
        """Get career path suggestions for an employee"""
        employee = self.db.get_employee(employee_id)
        if not employee:
            return {}
        
        # Categorize current skills
        skill_categories = self.skill_analyzer.categorize_skills(employee.skills)
        
        # Find projects that match current skill level
        current_matches = self.match_employee_to_projects(employee_id, top_k=10)
        
        # Analyze skill gaps across all projects
        skill_gap_analysis = self.analyze_skill_gaps(employee_id)
        
        # Generate career path suggestions
        career_suggestions = {
            'current_level': self._assess_skill_level(employee.skills),
            'skill_categories': skill_categories,
            'current_matches': current_matches[:3],  # Top 3 current matches
            'skill_gaps': skill_gap_analysis.get('skill_gap_analysis', {}),
            'recommended_courses': skill_gap_analysis.get('recommended_courses', []),
            'next_level_projects': self._find_next_level_projects(employee.skills),
            'career_trajectory': self._suggest_career_trajectory(skill_categories)
        }
        
        return career_suggestions
    
    def _calculate_overall_score(self, similarity_score: float, skill_match_percentage: float) -> float:
        """Calculate overall matching score combining similarity and skill match"""
        # Weighted combination: 60% similarity, 40% skill match
        return round(0.6 * similarity_score + 0.4 * skill_match_percentage, 2)
    
    def _get_recommended_courses(self, missing_skills: List[str]) -> List[Dict]:
        """Get recommended courses for missing skills"""
        all_courses = self.db.get_all_courses()
        course_dicts = [
            {
                'id': course.id,
                'title': course.title,
                'skill_tags': course.skill_tags,
                'provider': course.provider,
                'url': course.url,
                'description': course.description
            }
            for course in all_courses
        ]
        
        return self.skill_analyzer.recommend_courses(missing_skills, course_dicts)
    
    def _assess_skill_level(self, skills: List[str]) -> str:
        """Assess overall skill level based on skills"""
        skill_count = len(skills)
        
        if skill_count < 5:
            return "Junior"
        elif skill_count < 15:
            return "Mid-level"
        elif skill_count < 25:
            return "Senior"
        else:
            return "Expert"
    
    def _find_next_level_projects(self, current_skills: List[str]) -> List[Dict]:
        """Find projects that would help employee grow to next level"""
        all_projects = self.db.get_all_projects()
        growth_projects = []
        
        for project in all_projects:
            skill_gap_analysis = self.skill_analyzer.find_skill_gaps(
                current_skills, project.required_skills
            )
            
            # Projects with 40-70% skill match are good for growth
            if 40 <= skill_gap_analysis['match_percentage'] <= 70:
                growth_projects.append({
                    'project_id': project.id,
                    'title': project.title,
                    'description': project.description,
                    'skill_match_percentage': skill_gap_analysis['match_percentage'],
                    'missing_skills': skill_gap_analysis['missing_skills']
                })
        
        # Sort by skill match percentage (ascending for growth potential)
        growth_projects.sort(key=lambda x: x['skill_match_percentage'])
        return growth_projects[:3]
    
    def _suggest_career_trajectory(self, skill_categories: Dict[str, List[str]]) -> List[str]:
        """Suggest career trajectory based on current skills"""
        trajectories = []
        
        # Analyze dominant skill areas
        category_counts = {cat: len(skills) for cat, skills in skill_categories.items()}
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_categories[0][0] == 'programming' and sorted_categories[0][1] > 3:
            trajectories.append("Software Engineer → Senior Software Engineer → Tech Lead")
        
        if 'data_science' in category_counts and category_counts['data_science'] > 2:
            trajectories.append("Data Analyst → Data Scientist → ML Engineer")
        
        if 'web_development' in category_counts and category_counts['web_development'] > 2:
            trajectories.append("Frontend Developer → Full-stack Developer → Solutions Architect")
        
        if 'cloud_devops' in category_counts and category_counts['cloud_devops'] > 2:
            trajectories.append("DevOps Engineer → Cloud Architect → Platform Engineer")
        
        if not trajectories:
            trajectories.append("General Developer → Specialized Developer → Technical Lead")
        
        return trajectories


# Global talent matcher instance
talent_matcher = TalentMatcher()
