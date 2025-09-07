"""
Utility functions for HR Talent Matching System
Handles resume parsing, skill extraction, and text processing
"""

import re
import json
from typing import List, Dict, Set, Tuple
from collections import Counter


class ResumeParser:
    """Simple resume parser for extracting skills and information"""
    
    def __init__(self):
        # Common technical skills patterns
        self.technical_skills = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'fastapi', 'node.js', 'asp.net', 'tensorflow',
                'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'sqlite', 'oracle', 'sql server', 'cassandra', 'dynamodb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
                'microsoft azure', 'kubernetes', 'docker', 'terraform'
            ],
            'tools': [
                'git', 'jenkins', 'jira', 'confluence', 'slack', 'figma',
                'photoshop', 'illustrator', 'tableau', 'power bi', 'excel'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
                'microservices', 'rest api', 'graphql', 'machine learning', 'ai'
            ]
        }
        
        # Soft skills patterns
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'critical thinking', 'time management', 'project management',
            'mentoring', 'collaboration', 'adaptability', 'creativity',
            'analytical', 'detail oriented', 'self motivated', 'initiative'
        ]
    
    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = resume_text.lower()
        found_skills = set()
        
        # Extract technical skills
        for category, skills in self.technical_skills.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.add(skill)
        
        # Extract soft skills
        for skill in self.soft_skills:
            if skill in text_lower:
                found_skills.add(skill)
        
        # Extract skills from common patterns
        skill_patterns = [
            r'skills?[:\s]*([^.\n]+)',
            r'technologies?[:\s]*([^.\n]+)',
            r'expertise[:\s]*([^.\n]+)',
            r'proficient in[:\s]*([^.\n]+)',
            r'experience with[:\s]*([^.\n]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters and clean up
                skills_in_match = re.split(r'[,;|•\n]', match)
                for skill in skills_in_match:
                    skill = skill.strip()
                    if len(skill) > 2 and len(skill) < 50:  # Reasonable skill length
                        found_skills.add(skill)
        
        return list(found_skills)
    
    def extract_name(self, resume_text: str) -> str:
        """Extract name from resume text"""
        lines = resume_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 2 and len(line) < 50:
                # Simple heuristic: first non-empty line that looks like a name
                if not any(char.isdigit() for char in line) and not '@' in line:
                    return line
        return "Unknown"
    
    def extract_experience_years(self, resume_text: str) -> int:
        """Extract years of experience from resume text"""
        text_lower = resume_text.lower()
        
        # Look for experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?field',
            r'(\d+)\+?\s*years?\s*of\s*(?:professional\s*)?experience'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    return int(matches[0])
                except ValueError:
                    continue
        
        return 0
    
    def extract_education(self, resume_text: str) -> List[str]:
        """Extract education information"""
        text_lower = resume_text.lower()
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r'bachelor[^s]*\s*(?:of\s*)?(?:science|arts|engineering|business|computer science)',
            r'master[^s]*\s*(?:of\s*)?(?:science|arts|engineering|business|computer science)',
            r'phd|doctorate|ph\.d\.',
            r'associate[^s]*\s*(?:of\s*)?(?:science|arts)',
            r'diploma|certificate'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))


class SkillAnalyzer:
    """Analyze skills and detect gaps"""
    
    def __init__(self):
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust'],
            'web_development': ['react', 'angular', 'vue', 'html', 'css', 'node.js', 'express'],
            'data_science': ['python', 'r', 'sql', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn'],
            'cloud_devops': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite'],
            'mobile': ['swift', 'kotlin', 'react native', 'flutter', 'ios', 'android'],
            'ai_ml': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch']
        }
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different areas"""
        categorized = {category: [] for category in self.skill_categories.keys()}
        categorized['other'] = []
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized_flag = False
            
            for category, category_skills in self.skill_categories.items():
                if any(cat_skill in skill_lower for cat_skill in category_skills):
                    categorized[category].append(skill)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(skill)
        
        return categorized
    
    def find_skill_gaps(self, employee_skills: List[str], 
                       required_skills: List[str]) -> Dict[str, List[str]]:
        """Find missing skills for a project"""
        employee_skills_lower = [skill.lower() for skill in employee_skills]
        missing_skills = []
        
        for required_skill in required_skills:
            required_lower = required_skill.lower()
            if not any(required_lower in emp_skill or emp_skill in required_lower 
                      for emp_skill in employee_skills_lower):
                missing_skills.append(required_skill)
        
        return {
            'missing_skills': missing_skills,
            'matched_skills': [skill for skill in required_skills if skill not in missing_skills],
            'match_percentage': round((len(required_skills) - len(missing_skills)) / len(required_skills) * 100, 2) if required_skills else 0
        }
    
    def recommend_courses(self, missing_skills: List[str], 
                         available_courses: List[Dict]) -> List[Dict]:
        """Recommend courses based on missing skills"""
        recommendations = []
        
        for course in available_courses:
            course_skills = course.get('skill_tags', [])
            course_skills_lower = [skill.lower() for skill in course_skills]
            
            # Calculate relevance score
            relevance_score = 0
            for missing_skill in missing_skills:
                missing_lower = missing_skill.lower()
                if any(missing_lower in course_skill or course_skill in missing_lower 
                      for course_skill in course_skills_lower):
                    relevance_score += 1
            
            if relevance_score > 0:
                recommendations.append({
                    **course,
                    'relevance_score': relevance_score,
                    'relevance_percentage': round(relevance_score / len(missing_skills) * 100, 2)
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations[:3]  # Top 3 recommendations


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text.strip()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stop words and short words
    keywords = [word for word in words 
                if word not in stop_words and len(word) >= min_length]
    
    # Count frequency and return most common
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(20)]


# Global instances
resume_parser = ResumeParser()
skill_analyzer = SkillAnalyzer()
