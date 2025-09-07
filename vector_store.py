"""
Vector store operations using ChromaDB for HR Talent Matching System
Handles embedding generation and similarity search
"""

import chromadb
from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
import hashlib


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client and simple embedding model"""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use a simple embedding function for now
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collections
        self.employees_collection = self.client.get_or_create_collection(
            name="employees",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.projects_collection = self.client.get_or_create_collection(
            name="projects", 
            metadata={"hnsw:space": "cosine"}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for given text using hash-based approach"""
        # Create a simple embedding using text hashing
        # This is a temporary solution until we fix the sentence transformers issue
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a 384-dimensional vector (same as all-MiniLM-L6-v2)
        embedding = []
        for i in range(0, len(text_hash), 2):
            # Convert hex pairs to float values
            hex_pair = text_hash[i:i+2]
            val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(val)
        
        # Pad or truncate to exactly 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        embedding = embedding[:384]
        
        return embedding
    
    def add_employee_embedding(self, employee_id: int, name: str, skills: List[str], 
                              resume_text: str, embedding: List[float]):
        """Add employee embedding to ChromaDB"""
        # Combine skills and resume text for better matching
        combined_text = f"{name} Skills: {', '.join(skills)} Resume: {resume_text}"
        
        self.employees_collection.add(
            ids=[str(employee_id)],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "employee_id": employee_id,
                "name": name,
                "skills": json.dumps(skills)
            }]
        )
    
    def add_project_embedding(self, project_id: int, title: str, required_skills: List[str],
                             description: str, embedding: List[float]):
        """Add project embedding to ChromaDB"""
        # Combine title, skills, and description for better matching
        combined_text = f"{title} Required Skills: {', '.join(required_skills)} Description: {description}"
        
        self.projects_collection.add(
            ids=[str(project_id)],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "project_id": project_id,
                "title": title,
                "required_skills": json.dumps(required_skills)
            }]
        )
    
    def find_similar_projects(self, employee_embedding: List[float], 
                            top_k: int = 5) -> List[Dict]:
        """Find most similar projects for an employee"""
        results = self.projects_collection.query(
            query_embeddings=[employee_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        similar_projects = []
        if results['ids'] and results['ids'][0]:
            for i, project_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                
                # Convert distance to similarity score (0-100%)
                similarity_score = (1 - distance) * 100
                
                similar_projects.append({
                    "project_id": int(project_id),
                    "title": metadata["title"],
                    "required_skills": json.loads(metadata["required_skills"]),
                    "similarity_score": round(similarity_score, 2),
                    "document": document
                })
        
        return similar_projects
    
    def find_similar_employees(self, project_embedding: List[float],
                              top_k: int = 5) -> List[Dict]:
        """Find most similar employees for a project"""
        results = self.employees_collection.query(
            query_embeddings=[project_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        similar_employees = []
        if results['ids'] and results['ids'][0]:
            for i, employee_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                
                # Convert distance to similarity score (0-100%)
                similarity_score = (1 - distance) * 100
                
                similar_employees.append({
                    "employee_id": int(employee_id),
                    "name": metadata["name"],
                    "skills": json.loads(metadata["skills"]),
                    "similarity_score": round(similarity_score, 2),
                    "document": document
                })
        
        return similar_employees
    
    def update_employee_embedding(self, employee_id: int, name: str, skills: List[str],
                                 resume_text: str, embedding: List[float]):
        """Update existing employee embedding"""
        # Delete existing embedding
        try:
            self.employees_collection.delete(ids=[str(employee_id)])
        except:
            pass  # Employee might not exist in vector store yet
        
        # Add new embedding
        self.add_employee_embedding(employee_id, name, skills, resume_text, embedding)
    
    def update_project_embedding(self, project_id: int, title: str, required_skills: List[str],
                                description: str, embedding: List[float]):
        """Update existing project embedding"""
        # Delete existing embedding
        try:
            self.projects_collection.delete(ids=[str(project_id)])
        except:
            pass  # Project might not exist in vector store yet
        
        # Add new embedding
        self.add_project_embedding(project_id, title, required_skills, description, embedding)
    
    def get_embedding_for_text(self, text: str) -> List[float]:
        """Get embedding for any text (useful for skill gap analysis)"""
        return self.generate_embedding(text)


# Global vector store instance
vector_store = VectorStore()
