import numpy as np
from gensim.models import Word2Vec
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# Path to the Word2Vec model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'w2vmodel', 'job_word2vec_large.model')

def load_model():
    """Load the Word2Vec model"""
    try:
        model = Word2Vec.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None

def preprocess_text(text):
    """Clean and tokenize text"""
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize by whitespace
    tokens = text.split()
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

def get_document_vector(model, document):
    """Convert document to vector by averaging word vectors"""
    tokens = preprocess_text(document)
    
    # Filter tokens that are in the model's vocabulary
    tokens = [token for token in tokens if token in model.wv]
    
    if not tokens:
        return None
    
    # Get the word vectors for each token
    word_vectors = [model.wv[token] for token in tokens]
    
    # Return the average of the word vectors
    return np.mean(word_vectors, axis=0)

def extract_skills_from_job(model, job_description):
    """Extract skills from job description using the Word2Vec model"""
    # This is a placeholder - in a real implementation, you'd use the model to
    # identify skills in the description or query a skills database
    job_vector = get_document_vector(model, job_description)
    
    if job_vector is None:
        return []
    
    # Get most similar words to the job description
    # These would be the skills most relevant to the job
    similar_words = model.wv.most_similar(positive=[job_vector], topn=20)
    
    # Return the skill words
    return [word for word, _ in similar_words]

def calculate_match_score(model, job_seeker_skills, job_description, job_requirements=None):
    """
    Calculate the match score between a job seeker and a job
    
    Parameters:
    - model: Word2Vec model
    - job_seeker_skills: List of job seeker skills
    - job_description: Job description text
    - job_requirements: List of job requirements (if available)
    
    Returns:
    - Match score (0-100)
    """
    if not job_seeker_skills or not job_description:
        return 0
    
    # If job requirements are not provided, extract them from the description
    if not job_requirements:
        job_requirements = extract_skills_from_job(model, job_description)
    
    # Convert job seeker skills to vectors
    seeker_vectors = []
    for skill in job_seeker_skills:
        if skill in model.wv:
            seeker_vectors.append(model.wv[skill])
    
    # Convert job requirements to vectors
    job_vectors = []
    for req in job_requirements:
        if req in model.wv:
            job_vectors.append(model.wv[req])
    
    if not seeker_vectors or not job_vectors:
        return 0
    
    # Calculate cosine similarity between each skill and each requirement
    similarities = []
    for seeker_vec in seeker_vectors:
        for job_vec in job_vectors:
            sim = cosine_similarity([seeker_vec], [job_vec])[0][0]
            similarities.append(sim)
    
    # Calculate overall match score
    avg_similarity = np.mean(similarities)
    
    # Convert to percentage (0-100)
    match_score = int(avg_similarity * 100)
    
    # Apply some adjustments to make it realistic
    # If the seeker has few of the required skills, reduce the score
    coverage = min(1.0, len(seeker_vectors) / max(1, len(job_vectors)))
    match_score = int(match_score * (0.5 + 0.5 * coverage))
    
    return match_score

def calculate_seeker_to_jobs_scores(model, job_seeker_skills, jobs):
    """
    Calculate match scores between a job seeker and multiple jobs
    
    Parameters:
    - model: Word2Vec model
    - job_seeker_skills: List of job seeker skills
    - jobs: List of job objects with descriptions
    
    Returns:
    - List of (job, score) tuples sorted by score
    """
    job_scores = []
    
    for job in jobs:
        score = calculate_match_score(model, job_seeker_skills, job.description)
        job_scores.append((job, score))
    
    # Sort by score (highest first)
    job_scores.sort(key=lambda x: x[1], reverse=True)
    
    return job_scores

def calculate_job_to_seekers_scores(model, job_description, job_seekers):
    """
    Calculate match scores between a job and multiple job seekers
    
    Parameters:
    - model: Word2Vec model
    - job_description: Job description text
    - job_seekers: List of job seeker objects with skills
    
    Returns:
    - List of (job_seeker, score) tuples sorted by score
    """
    seeker_scores = []
    
    for seeker in job_seekers:
        # Get skills for this seeker
        skills = [skill.name for skill in seeker.skills]
        
        score = calculate_match_score(model, skills, job_description)
        seeker_scores.append((seeker, score))
    
    # Sort by score (highest first)
    seeker_scores.sort(key=lambda x: x[1], reverse=True)
    
    return seeker_scores

