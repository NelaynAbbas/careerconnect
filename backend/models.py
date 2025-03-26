from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
  __tablename__ = 'users'
  
  id = db.Column(db.Integer, primary_key=True)
  email = db.Column(db.String(255), unique=True, nullable=False)
  password = db.Column(db.String(255), nullable=False)
  full_name = db.Column(db.String(255), nullable=False)
  user_type = db.Column(db.String(50), nullable=False)  # 'jobSeeker' or 'employer'
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
  
  # Relationships
  profile = db.relationship('Profile', backref='user', uselist=False)
  skills = db.relationship('Skill', backref='user')
  job_preferences = db.relationship('JobPreference', backref='user', uselist=False)
  experiences = db.relationship('Experience', backref='user')
  education = db.relationship('Education', backref='user')
  jobs = db.relationship('Job', backref='employer')
  applications = db.relationship('JobApplication', backref='applicant')

class Profile(db.Model):
  __tablename__ = 'profiles'
  
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  
  # Job seeker fields
  title = db.Column(db.String(255))
  bio = db.Column(db.Text)
  phone = db.Column(db.String(20))
  location = db.Column(db.String(255))
  resume_url = db.Column(db.String(255))
  
  # Employer fields
  company_name = db.Column(db.String(255))
  industry = db.Column(db.String(255))
  company_size = db.Column(db.String(255))
  founded_year = db.Column(db.String(4))
  company_website = db.Column(db.String(255))
  company_location = db.Column(db.String(255))
  company_description = db.Column(db.Text)
  logo_url = db.Column(db.String(255))
  contact_name = db.Column(db.String(255))
  contact_title = db.Column(db.String(255))
  contact_email = db.Column(db.String(255))
  contact_phone = db.Column(db.String(20))
  linkedin_url = db.Column(db.String(255))
  twitter_url = db.Column(db.String(255))
  facebook_url = db.Column(db.String(255))
  
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

class Skill(db.Model):
  __tablename__ = 'skills'
  
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  name = db.Column(db.String(100), nullable=False)
  created_at = db.Column(db.DateTime, default=datetime.utcnow)

class JobPreference(db.Model):
  __tablename__ = 'job_preferences'
  
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  job_types = db.Column(db.Text)  # JSON string array
  locations = db.Column(db.Text)  # JSON string array
  industries = db.Column(db.Text)  # JSON string array
  min_salary = db.Column(db.String(50))
  availability = db.Column(db.String(50))
  remote_preference = db.Column(db.String(50))
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

class Experience(db.Model):
  __tablename__ = 'experiences'
  
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  title = db.Column(db.String(255), nullable=False)
  company = db.Column(db.String(255), nullable=False)
  location = db.Column(db.String(255))
  start_date = db.Column(db.DateTime)
  end_date = db.Column(db.DateTime)
  current = db.Column(db.Boolean, default=False)
  description = db.Column(db.Text)
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

class Education(db.Model):
  __tablename__ = 'education'
  
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  degree = db.Column(db.String(255), nullable=False)
  institution = db.Column(db.String(255), nullable=False)
  location = db.Column(db.String(255))
  start_date = db.Column(db.DateTime)
  end_date = db.Column(db.DateTime)
  current = db.Column(db.Boolean, default=False)
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

class Job(db.Model):
  __tablename__ = 'jobs'
  
  id = db.Column(db.Integer, primary_key=True)
  employer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  title = db.Column(db.String(255), nullable=False)
  company = db.Column(db.String(255))
  location = db.Column(db.String(255))
  type = db.Column(db.String(50))  # Full-time, Part-time, etc.
  salary = db.Column(db.String(100))
  description = db.Column(db.Text, nullable=False)
  requirements = db.Column(db.Text)  # JSON string array
  responsibilities = db.Column(db.Text)  # JSON string array
  benefits = db.Column(db.Text)  # JSON string array
  is_remote = db.Column(db.Boolean, default=False)
  experience_level = db.Column(db.String(50))
  application_deadline = db.Column(db.DateTime)
  application_email = db.Column(db.String(255))
  application_url = db.Column(db.String(255))
  is_active = db.Column(db.Boolean, default=True)
  is_draft = db.Column(db.Boolean, default=False)
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
  
  # Relationships
  applications = db.relationship('JobApplication', backref='job')
  
  def to_dict(self):
    return {
      'id': self.id,
      'title': self.title,
      'company': self.company,
      'location': self.location,
      'type': self.type,
      'salary': self.salary,
      'description': self.description,
      'requirements': json.loads(self.requirements) if self.requirements else [],
      'responsibilities': json.loads(self.responsibilities) if self.responsibilities else [],
      'benefits': json.loads(self.benefits) if self.benefits else [],
      'isRemote': self.is_remote,
      'experienceLevel': self.experience_level,
      'applicationDeadline': self.application_deadline.isoformat() if self.application_deadline else None,
      'applicationEmail': self.application_email,
      'applicationUrl': self.application_url,
      'isActive': self.is_active,
      'isDraft': self.is_draft,
      'postedDate': self.created_at.isoformat(),
      'applicationsCount': len(self.applications)
    }

class JobApplication(db.Model):
  __tablename__ = 'job_applications'
  
  id = db.Column(db.Integer, primary_key=True)
  job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
  applicant_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
  resume_url = db.Column(db.String(255))
  cover_letter = db.Column(db.Text)
  status = db.Column(db.String(50), default='pending')  # pending, reviewed, interviewed, rejected, hired
  feedback = db.Column(db.Text)  # For rejection feedback
  interview_notes = db.Column(db.Text)  # For interview feedback
  offer_details = db.Column(db.Text)  # For job offer details
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
  
  def to_dict(self):
    return {
      'id': self.id,
      'jobId': self.job_id,
      'applicantId': self.applicant_id,
      'resumeUrl': self.resume_url,
      'coverLetter': self.cover_letter,
      'status': self.status,
      'feedback': self.feedback,
      'interviewNotes': self.interview_notes,
      'offerDetails': self.offer_details,
      'appliedDate': self.created_at.isoformat(),
      'updatedAt': self.updated_at.isoformat() if self.updated_at else None
    }

