from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, Float, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from database import Base
from passlib.context import CryptContext
import logging
import enum
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic schemas
class UserProgressSchema(BaseModel):
    id: int
    user_id: int
    question_id: int
    correct: int  # Changed from is_correct to correct
    time_taken: Optional[float] = None
    attempted_at: datetime

    class Config:
        orm_mode = True

class UserSchema(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class QuestionSchema(BaseModel):
    id: int
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: Optional[str] = None
    has_diagram: bool
    diagram_description: Optional[str] = None
    year: int
    exam_type: str
    exam_stage: str
    subject: str
    topic: Optional[str] = None
    difficulty_level: str
    source: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class TagSchema(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        orm_mode = True

class QuestionTagSchema(BaseModel):
    question_id: int
    tag_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class QuestionStatisticsSchema(BaseModel):
    question_id: int
    total_attempts: int
    correct_attempts: int
    average_time: float
    last_updated: datetime

    class Config:
        orm_mode = True

class QuestionImageSchema(BaseModel):
    id: int
    question_id: int
    image_path: str
    image_type: Optional[str] = None
    caption: Optional[str] = None
    display_order: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# SQLAlchemy models
class ExamType(enum.Enum):
    NTPC = "NTPC"
    GROUP_D = "GROUP D"
    JE = "JE"
    SSC = "SSC"
    CGL = "CGL"

class ExamStage(enum.Enum):
    CBT1 = "CBT 1"
    CBT2 = "CBT 2"
    CBT3 = "CBT 3"
    PET = "PET"
    DV = "DV"
    TIER_1 = "Tier 1"
    TIER_2 = "Tier 2"
    TIER_3 = "Tier 3"
    TIER_4 = "Tier 4"

class Subject(enum.Enum):
    GENERAL_AWARENESS = "General Awareness"
    ARITHMETIC = "Arithmetic"
    GENERAL_INTELLIGENCE = "General Intelligence & Reasoning"
    BASIC_SCIENCE = "Basic Science & Engineering"
    TECHNICAL_ABILITIES = "Technical Abilities"
    REASONING = "Reasoning"
    LOGICAL_REASONING = "Logical Reasoning"
    GENERAL_SCIENCE = "General Science"
    MATHEMATICS = "Mathematics"
    SCIENCE = "Science"
    CURRENT_AFFAIRS = "Current Affairs"
    HISTORY = "History"
    GEOGRAPHY = "Geography"
    POLITY = "Polity"
    BIOLOGY = "Biology"
    CHEMISTRY = "Chemistry"
    INDIAN_CULTURE = "Indian Culture"
    ENVIRONMENT = "Environment"
    COMPUTER = "Computer"
    RAILWAY_AWARENESS = "Railway Awareness"
    INTERNATIONAL_AFFAIRS = "International Affairs"
    BANKING = "Banking"
    SCIENCE_AND_TECH = "Science and Tech"
    PHYSICS = "Physics"
    ENGLISH_GRAMMAR = "English Grammar"

class DifficultyLevel(enum.Enum):
    EASY = "Easy"
    MODERATE = "Moderate"
    HARD = "Hard"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def set_password(self, password: str):
        """Hash and set the user's password"""
        try:
            if not password:
                raise ValueError("Password cannot be empty")
            if len(password) < 6:
                raise ValueError("Password must be at least 6 characters long")
            self.hashed_password = pwd_context.hash(password)
            logger.info("Password hashed successfully")
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            raise ValueError(f"Error hashing password: {str(e)}")

    def verify_password(self, password: str) -> bool:
        """Verify the user's password"""
        try:
            if not password or not self.hashed_password:
                logger.warning("Password or hashed_password is empty")
                return False
            is_valid = pwd_context.verify(password, self.hashed_password)
            if not is_valid:
                logger.warning("Invalid password")
            return is_valid
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False

    def to_schema(self):
        return UserSchema(
            id=self.id,
            email=self.email,
            username=self.username,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    option_a = Column(Text, nullable=False)
    option_b = Column(Text, nullable=False)
    option_c = Column(Text, nullable=False)
    option_d = Column(Text, nullable=False)
    correct_answer = Column(String(1), nullable=False)  # A, B, C, or D
    explanation = Column(Text)
    
    # Diagram/Image related fields
    has_diagram = Column(Boolean, default=False)
    diagram_description = Column(Text)  # Description of the diagram for accessibility
    
    # Metadata
    year = Column(Integer, nullable=False)
    exam_type = Column(Enum(ExamType), nullable=False)
    exam_stage = Column(Enum(ExamStage), nullable=False)
    subject = Column(Enum(Subject), nullable=False)
    topic = Column(String(100))  # Optional topic within subject
    
    # Additional metadata
    difficulty_level = Column(Enum(DifficultyLevel), default=DifficultyLevel.MODERATE)
    source = Column(String(255))  # Source of the question (e.g., "Previous Year Paper", "Practice Set")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)