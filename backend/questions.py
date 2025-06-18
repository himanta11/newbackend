from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from database import get_db
from models import Question, ExamType, ExamStage, Subject
from pydantic import BaseModel
import logging
import traceback

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

class QuestionResponse(BaseModel):
    id: int
    exam_type: str
    exam_stage: str
    subject: str
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: Optional[str] = None
    year: Optional[int] = None

    class Config:
        from_attributes = True

class QuestionFilter(BaseModel):
    exam_type: Optional[str] = None
    exam_stage: Optional[str] = None
    subject: Optional[str] = None
    limit: int = 10

@router.post("/questions", response_model=List[QuestionResponse])
async def get_questions(
    filter: QuestionFilter,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received request with filters: {filter}")
        
        # Start with base query
        query = db.query(Question)
        logger.info("Base query created")

        # Apply filters if provided
        if filter.exam_type:
            logger.info(f"Filtering by exam_type: {filter.exam_type}")
            query = query.filter(Question.exam_type == filter.exam_type)
        if filter.exam_stage:
            logger.info(f"Filtering by exam_stage: {filter.exam_stage}")
            query = query.filter(Question.exam_stage == filter.exam_stage)
        if filter.subject:
            logger.info(f"Filtering by subject: {filter.subject}")
            query = query.filter(Question.subject == filter.subject)

        # Order by random and limit results
        logger.info(f"Limiting results to {filter.limit}")
        questions = query.order_by(func.random()).limit(filter.limit).all()
        logger.info(f"Found {len(questions)} questions")

        if not questions:
            logger.warning("No questions found for the given filters")
            raise HTTPException(
                status_code=404,
                detail="No questions found for the given filters"
            )

        return questions

    except Exception as e:
        logger.error(f"Error in get_questions: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching questions: {str(e)}"
        ) 