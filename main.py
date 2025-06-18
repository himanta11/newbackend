from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
import models
import schemas
from database import SessionLocal, engine, Base
from auth import create_access_token, get_current_user, router as auth_router
from datetime import datetime, timedelta
import logging
import traceback
from fastapi.responses import JSONResponse
from questions import router as questions_router
from fastapi.routing import APIRouter
from pydantic import BaseModel
import os
import json
from sqlalchemy import text
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app with CORS configuration
app = FastAPI(
    title="Aspirant Backend",
    description="Backend service for Aspirant application",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS with more specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "https://*.netlify.app",
        "https://newfrontend-sage.vercel.app",
        "https://*.github.io",
        "*"  # Allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
TOGETHER_API_KEY = "39b58efc9f06bc95aeb6a246badf5561100d6247136a4cd33bc6f2c96cc9d6bf"
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    history: Optional[List[Message]] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str

class QuestionFilters(BaseModel):
    exam_type: Optional[str] = None
    topics: Optional[List[str]] = None
    has_diagram: Optional[bool] = None
    limit: Optional[int] = 50
    subject: Optional[str] = None
    year: Optional[int] = None
    exam_stage: Optional[str] = None

class QuestionExplanationRequest(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None

class QuestionExplanationResponse(BaseModel):
    response: str
    chat_id: str

# Create routers
question_router = APIRouter(prefix="/api/questions", tags=["Questions"])

# Middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json"
            }
        )

# Database initialization
@app.on_event("startup")
async def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Auth endpoints
@app.post("/signup", response_model=schemas.Token)
async def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Attempting to create user with email: {user.email}")
        
        # Validate input
        if not user.email or '@' not in user.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        if not user.password or len(user.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        if not user.username or len(user.username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
        
        # Check if user exists
        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
            
        if user.username:
            db_user = db.query(models.User).filter(models.User.username == user.username).first()
            if db_user:
                raise HTTPException(status_code=400, detail="Username already taken")
        
        # Create new user
        new_user = models.User(
            email=user.email,
            username=user.username
        )
        new_user.set_password(user.password)
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        access_token = create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/token", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not user.verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email}

# Question endpoints
@question_router.post("/")
async def get_questions(filters: QuestionFilters, db: Session = Depends(get_db)):
    try:
        logger.info(f"Received request with filters: {filters}")
        
        # Build query
        query = """
            SELECT id, question_text, option_a, option_b, option_c, option_d, 
                   correct_answer, explanation, topic, exam_type, has_diagram,
                   subject, year, exam_stage, difficulty_level
            FROM questions
            WHERE 1=1
        """
        params = {}
        
        if filters.exam_type:
            query += " AND exam_type = :exam_type"
            params['exam_type'] = filters.exam_type
        
        if filters.has_diagram is not None:
            query += " AND has_diagram = :has_diagram"
            params['has_diagram'] = filters.has_diagram
        
        if filters.topics:
            topic_conditions = []
            for i, topic in enumerate(filters.topics):
                param_name = f"topic_{i}"
                topic_conditions.append(f"topic = :{param_name}")
                params[param_name] = topic.replace('_', ' ').title()
            if topic_conditions:
                query += " AND (" + " OR ".join(topic_conditions) + ")"
        
        if filters.subject:
            query += " AND subject = :subject"
            params['subject'] = filters.subject
            
        if filters.year:
            query += " AND year = :year"
            params['year'] = filters.year
            
        if filters.exam_stage:
            query += " AND exam_stage = :exam_stage"
            params['exam_stage'] = filters.exam_stage
        
        query += " ORDER BY RANDOM() LIMIT :limit"
        params['limit'] = filters.limit or 50
        
        result = db.execute(text(query), params)
        questions = result.fetchall()
        
        questions_list = []
        for q in questions:
            questions_list.append({
                "id": q.id,
                "question_text": q.question_text,
                "option_a": q.option_a,
                "option_b": q.option_b,
                "option_c": q.option_c,
                "option_d": q.option_d,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "topic": q.topic,
                "exam_type": q.exam_type,
                "has_diagram": q.has_diagram,
                "subject": q.subject,
                "year": q.year,
                "exam_stage": q.exam_stage,
                "difficulty_level": q.difficulty_level
            })
        
        return JSONResponse(
            content={"questions": questions_list},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json"
            }
        )
            
    except Exception as e:
        logger.error(f"Error in get_questions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@question_router.options("/")
async def options_questions():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "application/json"
        }
    )

@question_router.post("/explain")
async def explain_question(request: QuestionExplanationRequest):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(request.options)])
        prompt = f"""Please explain this question in a friendly and simple way:

Question: {request.question}

Options:
{options_text}

Correct Answer: {request.correct_answer}

Current Explanation: {request.explanation or 'No explanation provided'}

Please provide a simple, easy-to-understand explanation with emojis to make it engaging."""

        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["Human:", "\n\n"]
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                ai_response = response.json()
                chat_id = str(hash(request.question + str(request.options)))
                
                return QuestionExplanationResponse(
                    response=ai_response["choices"][0]["text"].strip(),
                    chat_id=chat_id
                )
                
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
                
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation")

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        chat_id = request.chat_id or str(datetime.now().timestamp())
        history = request.history or []

        prompt = """You are a helpful AI assistant focused on helping students with their studies. Provide complete, well-structured responses using markdown and emojis.\n\n"""
        
        for msg in history:
            if msg.role == "user":
                prompt += f"Human: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        prompt += f"Human: {request.message}\nAssistant:"

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["Human:", "\n\n"]
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                ai_response = response.json()
                return ChatResponse(
                    response=ai_response["choices"][0]["text"].strip(),
                    chat_id=chat_id
                )
                
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
                
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(questions_router, prefix="/questions", tags=["questions"])
app.include_router(question_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
