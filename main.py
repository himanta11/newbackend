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
from fastapi.responses import JSONResponse, StreamingResponse
from questions import router as questions_router
from fastapi.routing import APIRouter
from pydantic import BaseModel
import os
import json
from sqlalchemy import text
import requests
from together import Together
from dotenv import load_dotenv
import time

# Initialize FastAPI app with CORS configuration
app = FastAPI(
    title="Aspirant Backend",
    description="Backend service for Aspirant application",
    version="1.0.0"
)

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://newfrontend-sage.vercel.app",    # Production frontend
        "http://localhost:5500",                  # Local development
        "http://127.0.0.1:5500"                   # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(questions_router, tags=["questions"])

# Add startup event
@app.on_event("startup")
async def startup_event():
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Application started and database initialized")

# Load environment variables
load_dotenv()

# Initialize Together AI client with hardcoded API key
TOGETHER_API_KEY = "39b58efc9f06bc95aeb6a246badf5561100d6247136a4cd33bc6f2c96cc9d6bf"
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
together_client = Together(api_key=TOGETHER_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat request/response models
class Message(BaseModel):
    role: str
    content: str

class FormatRequest(BaseModel):
    type: str
    style: str
    structured: bool = True

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    history: Optional[List[Message]] = None
    questionType: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[FormatRequest] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")
    logger.error(traceback.format_exc())

# Models
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

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/signup", response_model=schemas.Token)
async def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Attempting to create user with email: {user.email}")
        
        # Validate email format
        if not user.email or '@' not in user.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Validate password length
        if not user.password or len(user.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
        
        # Validate username length
        if not user.username or len(user.username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters long")
        
        # Check if email exists
        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if db_user:
            logger.warning(f"Email {user.email} already exists")
            raise HTTPException(status_code=400, detail="Email already registered")
            
        # Check if username exists
        if user.username:
            db_user = db.query(models.User).filter(models.User.username == user.username).first()
            if db_user:
                logger.warning(f"Username {user.username} already exists")
                raise HTTPException(status_code=400, detail="Username already taken")
        
        # Create new user
        try:
            new_user = models.User(
                email=user.email,
                username=user.username
            )
            new_user.set_password(user.password)
            logger.info("User object created successfully")
            
            # Add to database
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            logger.info("User added to database successfully")
            
            # Create access token
            access_token = create_access_token(data={"sub": user.email})
            logger.info("Access token created successfully")
            return {"access_token": access_token, "token_type": "bearer"}
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/token", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Try to find user by email first
        user = db.query(models.User).filter(models.User.email == form_data.username).first()
        if not user or not user.verify_password(form_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/me")
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email}

# Question Router
@questions_router.post("/")
async def get_questions(filters: QuestionFilters):
    try:
        logger.info(f"Received request with filters: {filters}")
        db = SessionLocal()
        
        # Check if questions table exists and has data
        try:
            result = db.execute(text("SELECT COUNT(*) FROM questions"))
            count = result.scalar()
            logger.info(f"Number of questions in database: {count}")
        except Exception as e:
            logger.error(f"Error checking questions table: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Database error: Questions table may not exist"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                    "Access-Control-Allow-Credentials": "true",
                    "Content-Type": "application/json"
                }
            )
        
        # Build the SQL query
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
        
        # Add limit and randomize
        query += " ORDER BY RANDOM() LIMIT :limit"
        params['limit'] = filters.limit or 50
        
        logger.info(f"Executing query: {query}")
        logger.info(f"With parameters: {params}")
        
        try:
            result = db.execute(text(query), params)
            questions = result.fetchall()
            logger.info(f"Found {len(questions)} questions")
            
            # Convert to list of dicts
            questions_list = []
            for q in questions:
                question_dict = {
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
                }
                questions_list.append(question_dict)
            
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
            logger.error(f"Error executing query: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Error executing query: {str(e)}"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                    "Access-Control-Allow-Credentials": "true",
                    "Content-Type": "application/json"
                }
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in get_questions: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json"
            }
        )
    finally:
        db.close()

# Add explicit OPTIONS handler for questions endpoint
@questions_router.options("/")
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

# Add explain endpoint
@questions_router.post("/explain")
async def explain_question(request: QuestionExplanationRequest):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Format the prompt for the AI
        options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(request.options)])
        prompt = f"""Please explain this question in a friendly and simple way:

Question: {request.question}

Options:
{options_text}

Correct Answer: {request.correct_answer}

Current Explanation: {request.explanation or 'No explanation provided'}

Please provide:
1. A simple, easy-to-understand explanation of the concept
2. Break down the question in simple steps
3. Explain why the correct answer is right in a friendly way
4. Use emojis to make it engaging
5. End with an encouraging message and invite questions

Remember to:
- Use simple language
- Be friendly and encouraging
- Add relevant emojis
- Make it engaging and fun to read
- End with "Feel free to ask if you have any doubts! 😊"
"""
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a friendly and encouraging AI tutor. Follow these rules strictly:
- Use simple, easy-to-understand language
- Be warm and friendly in your tone
- Use emojis to make explanations engaging
- Break down complex concepts into simple steps
- Encourage questions and interaction
- End with an invitation for follow-up questions
- If asked general questions, answer them in a friendly way
- Use markdown formatting for better readability
- Make learning fun and engaging
- If user says "thank you", respond warmly and encourage further interaction
- Be conversational and maintain a friendly chat
- Handle both academic and general questions
- Keep the conversation going naturally
- If user asks about other topics, engage in friendly conversation
- Remember previous context and maintain conversation flow"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(TOGETHER_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        ai_response = response.json()
        logger.info(f"Together AI response: {ai_response}")
        
        if not ai_response.get("choices") or not ai_response["choices"][0].get("message"):
            logger.error(f"Unexpected Together AI response format: {ai_response}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response from AI service"
            )
        
        explanation = ai_response["choices"][0]["message"]["content"]
        if not explanation or explanation.strip() == "😊":
            logger.error("Together AI returned empty or emoji-only response")
            explanation = f"""I apologize, but I'm having trouble generating an explanation right now. Here's what we know:

📝 Question: {request.question}

✅ Correct Answer: {request.correct_answer}

{request.explanation or "I'll try to provide a better explanation soon. Feel free to try again!"}

Feel free to ask if you have any questions! 😊"""
        
        chat_id = str(hash(request.question + str(request.options)))
        
        return QuestionExplanationResponse(
            response=explanation,
            chat_id=chat_id
        )
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.dict()}")
        
        # Initialize chat history if not provided
        chat_id = request.chat_id or str(datetime.now().timestamp())
        history = request.history or []

        # Define headers for Together AI API
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        # Get question analysis from frontend
        question_type = request.questionType or "explanation"
        complexity = request.complexity or "simple"
        format_style = request.format.style if request.format else "natural"
        
        # Create dynamic system message based on question type and complexity
        def get_system_message(q_type, comp):
            base_system = """You are an expert AI tutor specializing in educational content. Always respond in well-formatted markdown."""
            
            if q_type == "definition":
                return f"""{base_system}
                
For definition questions:
- Start with a clear, concise definition
- Provide 2-3 key characteristics
- Include practical examples
- Use markdown formatting with headers and bullet points
- Keep it {'detailed' if comp == 'complex' else 'concise'}"""
                
            elif q_type == "explanation":
                return f"""{base_system}
                
For explanation questions:
- Use clear markdown headers (# ## ###)
- Break down complex concepts into digestible sections
- Use bullet points and numbered lists appropriately
- Include examples and analogies
- {'Provide comprehensive coverage with multiple sections' if comp == 'complex' else 'Keep explanations focused and direct'}"""
                
            elif q_type == "steps":
                return f"""{base_system}
                
For step-by-step questions:
- Use numbered lists (1. 2. 3.)
- Create clear section headers
- Provide detailed instructions for each step
- Include tips or notes where helpful
- Format with proper markdown structure"""
                
            elif q_type == "list":
                return f"""{base_system}
                
For list-based questions:
- Use bullet points or numbered lists as appropriate
- Group related items under subheadings
- Provide brief explanations for each item
- Use markdown formatting for clarity"""
                
            elif q_type == "comparison":
                return f"""{base_system}
                
For comparison questions:
- Use markdown tables when appropriate
- Create clear sections for each item being compared
- Highlight key differences and similarities
- Use headers to organize the comparison"""
                
            else:  # default/shortAnswer
                return f"""{base_system}
                
Provide clear, well-structured responses using appropriate markdown formatting including headers, lists, and emphasis where helpful."""

        # Build the conversation messages
        messages = [
            {
                "role": "system",
                "content": get_system_message(question_type, complexity)
            }
        ]
        
        # Add conversation history
        for msg in history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg.role if msg.role in ["user", "assistant"] else "user",
                "content": msg.content
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Use chat completion endpoint with proper parameters
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": None
        }

        logger.info("Making request to Together AI...")
        logger.info(f"Question type: {question_type}, Complexity: {complexity}")

        # Make request to Together AI
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    TOGETHER_CHAT_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                
                if response.ok:
                    ai_response = response.json()
                    logger.info(f"Received response from Together AI: {ai_response}")
                    
                    if "choices" not in ai_response or not ai_response["choices"]:
                        error_msg = "Invalid response format from Together AI"
                        logger.error(error_msg)
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying... Attempt {attempt + 2} of {max_retries}")
                            time.sleep(retry_delay)
                            continue
                        return JSONResponse(
                            status_code=500,
                            content={"detail": error_msg},
                            headers={
                                "Access-Control-Allow-Origin": "*",
                                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                                "Access-Control-Allow-Credentials": "true",
                                "Content-Type": "application/json"
                            }
                        )
                    
                    # Extract message content from chat completion response
                    choice = ai_response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        ai_message = choice["message"]["content"].strip()
                    elif "text" in choice:
                        ai_message = choice["text"].strip()
                    else:
                        error_msg = "No content found in AI response"
                        logger.error(f"{error_msg}: {ai_response}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying... Attempt {attempt + 2} of {max_retries}")
                            time.sleep(retry_delay)
                            continue
                        ai_message = "I apologize, but I'm having trouble generating a response right now. Please try again."
                    
                    return JSONResponse(
                        content=ChatResponse(
                            response=ai_message,
                            chat_id=chat_id
                        ).dict(),
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                            "Access-Control-Allow-Credentials": "true",
                            "Content-Type": "application/json"
                        }
                    )
                else:
                    error_msg = f"Together AI API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    
                    # Check if we should retry based on status code
                    if response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                        logger.info(f"Retrying... Attempt {attempt + 2} of {max_retries}")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    
                    return JSONResponse(
                        status_code=500,
                        content={"detail": error_msg},
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                            "Access-Control-Allow-Credentials": "true",
                            "Content-Type": "application/json"
                        }
                    )
                    
            except requests.exceptions.Timeout:
                error_msg = "Request to Together AI timed out"
                logger.error(error_msg)
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... Attempt {attempt + 2} of {max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return JSONResponse(
                    status_code=500,
                    content={"detail": error_msg},
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                        "Access-Control-Allow-Credentials": "true",
                        "Content-Type": "application/json"
                    }
                )
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {str(e)}"
                logger.error(error_msg)
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... Attempt {attempt + 2} of {max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return JSONResponse(
                    status_code=500,
                    content={"detail": error_msg},
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                        "Access-Control-Allow-Credentials": "true",
                        "Content-Type": "application/json"
                    }
                )
        
        # If we get here, all retries failed
        return JSONResponse(
            status_code=500,
            content={"detail": "All retry attempts failed"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json"
            }
        )
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": error_msg},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json"
            }
        )

# Event handler for startup
@app.on_event("startup")
def on_startup():
    logger.info("Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)