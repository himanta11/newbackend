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
app.include_router(questions_router, prefix="/questions", tags=["questions"])

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
together_client = Together(api_key=TOGETHER_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat request/response models
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

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")
    logger.error(traceback.format_exc())

# Create routers
question_router = APIRouter(prefix="/api/questions", tags=["Questions"])

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
@question_router.post("/")
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

# Add explain endpoint
@question_router.post("/explain")
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
        
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        ai_response = response.json()
        chat_id = str(hash(request.question + str(request.options)))
        
        return QuestionExplanationResponse(
            response=ai_response["choices"][0]["message"]["content"],
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

        # Format the prompt for Together AI
        prompt = """You are a helpful AI assistant focused on helping students with their studies. You MUST ALWAYS provide complete, well-structured responses using markdown and include relevant emojis. Never give incomplete or single-line responses.

IMPORTANT: You MUST use emojis in your responses! Use them:
- At the start of each main section
- For important points
- For examples
- For summaries
- For key concepts

Example of a good response:
# Neural Networks 🧠

## What are Neural Networks? 💡
Neural networks are computing systems inspired by the human brain. They consist of:
- **Input Layer** 📥: Receives initial data
- **Hidden Layers** 🔄: Process the information
- **Output Layer** 📤: Produces the final result

## How They Work ⚡
1. **Data Input** 📝: Information enters through input nodes
2. **Processing** 🔧: 
   - Each connection has a **weight** ⚖️
   - Nodes perform calculations 🧮
   - Results pass through activation functions 📊
3. **Output** 🎯: Final result is produced

## Key Components 🔍
- **Neurons** 🧠: Basic processing units
- **Weights** ⚖️: Connection strengths
- **Bias** 🎯: Helps adjust output
- **Activation Functions** 📈: Determine neuron output

## Common Applications 📱
- Image recognition 👁️
- Natural language processing 💬
- Speech recognition 🗣️
- Game playing 🎮

## Learning Process 📚
> Neural networks learn by adjusting weights based on errors 🔄

## Summary 🎯
1. Neural networks mimic brain structure 🧠
2. They process information in layers 📚
3. They learn from examples 📖
4. They can solve complex problems 🎯

Remember:
1. ALWAYS provide complete explanations 📝
2. ALWAYS use markdown headings (# for main, ## for subsections) 📑
3. ALWAYS use bullet points (-) or numbers (1. 2. 3.) 📌
4. ALWAYS add relevant emojis 🎨
5. ALWAYS use **bold** for important terms 💪
6. ALWAYS use `code` for technical terms 💻
7. ALWAYS use > for important notes 📝
8. ALWAYS structure with clear sections 📋
9. NEVER give incomplete responses ❌
10. NEVER give single-line responses ❌
11. ALWAYS include multiple sections 📚
12. ALWAYS provide examples where relevant 📖
13. ALWAYS use emojis for visual appeal and emphasis 🎨

Now, let's begin our conversation:\n\n"""
        
        # Add history messages
        for msg in history:
            if msg.role == "user":
                prompt += f"Human: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        # Add current message
        prompt += f"Human: {request.message}\nAssistant:"

        logger.info(f"Prepared prompt for Together AI: {prompt}")

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        # Use completion endpoint
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

        logger.info("Making request to Together AI...")
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")

        # Make request to Together AI
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
                
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
                    
                    ai_message = ai_response["choices"][0]["text"].strip()
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

# Include routers
app.include_router(question_router)
app.include_router(auth_router)
app.include_router(questions_router)

# Event handler for startup
@app.on_event("startup")
def on_startup():
    logger.info("Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)