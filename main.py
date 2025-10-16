from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
import random
import smtplib
from twilio.rest import Client   # <-- Uncomment when using SMS

from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
import pytz

from sqlalchemy.orm import Session
from DataBase.database import SessionLocal, engine, Base
from DataBase.models import User
from DataBase import curd

import os

from dotenv import load_dotenv
from RAG.src.rag_chain import create_rag_chain
from typing import Optional

load_dotenv()  # Load .env automatically
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment!")


app = FastAPI(title="KrishiSetu RAG Backend", version="1.0")

# Temporary OTP store (use Redis or DB in production)
otp_store = {}

class OTPRequest(BaseModel):
    phone: str | None = None
    email: EmailStr | None = None

class VerifyRequest(BaseModel):
    identifier: str
    otp: str

def generate_otp():
    """Generate 6-digit OTP"""
    return str(random.randint(100000, 999999))

# -------- EMAIL SENDING --------
def send_email(to_email: str, otp: str):
    sender = "your@gmail.com"
    password = "pfgxxybyewaglrne"  # Create App Password in Gmail settings
    subject = "Your OTP Code"
    message = f"Subject: {subject}\n\nYour OTP is {otp}"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, to_email, message)
        print(f"✅ Email sent to {to_email} with OTP: {otp}")
    except Exception as e:
        print(f"❌ Email failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

# -------- SMS SENDING (commented out for now) --------
def send_sms(to_phone: str, otp: str):
    account_sid = "AC7cce1388063f23f64f1e7b8dc9e3959b"
    auth_token = "f27c070153cf96a04cc7b7c6cfcb7c4d"
    client = Client(account_sid, auth_token)
    try:
        message = client.messages.create(
            body=f"Your OTP is {otp}",
            from_="+12792394257",   # Your Twilio phone number
            to=to_phone
        )
        print(f"✅ SMS sent to {to_phone} with OTP: {otp}")
        return message.sid
    except Exception as e:
        print(f"❌ SMS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send SMS: {e}")

@app.post("/send-otp")
def send_otp(request: OTPRequest):
    """
    FIXED: The logic was backwards!
    - If phone is provided -> send SMS
    - If email is provided -> send Email
    """
    otp = generate_otp()

    # CASE 1: Phone number provided -> Send SMS
    if request.phone:
        send_sms(request.phone, otp)   # <-- Uncomment when using SMS
        print(f"📱 DEBUG: OTP {otp} for phone {request.phone}")  # dev only
        otp_store[request.phone] = otp
        return {"message": "OTP sent via SMS (debug mode)", "otp": otp}  # Remove 'otp' in production

    # CASE 2: Email provided -> Send Email
    elif request.email:
        send_email(request.email, otp)
        otp_store[request.email] = otp
        print(f"📧 Email sent to {request.email}")
        return {"message": "OTP sent via Email"}

    # CASE 3: Neither provided -> Error
    else:
        raise HTTPException(status_code=400, detail="Phone or Email required")

@app.post("/verify-otp")
def verify_otp(request: VerifyRequest):
    """Verify the OTP"""
    stored_otp = otp_store.get(request.identifier)
    
    print(f"🔍 Verifying OTP for: {request.identifier}")
    print(f"📝 Stored OTP: {stored_otp}, Provided OTP: {request.otp}")
    
    if not stored_otp:
        raise HTTPException(status_code=400, detail="No OTP found or expired")
    
    if stored_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # OTP verified -> remove from store
    del otp_store[request.identifier]
    print(f"✅ OTP verified successfully for {request.identifier}")
    return {"message": "OTP verified successfully"}

# CORS support (if your Flutter app needs it)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Flutter app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class UserInfo(BaseModel):
    name: str
    city: str

# --- Helper: Determine greeting based on current time ---
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register-db")
def register_user_db(user: dict, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user["email"]).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    curd.create_user(
        db, user["name"], user["city"], user["email"], user["phone"], user["password"]
    )
    return {"message": "User stored successfully"}

@app.post("/login-db")
def login_user(user: dict, db: Session = Depends(get_db)):
    valid = curd.verify_user(db, user["email"], user["password"])
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user": valid.name}

# --- Endpoint: Get user dashboard data ---
@app.post("/dashboard")
def get_dashboard(user: UserInfo):
    """
    Returns:
    - Greeting message
    - Real date
    - Real-time weather for given city
    """

    # Fetch real-time weather data
    api_key = "ba859c6b236435bf5f6cc15f0545ac67"  # <-- Replace with your actual OpenWeatherMap API key
    city = user.city
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")

        # Extract weather data
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        weather_main = data["weather"][0]["main"]
        precipitation = data.get("rain", {}).get("1h", 0.0)

        # Get current date
        tz = pytz.timezone("Asia/Kathmandu")  # Change if needed
        date_today = datetime.now(tz).strftime("%A, %d %b %Y")

        # Greeting
        greeting = get_greeting()

        return {
            "greeting": f"{greeting}, {user.name}",
            "city": city,
            "date": date_today,
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "wind": f"{wind_speed} m/s",
            "precipitation": f"{precipitation} mm",
            "weather": weather_main,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.post("/register")
def register_user(data: dict):
    name = data.get("name")
    city = data.get("city")
    email = data.get("email")
    phone = data.get("phone")

    # Store or process user data (for now just print it)
    print(f"👤 New user registered: {name} ({city})")
    return {"message": "User registered successfully", "user": data}

# Initialize RAG chain
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE")
rag_chain = create_rag_chain(API_KEY)

# Request and Response models
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "English"  # default English

class ChatResponse(BaseModel):
    response: str
    source: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Here you can handle different languages if needed
        query_text = request.message
        lang = request.language.lower()

        # If you want, you can do language-specific preprocessing
        if lang in ["नेपाली", "nepali"]:
            query_text = f"Translate and respond in Nepali: {query_text}"
        elif lang in ["हिन्दी", "hindi"]:
            query_text = f"Translate and respond in Hindi: {query_text}"
        # English default doesn't need changes

        # Query RAG chain
        result = rag_chain.query(query_text)

        # If result is a dict with 'answer' and 'source', else fallback
        answer = result.get("answer") if isinstance(result, dict) else str(result)
        source = result.get("source") if isinstance(result, dict) else None

        return ChatResponse(response=answer, source=source)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
