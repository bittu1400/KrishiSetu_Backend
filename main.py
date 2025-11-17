from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import smtplib
import requests
import pytz
from datetime import datetime
from typing import Optional
import random
import numpy as np
import io
import os
from PIL import Image
import logging

# Database and RAG imports - SPECIFIC ORDER MATTERS
from DataBase.database import SessionLocal, engine, Base, init_db, check_db_health, close_db
from DataBase import curd  
from DataBase.models import User, Expert, Booking  # Import models last
from RAG.chatbot import get_chatbot_response
import json
# TensorFlow and Keras imports
try:
    from tf_keras.models import load_model
    from tf_keras.preprocessing import image as keras_image
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: TensorFlow not installed. Using mock detection.")

if os.getenv("ENVIRONMENT") != "PRODUCTION":
    load_dotenv()


ENV = os.getenv("ENV")
DATABASE_URL=os.getenv("DATABASE_URLENV")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN")
SENDER_EMAIL=os.getenv("SENDER_EMAIL")
SENDER_PASSWORD=os.getenv("SENDER_PASSWORD")

logging.basicConfig(
    level=logging.DEBUG if ENV == "development" else logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def validate_environment():
    """Validate all required environment variables are set."""
    required_vars = {
        "DATABASE_URL": "Neon PostgreSQL connection string",
        "GROQ_API_KEY": "Groq API key for chatbot",
        "OPENWEATHER_API_KEY": "OpenWeather API key",
        "TWILIO_ACCOUNT_SID": "Twilio account SID",
        "TWILIO_AUTH_TOKEN": "Twilio auth token",
        "SENDER_EMAIL": "Gmail sender email",
        "SENDER_PASSWORD": "Gmail app password"
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"  - {var}: {description}")
    
    if missing:
        error_msg = "❌ Missing required environment variables:\n" + "\n".join(missing)
        raise RuntimeError(error_msg)
    
    print("✅ All environment variables validated")

app = FastAPI(title="KrishiSetu RAG Backend", version="1.0")

validate_environment()



# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary OTP store (use Redis or DB in production)
otp_store = {}

# Pydantic models
class OTPRequest(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None

class VerifyRequest(BaseModel):
    identifier: str
    otp: str

# Pydantic model for registration
class RegisterRequest(BaseModel):
    name: str
    city: str
    email: EmailStr
    phone: str
    password: str  # Changed from 'pin' to 'password'
    
# Update the LoginRequest model to accept phone or email
class LoginRequest(BaseModel):
    identifier: str  # Can be email or phone
    password: str

class DashboardRequest(BaseModel):
    email: str

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "English"

class ChatResponse(BaseModel):
    response: str
    source: Optional[str] = None

# Utility functions
def generate_otp():
    """Generate a 6-digit OTP."""
    return str(random.randint(100000, 999999))

def send_email(to_email: str, otp: str):
    """Send OTP via email."""    
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
    subject = "Your OTP Code"
    message = f"Subject: {subject}\n\nYour OTP is {otp}"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, message)
        print(f"✅ Email sent to {to_email} with OTP: {otp}")
    except Exception as e:
        print(f"❌ Email failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

def send_sms(to_phone: str, otp: str):
    from twilio.rest import Client
    account_sid = TWILIO_ACCOUNT_SID
    auth_token = TWILIO_AUTH_TOKEN
    client = Client(account_sid, auth_token)
    try:
        message = client.messages.create(
            body=f"Your OTP is {otp}",
            from_="+19412997133",
            to=to_phone
        )
        print(f"✅ SMS sent to {to_phone} with OTP: {otp}")
        return message.sid
    except Exception as e:
        print(f"❌ SMS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send SMS: {e}")


# Database setup
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Model loading and disease detection
# MODEL_PATH = "Trained_model.keras"
MODEL_PATH = "trained_model.h5"
with open('class_names.json', 'r') as f:
    DISEASE_CLASSES = json.load(f)

DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab": {
        "recommendation": "Prune infected leaves and improve air circulation.",
        "treatment": "Spray Mancozeb or Captan every 7–10 days."
    },
    "Apple___Black_rot": {
        "recommendation": "Remove mummified fruits and prune cankers.",
        "treatment": "Use a Captan or Thiophanate-methyl spray weekly."
    },
    "Apple___Cedar_apple_rust": {
        "recommendation": "Remove nearby juniper hosts and protect new growth.",
        "treatment": "Apply Myclobutanil every 10 days."
    },
    "Apple___healthy": {
        "recommendation": "Tree is healthy; maintain sanitation.",
        "treatment": "None needed."
    },
    "Blueberry___healthy": {
        "recommendation": "Healthy plant; ensure good mulching and watering.",
        "treatment": "None needed."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "recommendation": "Remove affected shoots and avoid overcrowding.",
        "treatment": "Spray Sulfur or Potassium bicarbonate weekly."
    },
    "Cherry_(including_sour)___healthy": {
        "recommendation": "Healthy tree; keep pruning balanced.",
        "treatment": "None needed."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "recommendation": "Rotate crops and remove old debris.",
        "treatment": "Apply Strobilurin-based fungicide every 7 days."
    },
    "Corn_(maize)___Common_rust_": {
        "recommendation": "Use resistant varieties and monitor humidity.",
        "treatment": "Spray Mancozeb or Triazole fungicide weekly."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "recommendation": "Increase spacing and remove crop residue.",
        "treatment": "Use a Triazole + Strobilurin mix every 10 days."
    },
    "Corn_(maize)___healthy": {
        "recommendation": "Crop is healthy; maintain fertilizer schedule.",
        "treatment": "None needed."
    },
    "Grape___Black_rot": {
        "recommendation": "Remove infected leaves and improve canopy airflow.",
        "treatment": "Spray Mancozeb or Myclobutanil every 7–14 days."
    },
    "Grape___Esca_(Black_Measles)": {
        "recommendation": "Avoid pruning wounds and remove infected vines.",
        "treatment": "No cure; apply Trichoderma-based protectants."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "recommendation": "Improve air flow and remove infected leaves.",
        "treatment": "Use Copper fungicide every 10 days."
    },
    "Grape___healthy": {
        "recommendation": "Vine is healthy; maintain irrigation and training.",
        "treatment": "None needed."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "recommendation": "Remove infected trees and control psyllid insects.",
        "treatment": "No cure; apply Imidacloprid for vector control."
    },
    "Peach___Bacterial_spot": {
        "recommendation": "Avoid overhead watering and prune diseased twigs.",
        "treatment": "Spray Copper-based fungicide every 7 days."
    },
    "Peach___healthy": {
        "recommendation": "Tree is healthy; maintain mulch and watering.",
        "treatment": "None required."
    },
    "Pepper,_bell___Bacterial_spot": {
        "recommendation": "Use drip irrigation and remove infected leaves.",
        "treatment": "Apply Copper + Mancozeb mix weekly."
    },
    "Pepper,_bell___healthy": {
        "recommendation": "Plant is healthy; maintain balanced nutrients.",
        "treatment": "None needed."
    },
    "Potato___Early_blight": {
        "recommendation": "Remove old foliage and avoid leaf wetness.",
        "treatment": "Spray Mancozeb or Chlorothalonil every 7 days."
    },
    "Potato___Late_blight": {
        "recommendation": "Ensure good airflow and avoid overhead irrigation.",
        "treatment": "Use Metalaxyl or Ridomil Gold every 5 days."
    },
    "Potato___healthy": {
        "recommendation": "Healthy crop; maintain regular monitoring.",
        "treatment": "None required."
    },
    "Raspberry___healthy": {
        "recommendation": "Healthy plant; keep weeds down and water evenly.",
        "treatment": "None needed."
    },
    "Soybean___healthy": {
        "recommendation": "Healthy crop; ensure nitrogen balance.",
        "treatment": "None needed."
    },
    "Squash___Powdery_mildew": {
        "recommendation": "Improve airflow and remove infected leaves.",
        "treatment": "Spray Potassium bicarbonate or Sulfur weekly."
    },
    "Strawberry___Leaf_scorch": {
        "recommendation": "Remove infected leaves and avoid overhead irrigation.",
        "treatment": "Use Captan fungicide every 7 days."
    },
    "Strawberry___healthy": {
        "recommendation": "Plant is healthy; maintain mulching.",
        "treatment": "None needed."
    },
    "Tomato___Bacterial_spot": {
        "recommendation": "Avoid leaf wetness and remove infected foliage.",
        "treatment": "Copper sprays every 5–7 days."
    },
    "Tomato___Early_blight": {
        "recommendation": "Remove bottom leaves and improve airflow.",
        "treatment": "Spray Mancozeb or Chlorothalonil every 7 days."
    },
    "Tomato___Late_blight": {
        "recommendation": "Prevent leaf wetness and space plants well.",
        "treatment": "Spray Metalaxyl or Ridomil Gold every 5 days."
    },
    "Tomato___Leaf_Mold": {
        "recommendation": "Increase ventilation and avoid high humidity.",
        "treatment": "Use Copper or Chlorothalonil weekly."
    },
    "Tomato___Septoria_leaf_spot": {
        "recommendation": "Remove infected leaves and avoid splash watering.",
        "treatment": "Apply Mancozeb or Chlorothalonil every 7 days."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "recommendation": "Rinse leaves and reduce plant stress.",
        "treatment": "Use Abamectin or Neem oil every 5 days."
    },
    "Tomato___Target_Spot": {
        "recommendation": "Improve airflow and remove lower infected leaves.",
        "treatment": "Spray Copper or Mancozeb weekly."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "recommendation": "Control whiteflies and remove infected plants.",
        "treatment": "No cure; use Imidacloprid for vector control."
    },
    "Tomato___Tomato_mosaic_virus": {
        "recommendation": "Avoid handling plants when wet and sanitize tools.",
        "treatment": "No cure; remove infected plants."
    },
    "Tomato___healthy": {
        "recommendation": "Plant is healthy; keep nutrient levels balanced.",
        "treatment": "None needed."
    }
}

if MODEL_AVAILABLE:
    try:
        MODEL = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        MODEL = None
        print(f"❌ Failed to load model: {e}")

def preprocess_image(image_bytes):
    """Convert raw image bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image(image_bytes):
    """Convert raw image bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(preprocessed_image):
    """Run model prediction and return structured output."""
    try:
        if MODEL_AVAILABLE and MODEL is not None:
            preds = MODEL.predict(preprocessed_image)
            class_index = np.argmax(preds[0])
            confidence = float(np.max(preds[0]) * 100)
        else:
            class_index = random.randint(0, len(DISEASE_CLASSES) - 1)
            confidence = random.uniform(70, 99)
        
        # Get disease name
        if class_index < len(DISEASE_CLASSES):
            disease_full = DISEASE_CLASSES[class_index]
            disease_name = disease_full.split("___")[-1].replace("_", " ")
        else:
            disease_full = "Unknown"
            disease_name = "Unknown Disease"
        
        return {
            "class_index": int(class_index),
            "disease": disease_name,
            "disease_full": disease_full,  # Optional: full name with crop
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

# Example endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/send-otp")
def send_otp(request: OTPRequest, db: Session = Depends(get_db)):
    """Send OTP to phone (SMS) or email."""
    from DataBase.models import OTP  # Import your OTP model
    from datetime import timedelta
    
    otp_code = generate_otp()
    identifier = request.phone or request.email
    
    if not identifier:
        raise HTTPException(status_code=400, detail="Phone or Email required")
    
    # Store OTP in database instead of memory
    expires_at = datetime.utcnow() + timedelta(minutes=10)  # 10 min expiry
    
    # Delete old OTP if exists
    db.query(OTP).filter(OTP.identifier == identifier).delete()
    
    # Create new OTP
    new_otp = OTP(
        identifier=identifier,
        otp=otp_code,
        expires_at=expires_at
    )
    db.add(new_otp)
    db.commit()
    
    # Send OTP via SMS or Email
    if request.phone:
        phone = request.phone.strip()
        if not phone.startswith('+'):
            phone = f"+977{phone}" if len(phone) == 10 else phone
        
        try:
            send_sms(phone, otp_code)
            return {"message": "OTP sent via SMS"}  # DON'T return actual OTP!
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send SMS: {str(e)}")
    
    elif request.email:
        try:
            send_email(request.email, otp_code)
            return {"message": "OTP sent via Email"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")
       
@app.post("/verify-otp")
def verify_otp(request: VerifyRequest, db: Session = Depends(get_db)):
    """Verify the OTP from database."""
    from DataBase.models import OTP
    
    # Find OTP in database
    stored_otp = db.query(OTP).filter(OTP.identifier == request.identifier).first()
    
    if not stored_otp:
        raise HTTPException(status_code=400, detail="No OTP found or expired")
    
    # Check if expired
    if stored_otp.expires_at < datetime.utcnow():
        db.delete(stored_otp)
        db.commit()
        raise HTTPException(status_code=400, detail="OTP expired")
    
    # Verify OTP
    if stored_otp.otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # Delete OTP after successful verification
    db.delete(stored_otp)
    db.commit()
    
    return {"message": "OTP verified successfully"}

# Register endpoint
@app.post("/register")
def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user in the database."""
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if phone already exists
    existing_phone = db.query(User).filter(User.phone == data.phone).first()
    if existing_phone:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    try:
        new_user = curd.create_user(
            db=db,
            name=data.name,
            city=data.city,
            email=data.email,
            phone=data.phone,
            password=data.password  # Changed from 'pin' to 'password'
        )
        print(f"✅ New user registered: {data.name} ({data.city})")
        return {
            "message": "User registered successfully",
            "user": {
                "name": data.name,
                "email": data.email,
                "phone": data.phone,
                "city": data.city
            }
        }
    except Exception as e:
        db.rollback()
        print(f"❌ Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Add/Update these Pydantic models
class UserLoginRequest(BaseModel):
    identifier: str  # Can be email or phone
    password: str

# Update USER LOGIN endpoint
@app.post("/login")
def login_user(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Login user with email/phone and password."""
    identifier = request.identifier.strip()
    
    print(f"🔍 Login attempt - Identifier: {identifier}")
    
    # Try to find user by email or phone
    user = None
    if '@' in identifier:
        # It's an email
        user = db.query(User).filter(User.email == identifier).first()
    else:
        # It's a phone number
        user = db.query(User).filter(User.phone == identifier).first()
    
    if not user:
        print(f"❌ User not found: {identifier}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not curd.verify_password(request.password, user.password):
        print(f"❌ Invalid password for user: {identifier}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    print(f"✅ Login successful: {user.name}")
    
    return {
        "message": "Login successful",
        "user": {
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "city": user.city
        }
    }

@app.post("/dashboard")
def get_dashboard(request: DashboardRequest, db: Session = Depends(get_db)):
    """Return user information, greeting, date, and real-time weather."""
    db_user = db.query(User).filter(User.email == request.email).first()
    
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_name = db_user.name
    user_email = db_user.email
    user_phone = db_user.phone
    user_city = db_user.city
    
    # Fetch weather data
    url = f"https://api.openweathermap.org/data/2.5/weather?q={user_city}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("cod") != 200:
            raise HTTPException(status_code=404, detail=f"City '{user_city}' not found")
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        weather_main = data["weather"][0]["main"]
        precipitation = data.get("rain", {}).get("1h", 0.0)
        
        # Get current date and greeting
        tz = pytz.timezone("Asia/Kathmandu")
        date_today = datetime.now(tz).strftime("%A, %d %b %Y")
        hour = datetime.now(tz).hour
        greeting = (
            "Good Morning" if 5 <= hour < 12 else
            "Good Afternoon" if 12 <= hour < 18 else
            "Good Evening"
        )
        
        return {
            "name": user_name,
            "email": user_email,
            "phone": user_phone,
            "city": user_city,
            "greeting": greeting,
            "date": date_today,
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "wind": f"{wind_speed} m/s",
            "precipitation": f"{precipitation} mm",
            "weather": weather_main,
        }
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests using RAG chatbot."""
    result = get_chatbot_response(request.message, request.language)
    return ChatResponse(response=result.get("response", ""), source=result.get("source"))

@app.post("/diagnose")
async def diagnose_crop(file: UploadFile = File(...)):
    """Diagnose crop disease from uploaded image."""
    print("\n" + "="*50)
    print("🔍 DIAGNOSE ENDPOINT CALLED")
    print(f"📁 File name: {file.filename}")
    print(f"📝 Content type: {file.content_type}")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]:
            print(f"❌ Invalid content type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload JPG or PNG image."
            )
        
        # Read and validate file size
        image_data = await file.read()
        print(f"✅ Read {len(image_data)} bytes")
        
        if not image_data:
            print("❌ Empty file!")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(image_data) > MAX_FILE_SIZE:
            print(f"❌ File too large: {len(image_data)} bytes")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 10MB, got {len(image_data) / 1024 / 1024:.2f}MB"
            )
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_data)  # make sure this function exists
        print("✅ Image preprocessed successfully")
        
        # Predict disease
        prediction = predict_disease(preprocessed_image)  # make sure this function exists
        print(f"✅ Prediction: {prediction}")
        
        # Build response
        response = {
            "status": "success",
            # "file_name": file.filename,
            # "prediction": prediction,  # e.g., {"disease": "Apple Cedar Rust", "confidence": 0.93}
            "class_index": prediction["class_index"],
            "disease": prediction["disease"],
            "confidence": prediction["confidence"],
            "recommendation": DISEASE_RECOMMENDATIONS.get(prediction["disease_full"], {}).get("recommendation", "No recommendation available."),
        }
        
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException as e:
        print(f"❌ HTTP Exception: {e.detail}")
        return JSONResponse(
            content={"error": e.detail, "status": "error"},
            status_code=e.status_code
        )
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Error processing image: {str(e)}", "status": "error"},
            status_code=500
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Crop Disease Detection API",
        "version": "1.0",
        "endpoints": {
            "diagnose": "/diagnose (POST)",
            "health": "/health (GET)",
            "send-otp": "/send-otp (POST)",
            "verify-otp": "/verify-otp (POST)",
            "register": "/register (POST)",
            "login": "/login (POST)",
            "dashboard": "/dashboard (POST)",
            "chat": "/chat (POST)"
        }
    }

class ExpertRegisterRequest(BaseModel):
    name: str
    title: str
    experience: str
    email: str
    phone: str
    specialization: str
    price: str
    features: str
    password: str
    bio: Optional[str] = ""
    category: Optional[str] = "All"

class ExpertLoginRequest(BaseModel):
    identifier: str
    password: str

class ExpertUpdateStatusRequest(BaseModel):
    expert_id: int
    status: str  # "Available", "Busy", "Offline"

# Add to main.py - API Endpoints
@app.post("/expert/register")
def register_expert(data: ExpertRegisterRequest, db: Session = Depends(get_db)):
    """Register a new expert."""
    # Check if expert already exists
    existing_expert = db.query(Expert).filter(Expert.email == data.email).first()
    if existing_expert:
        raise HTTPException(status_code=400, detail="Expert already registered with this email")
    
    try:
        new_expert = curd.create_expert(
            db=db,
            name=data.name,
            title=data.title,
            experience=data.experience,
            email=data.email,
            phone=data.phone,
            specialization=data.specialization,
            price=data.price,
            features=data.features,
            password=data.password,
            bio=data.bio,
            category=data.category
        )
        
        print(f"✅ New expert registered: {data.name}")
        
        return {
            "message": "Expert registered successfully",
            "expert": {
                "id": new_expert.id,
                "name": new_expert.name,
                "email": new_expert.email,
                "title": new_expert.title
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/expert/login")
def login_expert(request: ExpertLoginRequest, db: Session = Depends(get_db)):
    """Login expert with email and password."""
    expert = curd.verify_expert(db, request.identifier, request.password)
    
    if not expert:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "message": "Login successful",
        "expert": {
            "id": expert.id,
            "name": expert.name,
            "email": expert.email,
            "phone": expert.phone,
            "title": expert.title,
            "experience": expert.experience,
            "specialization": expert.specialization,
            "price": expert.price,
            "features": expert.features,
            "rating": expert.rating,
            "reviews": expert.reviews,
            "status": expert.status,
            "bio": expert.bio,
            "category": expert.category
        }
    }

@app.get("/experts")
def get_experts(category: Optional[str] = None, db: Session = Depends(get_db)):
    """Get all experts, optionally filtered by category."""
    try:
        experts = curd.get_all_experts(db, category)
        
        return {
            "experts": [
                {
                    "id": expert.id,
                    "name": expert.name,
                    "title": expert.title,
                    "experience": expert.experience,
                    "email": expert.email,
                    "phone": expert.phone,
                    "specialization": expert.specialization,
                    "price": expert.price,
                    "features": expert.features,
                    "rating": expert.rating,
                    "reviews": expert.reviews,
                    "status": expert.status,
                    "bio": expert.bio,
                    "category": expert.category
                }
                for expert in experts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching experts: {str(e)}")

@app.post("/expert/update-status")
def update_expert_status_endpoint(request: ExpertUpdateStatusRequest, db: Session = Depends(get_db)):
    """Update expert availability status."""
    try:
        expert = curd.update_expert_status(db, request.expert_id, request.status)
        if not expert:
            raise HTTPException(status_code=404, detail="Expert not found")
        
        return {
            "message": "Status updated successfully",
            "status": expert.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating status: {str(e)}")

# KEEP THESE in main.py - these are Pydantic models for API validation
class BookingCreate(BaseModel):
    user_email: str
    expert_id: int
    expert_name: str
    expert_title: str
    specialization: str
    price: str
    booking_date: str
    status: str = "pending"

class BookingStatusUpdate(BaseModel):
    status: str

# Add these endpoints to main.py
@app.post("/bookings", response_model=dict)
def create_booking_endpoint(booking: BookingCreate, db: Session = Depends(get_db)):
    """Create a new booking."""
    try:
        # Parse datetime
        booking_datetime = datetime.fromisoformat(booking.booking_date.replace('Z', '+00:00'))
        
        # Create booking
        new_booking = curd.create_booking(
            db=db,
            user_email=booking.user_email,
            expert_id=booking.expert_id,
            expert_name=booking.expert_name,
            expert_title=booking.expert_title,
            specialization=booking.specialization,
            price=booking.price,
            booking_date=booking_datetime,
            status=booking.status
        )
        
        if not new_booking:
            raise HTTPException(status_code=404, detail="User not found")
        
        print(f"✅ Booking created: {new_booking.booking_id}")
        
        return {
            "message": "Booking created successfully",
            "booking_id": new_booking.booking_id,
            "booking": {
                "booking_id": new_booking.booking_id,
                "user_id": new_booking.user_id,
                "expert_id": new_booking.expert_id,
                "expert_name": new_booking.expert_name,
                "expert_title": new_booking.expert_title,
                "specialization": new_booking.specialization,
                "price": new_booking.price,
                "booking_date": new_booking.booking_date.isoformat(),
                "status": new_booking.status,
                "created_at": new_booking.created_at.isoformat()
            }
        }
    
    except Exception as e:
        print(f"❌ Error creating booking: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create booking: {str(e)}")

@app.get("/bookings")
def get_bookings_endpoint(
    user_email: str,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all bookings for a user."""
    try:
        bookings = curd.get_user_bookings(db, user_email, status)
        
        return {
            "bookings": [
                {
                    "booking_id": booking.booking_id,
                    "user_id": booking.user_id,
                    "expert_id": booking.expert_id,
                    "expert_name": booking.expert_name,
                    "expert_title": booking.expert_title,
                    "specialization": booking.specialization,
                    "price": booking.price,
                    "booking_date": booking.booking_date.isoformat(),
                    "status": booking.status,
                    "created_at": booking.created_at.isoformat()
                }
                for booking in bookings
            ]
        }
    
    except Exception as e:
        print(f"❌ Error fetching bookings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch bookings: {str(e)}")

@app.get("/bookings/{booking_id}")
def get_booking_endpoint(booking_id: str, db: Session = Depends(get_db)):
    """Get a single booking by ID."""
    try:
        booking = curd.get_booking_by_id(db, booking_id)
        
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        return {
            "booking_id": booking.booking_id,
            "user_id": booking.user_id,
            "expert_id": booking.expert_id,
            "expert_name": booking.expert_name,
            "expert_title": booking.expert_title,
            "specialization": booking.specialization,
            "price": booking.price,
            "booking_date": booking.booking_date.isoformat(),
            "status": booking.status,
            "created_at": booking.created_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching booking: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch booking: {str(e)}")

@app.patch("/bookings/{booking_id}")
def update_booking_endpoint(
    booking_id: str,
    update: BookingStatusUpdate,
    db: Session = Depends(get_db)
):
    """Update booking status."""
    try:
        booking = curd.update_booking_status(db, booking_id, update.status)
        
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        print(f"✅ Booking {booking_id} status updated to {update.status}")
        
        return {
            "message": "Booking updated successfully",
            "booking_id": booking.booking_id,
            "status": booking.status
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating booking: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update booking: {str(e)}")

@app.delete("/bookings/{booking_id}")
def delete_booking_endpoint(booking_id: str, db: Session = Depends(get_db)):
    """Cancel/delete a booking."""
    try:
        success = curd.delete_booking(db, booking_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        print(f"✅ Booking {booking_id} cancelled")
        
        return {
            "message": "Booking cancelled successfully",
            "booking_id": booking_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error cancelling booking: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel booking: {str(e)}")

@app.get("/bookings/count/{user_email}")
def get_bookings_count(user_email: str, db: Session = Depends(get_db)):
    """Get count of bookings for a user."""
    try:
        bookings = curd.get_user_bookings(db, user_email)
        
        return {
            "total": len(bookings),
            "pending": len([b for b in bookings if b.status == "pending"]),
            "confirmed": len([b for b in bookings if b.status == "confirmed"]),
            "completed": len([b for b in bookings if b.status == "completed"]),
            "cancelled": len([b for b in bookings if b.status == "cancelled"])
        }
    
    except Exception as e:
        print(f"❌ Error fetching booking count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch booking count: {str(e)}")

from sqlalchemy import desc

@app.get("/expert/{expert_id}/bookings")
def get_expert_bookings(
    expert_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all bookings for a specific expert."""
    try:
        expert = db.query(Expert).filter(Expert.id == expert_id).first()
        if not expert:
            raise HTTPException(status_code=404, detail="Expert not found")
        
        query = db.query(Booking).filter(Booking.expert_id == expert_id)
        
        if status and status.lower() != 'all':
            query = query.filter(Booking.status == status.lower())
        
        bookings = query.order_by(desc(Booking.created_at)).all()
        
        return {
            "bookings": [
                {
                    "booking_id": booking.booking_id,
                    "user_id": booking.user_id,
                    "expert_id": booking.expert_id,
                    "expert_name": booking.expert_name,
                    "expert_title": booking.expert_title,
                    "specialization": booking.specialization,
                    "price": booking.price,
                    "booking_date": booking.booking_date.isoformat(),
                    "status": booking.status,
                    "created_at": booking.created_at.isoformat()
                }
                for booking in bookings
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for Cloud Run."""
    db_healthy, db_message = check_db_health()
    
    health_status = {
        "status": "healthy" if db_healthy else "unhealthy",
        "model_loaded": MODEL is not None,
        "database": {
            "status": "connected" if db_healthy else "disconnected",
            "message": db_message
        },
        "available_diseases": len(DISEASE_CLASSES),
        "environment": "production" if os.getenv("ENVIRONMENT") == "production" else "development",
        "timestamp": datetime.now().isoformat()
    }
    
    status_code = 200 if db_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/test-db")
def test_database(db: Session = Depends(get_db)):
    """Test database connection (use this to verify Neon works)."""
    try:
        # Test query
        result = db.execute("SELECT 'Neon PostgreSQL Connected!' as message, NOW() as time")
        row = result.fetchone()
        
        return {
            "status": "success",
            "message": row[0],
            "server_time": str(row[1]),
            "database": "Neon PostgreSQL"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

@app.on_event("startup")
def startup_event():
    init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)