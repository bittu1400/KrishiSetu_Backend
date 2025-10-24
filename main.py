from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
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

# TensorFlow and Keras imports
try:
    from tf_keras.models import load_model
    from tf_keras.preprocessing import image as keras_image
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: TensorFlow not installed. Using mock detection.")

# Database and RAG imports
from DataBase.database import SessionLocal, engine, Base
from DataBase.models import User, Expert
from DataBase import curd  # Note: Ensure 'crud' is correctly spelled (was 'curd' in original)
from RAG.chatbot import get_chatbot_response

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment!")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not set in environment!")

if not TWILIO_ACCOUNT_SID:
    raise RuntimeError("TWILIO_ACCOUNT_SID not set in environment!")

if not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_AUTH_TOKEN not set in environment!")

# FastAPI app setup
app = FastAPI(title="KrishiSetu RAG Backend", version="1.0")

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
    phone: str | None = None
    email: EmailStr | None = None

class VerifyRequest(BaseModel):
    identifier: str
    otp: str

class RegisterRequest(BaseModel):
    name: str
    city: str
    email: EmailStr
    phone: str
    pin: str

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
    sender = "your@gmail.com"  # Replace with your email
    password = "pfgxxybyewaglrne"  # Replace with your app-specific password
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

# Note: SMS sending is commented out as per original code
def send_sms(to_phone: str, otp: str):
    from twilio.rest import Client
    account_sid = TWILIO_ACCOUNT_SID
    auth_token = TWILIO_AUTH_TOKEN
    client = Client(account_sid, auth_token)
    try:
        message = client.messages.create(
            body=f"Your OTP is {otp}",
            from_="+12792394257",
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
MODEL_PATH = "Trained_model.keras"
DISEASE_CLASSES = [
    'Apple___Black_rot', 'Apple___healthy', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Grape___Esca', 'Grape___Leaf_blight', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

DISEASE_RECOMMENDATIONS = {
    "Early blight": {
        "recommendation": "Remove affected leaves and apply Mancozeb fungicide.",
        "treatment": "Spray Dithane M-45 once every 7 days."
    },
    "Late blight": {
        "recommendation": "Avoid overhead irrigation and use Copper-based fungicides.",
        "treatment": "Spray Ridomil Gold every 5 days."
    },
    "Healthy": {
        "recommendation": "No disease detected. Maintain regular irrigation and balanced nutrition.",
        "treatment": "None required."
    },
    # Add more recommendations as needed
}

def load_crop_model(model_path):
    """Load pre-trained crop disease detection model."""
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

MODEL = load_crop_model(MODEL_PATH)

def preprocess_image(image_bytes):
    """Convert raw image bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  # Adjust size based on model requirements
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Note: Normalization commented out; adjust based on model training
    # img_array /= 255.0
    print(f"📊 Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

def predict_disease(preprocessed_image):
    """Run model prediction and return structured output."""
    try:
        if MODEL_AVAILABLE and MODEL is not None:
            preds = MODEL.predict(preprocessed_image)
            class_index = np.argmax(preds[0])
            confidence = float(np.max(preds[0]) * 100)
        else:
            # Mock prediction for testing
            class_index = random.randint(0, len(DISEASE_CLASSES) - 1)
            confidence = random.uniform(70, 99)
        
        disease = DISEASE_CLASSES[class_index] if class_index < len(DISEASE_CLASSES) else "Unknown"
        disease_name = disease.split("___")[-1].replace("_", " ")
        
        return {
            "class_index": int(class_index),
            "disease": disease_name,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

# Endpoints
@app.post("/send-otp")
def send_otp(request: OTPRequest):
    """Send OTP to phone (SMS, commented out) or email."""
    otp = generate_otp()
    
    if request.phone:
        # send_sms(request.phone, otp)  # Uncomment when using SMS
        print(f"📱 DEBUG: OTP {otp} for phone {request.phone}")
        otp_store[request.phone] = otp
        return {"message": "OTP sent via SMS (debug mode)", "otp": otp}  # Remove 'otp' in production
    
    elif request.email:
        send_email(request.email, otp)
        otp_store[request.email] = otp
        print(f"📧 Email sent to {request.email}")
        return {"message": "OTP sent via Email"}
    
    else:
        raise HTTPException(status_code=400, detail="Phone or Email required")

@app.post("/verify-otp")
def verify_otp(request: VerifyRequest):
    """Verify the OTP."""
    stored_otp = otp_store.get(request.identifier)
    
    print(f"🔍 Verifying OTP for: {request.identifier}")
    print(f"📝 Stored OTP: {stored_otp}, Provided OTP: {request.otp}")
    
    if not stored_otp:
        raise HTTPException(status_code=400, detail="No OTP found or expired")
    
    if stored_otp != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    del otp_store[request.identifier]
    print(f"✅ OTP verified successfully for {request.identifier}")
    return {"message": "OTP verified successfully"}

@app.post("/register")
def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user in the database."""
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
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
            password=data.pin
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
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Update the login endpoint
@app.post("/login")
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    """Login user with email/phone and password/PIN."""
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
        raise HTTPException(status_code=401, detail="User not found")
    
    # Verify password/PIN using the verify_password function from curd.py
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
    
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]:
            print(f"❌ Invalid content type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload JPG or PNG image."
            )
        
        # Read file bytes
        image_data = await file.read()
        print(f"✅ Read {len(image_data)} bytes")
        
        if not image_data:
            print("❌ Empty file!")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Preprocess
        print("🔄 Preprocessing image...")
        preprocessed_image = preprocess_image(image_data)
        
        # Predict
        print("🤖 Running prediction...")
        prediction = predict_disease(preprocessed_image)
        print(f"✅ Prediction: {prediction}")
        
        disease_name = prediction["disease"]
        confidence = prediction["confidence"]
        
        # Get recommendations
        disease_info = DISEASE_RECOMMENDATIONS.get(
            disease_name,
            {
                "recommendation": "Unable to determine specific recommendations.",
                "treatment": "Please consult an agricultural expert."
            }
        )
        
        # Build response
        response = {
            "disease_detected": disease_name,
            "confidence_score": confidence,
            "is_healthy": disease_name.lower() == "healthy",
            "severity": "Low" if confidence < 75 else "Medium" if confidence < 85 else "High",
            "recommendation": disease_info["recommendation"],
            "treatment": disease_info["treatment"],
            "diagnosis_date": datetime.now().isoformat(),
            "additional_info": {
                "class_index": prediction["class_index"],
                "total_classes": len(DISEASE_CLASSES),
                "model_version": "1.0"
            }
        }
        
        print("✅ Response prepared successfully")
        print("="*50 + "\n")
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException as e:
        print(f"❌ HTTP Exception: {e.detail}")
        print("="*50 + "\n")
        return JSONResponse(
            content={"error": e.detail, "status": "error"},
            status_code=e.status_code
        )
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return JSONResponse(
            content={"error": f"Error processing image: {str(e)}", "status": "error"},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "available_diseases": DISEASE_CLASSES
    }

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
    email: EmailStr
    phone: str
    specialization: str
    price: str
    features: str
    password: str
    bio: Optional[str] = ""
    category: Optional[str] = "All"

class ExpertLoginRequest(BaseModel):
    email: EmailStr
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
    expert = curd.verify_expert(db, request.email, request.password)
    
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
