from sqlalchemy.orm import Session
from DataBase.models import User, Expert
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password using Argon2."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def create_user(db: Session, name: str, city: str, email: str, phone: str, password: str) -> User:
    """Create a new user in the database."""
    hashed_pw = hash_password(password)
    db_user = User(name=name, city=city, email=email, phone=phone, password=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_user(db: Session, email: str, password: str) -> User | bool:
    """Verify user credentials and return the user object or False."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user

def create_expert(db: Session, name: str, title: str, experience: str, email: str, 
                  phone: str, specialization: str, price: str, features: str, 
                  password: str, bio: str = "", category: str = "All") -> Expert:
    hashed_pw = hash_password(password)
    db_expert = Expert(
        name=name, title=title, experience=experience, email=email,
        phone=phone, specialization=specialization, price=price,
        features=features, password=hashed_pw, bio=bio, category=category
    )
    db.add(db_expert)
    db.commit()
    db.refresh(db_expert)
    return db_expert

def verify_expert(db: Session, email: str, password: str):
    expert = db.query(Expert).filter(Expert.email == email).first()
    if not expert:
        return False
    if not verify_password(password, expert.password):
        return False
    return expert

def get_all_experts(db: Session, category: str = None):
    query = db.query(Expert)
    if category and category != "All":
        query = query.filter(Expert.category == category)
    return query.all()

def update_expert_status(db: Session, expert_id: int, status: str):
    expert = db.query(Expert).filter(Expert.id == expert_id).first()
    if expert:
        expert.status = status
        db.commit()
        db.refresh(expert)
    return expert


# ============================================
# Add these functions to DataBase/curd.py
# ============================================

from sqlalchemy.orm import Session
from sqlalchemy import desc
from .models import Booking, User, Expert
from datetime import datetime
import uuid

def create_booking(
    db: Session,
    user_email: str,
    expert_id: int,
    expert_name: str,
    expert_title: str,
    specialization: str,
    price: str,
    booking_date: datetime,
    status: str = "pending"
):
    """Create a new booking."""
    # Get user by email
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return None
    
    # Verify expert exists
    expert = db.query(Expert).filter(Expert.id == expert_id).first()
    if not expert:
        return None
    
    # Generate unique booking ID
    booking_id = str(uuid.uuid4())[:8].upper()
    
    # Create booking
    new_booking = Booking(
        booking_id=booking_id,
        user_id=user.id,
        expert_id=expert_id,
        expert_name=expert_name,
        expert_title=expert_title,
        specialization=specialization,
        price=price,
        booking_date=booking_date,
        status=status,
        created_at=datetime.utcnow()
    )
    
    db.add(new_booking)
    db.commit()
    db.refresh(new_booking)
    
    print(f"✅ Created booking {booking_id} for user {user.name}")
    return new_booking

def get_user_bookings(db: Session, user_email: str, status: str = None):
    """Get all bookings for a user, optionally filtered by status."""
    # Get user
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return []
    
    # Build query
    query = db.query(Booking).filter(Booking.user_id == user.id)
    
    # Apply status filter if provided
    if status:
        query = query.filter(Booking.status == status)
    
    # Order by created_at descending (newest first)
    bookings = query.order_by(desc(Booking.created_at)).all()
    
    return bookings

def get_expert_bookings(db: Session, expert_id: int, status: str = None):
    """Get all bookings for an expert, optionally filtered by status."""
    query = db.query(Booking).filter(Booking.expert_id == expert_id)
    
    if status:
        query = query.filter(Booking.status == status)
    
    return query.order_by(desc(Booking.booking_date)).all()

def get_booking_by_id(db: Session, booking_id: str):
    """Get a single booking by booking_id."""
    return db.query(Booking).filter(Booking.booking_id == booking_id).first()

def update_booking_status(db: Session, booking_id: str, status: str):
    """Update the status of a booking."""
    booking = db.query(Booking).filter(Booking.booking_id == booking_id).first()
    
    if booking:
        booking.status = status
        db.commit()
        db.refresh(booking)
        print(f"✅ Updated booking {booking_id} status to {status}")
    
    return booking

def delete_booking(db: Session, booking_id: str):
    """Delete a booking (used for cancellation)."""
    booking = db.query(Booking).filter(Booking.booking_id == booking_id).first()
    
    if booking:
        db.delete(booking)
        db.commit()
        print(f"✅ Deleted booking {booking_id}")
        return True
    
    return False

def get_bookings_count(db: Session, user_email: str):
    """Get count of bookings by status for a user."""
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {
            "total": 0,
            "pending": 0,
            "confirmed": 0,
            "completed": 0,
            "cancelled": 0
        }
    
    bookings = db.query(Booking).filter(Booking.user_id == user.id).all()
    
    return {
        "total": len(bookings),
        "pending": len([b for b in bookings if b.status == "pending"]),
        "confirmed": len([b for b in bookings if b.status == "confirmed"]),
        "completed": len([b for b in bookings if b.status == "completed"]),
        "cancelled": len([b for b in bookings if b.status == "cancelled"])
    }
