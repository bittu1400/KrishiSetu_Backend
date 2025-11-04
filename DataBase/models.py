from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from DataBase.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True)
    password = Column(String, nullable=False)  # Hashed password
    
    # Relationship
    bookings = relationship("Booking", back_populates="user")


class Expert(Base):
    __tablename__ = "experts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    title = Column(String, nullable=False)  # e.g., "Plant Pathologist"
    experience = Column(String, nullable=False)  # e.g., "12 years experience"
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=False)
    specialization = Column(String, nullable=False)
    price = Column(String, nullable=False)  # e.g., "NPR 500/consultation"
    features = Column(String, nullable=False)  # e.g., "Video call available"
    rating = Column(Float, default=0.0)
    reviews = Column(Integer, default=0)
    status = Column(String, default="Available")  # "Available", "Busy", "Offline"
    password = Column(String, nullable=False)  # Hashed password
    bio = Column(String, default="")  # Optional bio
    category = Column(String, default="All")  # "Crop Disease", "Nutrition", "Organic", etc.
    
    # Relationship
    bookings = relationship("Booking", back_populates="expert")


class Booking(Base):
    """Booking model for storing consultation bookings."""
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expert_id = Column(Integer, ForeignKey("experts.id"), nullable=False)
    expert_name = Column(String, nullable=False)
    expert_title = Column(String, nullable=False)
    specialization = Column(String)
    price = Column(String)
    booking_date = Column(DateTime, nullable=False)
    status = Column(String, default="pending")  # pending, confirmed, completed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="bookings")
    expert = relationship("Expert", back_populates="bookings")