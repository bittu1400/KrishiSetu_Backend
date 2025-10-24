from sqlalchemy import Column, Integer, String, Float, Boolean
from DataBase.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True)
    password = Column(String, nullable=False)  # Hashed password

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
