from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# For SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./krishisetu.db"

# For PostgreSQL (later)
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/krishisetu"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # Only for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

print("✅ Farmers table created successfully!")

from DataBase.database import engine, Base
from DataBase.models import Expert  # Import your new model

# Create the experts table
Base.metadata.create_all(bind=engine)

print("✅ Expert table created successfully!")

