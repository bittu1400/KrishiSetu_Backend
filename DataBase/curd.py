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
