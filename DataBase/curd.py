from sqlalchemy.orm import Session
from DataBase.models import User
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