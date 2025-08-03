import smtplib

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from ..models import Student, Faculty, PhDStudent
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends
from typing import Union
from ..database import get_db
from fastapi.security import OAuth2PasswordBearer
from email.message import EmailMessage

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

if not SECRET_KEY or not ALGORITHM:
    raise ValueError("SECRET_KEY and ALGORITHM must be set in the environment variables.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Hash Password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Verify Password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# Create JWT Token
def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Verify JWT Token and get user
def verify_access_token(token: str, db: Session = Depends(get_db)) -> Union[Student, Faculty, PhDStudent]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None or role is None:
            raise credentials_exception

        # Find user by role
        user = None
        if role == "student":
            user = db.query(Student).filter(Student.username == username).first()
        elif role == "faculty":
            user = db.query(Faculty).filter(Faculty.username == username).first()
        elif role == "phd_student":
            user = db.query(PhDStudent).filter(PhDStudent.username == username).first()

        if user is None:
            raise credentials_exception  # If user is not found

    except JWTError:
        raise credentials_exception

    return user


# Get Current Student
def get_current_student(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Student:
    user = verify_access_token(token, db)
    if not isinstance(user, Student):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access this endpoint",
        )
    return user


# Get Current Faculty
def get_current_faculty(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Faculty:
    user = verify_access_token(token, db)
    if not isinstance(user, Faculty):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only faculty can access this endpoint",
        )
    return user


# Get Current PhD Student
def get_current_phd_student(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> PhDStudent:
    user = verify_access_token(token, db)
    if not isinstance(user, PhDStudent):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only PhD students can access this endpoint",
        )
    return user


def send_email(receiver_email: str, message: str):
    msg = EmailMessage()
    msg.set_content(message)
    msg["Subject"] = "Your OTP for Registration"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")
