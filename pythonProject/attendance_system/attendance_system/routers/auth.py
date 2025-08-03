from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from ..utils.auth import create_access_token, verify_password, send_email, hash_password
from ..models import Student, Faculty, PhDStudent, OTPVerification
from ..database import get_db
from ..schemas import TokenResponse, ResetPasswordRequest
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta, timezone
import random
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Load OTP expiry from .env, default to 5 minutes
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", 5))


@router.post("/token", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Check student
    student = db.query(Student).filter(Student.username == form_data.username).first()
    if student and verify_password(form_data.password, student.password):
        access_token = create_access_token(data={"sub": student.username, "role": "student"})
        return {"access_token": access_token, "token_type": "bearer"}

    # Check faculty
    faculty = db.query(Faculty).filter(Faculty.username == form_data.username).first()
    if faculty and verify_password(form_data.password, faculty.password):
        access_token = create_access_token(data={"sub": faculty.username, "role": "faculty"})
        return {"access_token": access_token, "token_type": "bearer"}

    # Check PhD student
    phd_student = db.query(PhDStudent).filter(PhDStudent.username == form_data.username).first()
    if phd_student and verify_password(form_data.password, phd_student.password):
        access_token = create_access_token(data={"sub": phd_student.username, "role": "phd_student"})
        return {"access_token": access_token, "token_type": "bearer"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/forgot-password")
def forgot_password(email: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Request a password reset OTP for the given email.
    Sends an OTP to the user's email if it exists in the system.
    """
    # Find the user by email across all user types
    user_type = None
    user_id = None

    # Check students
    student = db.query(Student).filter(Student.email == email).first()
    if student:
        user_type = "student"
        user_id = student.roll_number
    else:
        # Check faculty
        faculty = db.query(Faculty).filter(Faculty.email == email).first()
        if faculty:
            user_type = "faculty"
            user_id = faculty.employee_id
        else:
            # Check PhD students
            phd_student = db.query(PhDStudent).filter(PhDStudent.email == email).first()
            if phd_student:
                user_type = "phd"
                user_id = phd_student.employee_id

    # If no user is found, return a generic message for security
    if user_type is None or user_id is None:
        return {"message": "If the email exists, an OTP will be sent."}

    # Generate a 6-digit OTP
    otp_code = str(random.randint(100000, 999999))
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)

    # Delete any existing password reset OTPs for this email and user type
    db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.user_type == user_type,
        OTPVerification.purpose == "password_reset"
    ).delete(synchronize_session=False)
    db.commit()

    # Store the new OTP in the otp_verifications table
    otp_entry = OTPVerification(
        email=email,
        otp_code=otp_code,
        expiry_time=expiry_time,
        user_type=user_type,
        purpose="password_reset",
        is_verified=False,
        student_id=user_id if user_type == "student" else None,
        faculty_id=user_id if user_type == "faculty" else None,
        phd_id=user_id if user_type == "phd" else None
    )
    db.add(otp_entry)
    db.commit()

    # Send the OTP via email in the background
    email_message = (
        f"Your password reset OTP is: {otp_code}\n"
        f"This OTP will expire in {OTP_EXPIRY_MINUTES} minutes."
    )
    background_tasks.add_task(send_email, email, email_message)

    return {"message": "If the email exists, an OTP will be sent."}


@router.post("/reset-password")
def reset_password(
        request: ResetPasswordRequest,
        db: Session = Depends(get_db)
):
    """
    Verify the OTP and reset the user's password.
    """
    # Find the OTP entry
    otp_entry = db.query(OTPVerification).filter(
        OTPVerification.email == request.email,
        OTPVerification.otp_code == request.otp,
        OTPVerification.purpose == "password_reset",
        OTPVerification.is_verified.is_(False)
    ).first()

    if not otp_entry:
        print(f"OTP not found for email: {request.email}, OTP: {request.otp}")
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    current_time = datetime.now(timezone.utc)
    if otp_entry.expiry_time < current_time:
        raise HTTPException(status_code=400, detail="OTP has expired")

    # Determine user type and find the user
    user = None
    if otp_entry.user_type == "student":
        user = db.query(Student).filter(Student.roll_number == otp_entry.student_id).first()  # type: ignore
    elif otp_entry.user_type == "faculty":
        user = db.query(Faculty).filter(Faculty.employee_id == otp_entry.faculty_id).first()  # type: ignore
        print(f"Looking for faculty with employee_id: {otp_entry.faculty_id}, Found: {user}")
    elif otp_entry.user_type == "phd":
        user = db.query(PhDStudent).filter(PhDStudent.employee_id == otp_entry.phd_id).first()  # type: ignore
        print(f"Looking for phd with employee_id: {otp_entry.phd_id}, Found: {user}")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update the password
    user.password = hash_password(request.new_password)

    db.delete(otp_entry)
    db.commit()

    return {"message": "Password reset successfully"}
