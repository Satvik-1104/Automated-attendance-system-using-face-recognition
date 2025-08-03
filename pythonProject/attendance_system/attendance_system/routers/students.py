import uuid
import os
import json
import random
from datetime import datetime, timezone, timedelta, date
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional
from ..database import get_db
from ..models import Student, OTPVerification, AttendanceReport, AttendanceCorrection
from ..schemas import StudentRegister, OTPVerifyRequest, StudentOTPRequest
from ..utils.auth import get_current_student, hash_password, send_email
from ..services.section_assignment import assign_sections

router = APIRouter(tags=["students"])

# Configuration
UPLOAD_DIR = "uploads/students/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_FILE_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
OTP_EXPIRY_MINUTES = float(os.getenv("OTP_EXPIRY_MINUTES", 5))


# OTP Request Endpoint
@router.post("/register/request_otp")
def request_otp(
        student: StudentOTPRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
):
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] Received OTP Request:")
    print(f"  Endpoint: /students/register/request_otp")
    print(f"  Request Data: roll_number={student.roll_number}, email={student.email}")

    # valid_entry = db.query(ValidStudent).filter(
    #     ValidStudent.roll_number == student.roll_number,
    #     ValidStudent.email == student.email,
    #     ValidStudent.is_verified,
    # ).first()
    # if not valid_entry:
    #     print(f"[{timestamp}] Validation Failed: No valid student entry found")
    #     raise HTTPException(status_code=400, detail="Invalid roll number or email")

    existing_student = db.query(Student).filter(
        Student.roll_number == student.roll_number
    ).first()
    if existing_student:
        print(f"[{timestamp}] Registration Failed: Student already exists")
        raise HTTPException(status_code=400, detail="Student already registered")

    existing_otp = db.query(OTPVerification).filter(
        OTPVerification.email == student.email,
        OTPVerification.user_type == "student",
    ).first()
    if existing_otp:
        db.delete(existing_otp)
        db.commit()

    otp_code = random.randint(100000, 999999)
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)
    otp_entry = OTPVerification(
        email=student.email,
        otp_code=str(otp_code),
        expiry_time=expiry_time,
        user_type="student",
    )
    db.add(otp_entry)
    db.commit()

    background_tasks.add_task(
        send_email, student.email, f"Your OTP for student registration is {otp_code}"
    )
    print(f"[{timestamp}] OTP Request Successful: OTP sent to {student.email}")
    return {"message": "OTP sent to registered email"}


# OTP Verification Endpoint
@router.post("/register/verify_otp")
def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    otp_entry = db.query(OTPVerification).filter(
        OTPVerification.email == request.email,
        OTPVerification.otp_code == request.otp.strip(),
        OTPVerification.expiry_time > datetime.now(timezone.utc),
        OTPVerification.user_type == "student",
    ).order_by(OTPVerification.expiry_time.desc()).first()

    if not otp_entry:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    otp_entry.is_verified = True
    db.commit()
    return {"message": "OTP verified successfully"}


# Updated Registration Endpoint
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_student(
        roll_number: str = Form(...),
        full_name: str = Form(...),
        email: str = Form(...),
        branch: str = Form(...),
        semester: int = Form(...),
        batch: int = Form(...),
        username: str = Form(...),
        password: str = Form(...),
        sections_selected: str = Form(...),
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db),
        background_tasks: BackgroundTasks = BackgroundTasks(),
):
    try:
        # Validate student credentials
        # valid_entry = db.query(ValidStudent).filter(
        #     ValidStudent.roll_number == roll_number,
        #     ValidStudent.email == email,
        # ).first()
        # if not valid_entry:
        #     raise HTTPException(status_code=400, detail="Invalid roll number or email")

        # Check if student already exists
        existing_student = db.query(Student).filter(Student.roll_number == roll_number).first()
        if existing_student:
            raise HTTPException(status_code=400, detail="Student already exists")

        # Verify OTP
        verified_otp = db.query(OTPVerification).filter(
            OTPVerification.email == email,
            OTPVerification.user_type == "student",
            OTPVerification.is_verified,
        ).first()
        if not verified_otp:
            raise HTTPException(status_code=400, detail="OTP verification required")

        # Parse sections_selected
        try:
            sections_selected_list = json.loads(sections_selected)
            if not isinstance(sections_selected_list, list):
                raise ValueError("sections_selected must be a list")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid sections_selected format")

        # Hash password
        hashed_password = hash_password(password)

        # Create new student
        new_student = Student(
            roll_number=roll_number,
            full_name=full_name,
            email=email,
            branch=branch,
            semester=semester,
            batch=batch,
            username=username,
            password=hashed_password,
        )
        db.add(new_student)
        db.flush()

        # Handle file uploads and set photo_folder
        roll_folder = os.path.join(UPLOAD_DIR, roll_number)
        os.makedirs(roll_folder, exist_ok=True)

        for file in files:
            if file.content_type not in ALLOWED_FILE_TYPES:
                raise HTTPException(
                    status_code=400, detail=f"Invalid file type {file.content_type}"
                )
            if file.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} exceeds 5MB limit"
                )

            unique_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{unique_id}{os.path.splitext(file.filename)[1]}"
            file_path = os.path.join(roll_folder, filename)

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

        # Set the photo_folder for the new student
        new_student.photo_folder = roll_folder

        # Assign sections
        assigned = assign_sections(
            db=db,
            roll_number=new_student.roll_number,
            batch=new_student.batch,
            branch=new_student.branch,
            selected=sections_selected_list,
        )

        # Delete OTP entry
        db.delete(verified_otp)
        db.commit()

        return {
            "message": "Student registered successfully",
            "assigned_sections": assigned,
        }

    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        db.rollback()
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Commented out /upload_photos endpoint
'''
@router.post("/upload_photos")
async def upload_photos(
    files: List[UploadFile] = File(...),
    current_student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    roll_folder = os.path.join(UPLOAD_DIR, current_student.roll_number)
    os.makedirs(roll_folder, exist_ok=True)

    saved_files = []
    for file in files:
        if file.content_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(400, f"Invalid file type {file.content_type}")
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(400, f"File {file.filename} exceeds 5MB limit")

        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{unique_id}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(roll_folder, filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        saved_files.append(file_path)

    if not current_student.photo_folder:
        current_student.photo_folder = roll_folder
        db.commit()

    return {"message": f"Saved {len(saved_files)} photos", "folder": roll_folder}
'''


# Additional Endpoints (unchanged for brevity)
@router.get("/photos/{filename}")
async def get_photo(filename: str, current_student: Student = Depends(get_current_student)):
    if '/' in filename or '\\' in filename:
        raise HTTPException(400, "Invalid filename")
    photo_path = os.path.join(current_student.photo_folder, filename)
    if not os.path.exists(photo_path):
        raise HTTPException(404, "Photo not found")
    return FileResponse(photo_path)


@router.get("/my_attendance")
def my_attendance(current_student: Student = Depends(get_current_student), db: Session = Depends(get_db)):
    attendance_records = db.query(AttendanceReport).filter(
        AttendanceReport.roll_number == int(current_student.roll_number)
    ).all()
    return {
        "attendance_records": [
            {
                "report_id": record.report_id,
                "section_id": record.section_id,
                "course_code": record.course_code,
                "class_time": record.class_time,
                "is_absent": record.is_absent
            }
            for record in attendance_records
        ]
    }


@router.post("/request_correction")
def request_correction(
        report_id: int = Form(...),
        course_code: str = Form(...),
        reason: str = Form(...),
        supporting_image: Optional[UploadFile] = File(None),
        db: Session = Depends(get_db),
        current_student: Student = Depends(get_current_student)
):
    # Verify that the attendance record exists and belongs to the current student
    original_report = db.query(AttendanceReport).filter(
        and_(
            AttendanceReport.report_id == report_id,
            AttendanceReport.roll_number == current_student.roll_number,
            AttendanceReport.course_code == course_code
        )
    ).first()

    if not original_report:
        raise HTTPException(
            status_code=404,
            detail="Attendance record not found or not owned by student"
        )

    # Create a new correction request
    correction = AttendanceCorrection(
        roll_number=current_student.roll_number,
        report_id=report_id,
        section_id=original_report.section_id,
        course_code=course_code,
        reason=reason,
        request_time=datetime.now(timezone.utc)
    )

    # If a supporting image is provided, save the file and store the relative path
    if supporting_image:
        # Create a directory for correction uploads within the "uploads" folder
        upload_dir = os.path.join("uploads", "corrections")
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file; you may want to generate a unique filename here
        file_location = os.path.join(upload_dir, supporting_image.filename)
        with open(file_location, "wb") as file_object:
            file_object.write(supporting_image.file.read())
        # Store the relative path (omit the "uploads" prefix, but include a leading slash)
        relative_path = f"/corrections/{supporting_image.filename}"
        correction.supporting_image = relative_path

    db.add(correction)
    db.commit()

    return {"message": "Correction request submitted"}


@router.get("/check_attendance_status")
def check_attendance_status(current_student: Student = Depends(get_current_student), db: Session = Depends(get_db)):
    today = date.today()
    attendance_records = db.query(AttendanceReport).filter(
        AttendanceReport.roll_number == int(current_student.roll_number),
        AttendanceReport.class_time >= datetime(today.year, today.month, today.day)
    ).all()
    return {
        "attendance_records": [
            {
                "report_id": record.report_id,
                "section_id": record.section_id,
                "course_code": record.course_code,
                "class_time": record.class_time,
                "is_absent": record.is_absent
            }
            for record in attendance_records
        ]
    }


@router.get("/attendance/summary")
def get_attendance_summary(
        current_student: Student = Depends(get_current_student),
        db: Session = Depends(get_db)
):
    # Query all attendance records for the current student
    records = db.query(AttendanceReport).filter(
        AttendanceReport.roll_number == current_student.roll_number  # type: ignore
    ).all()

    total_absent = sum(1 for r in records if r.is_absent)
    total_attended = sum(1 for r in records if not r.is_absent)

    return {"total_absent": total_absent, "total_attended": total_attended}


@router.get("/corrections/pending")
def get_pending_corrections(
        current_student: Student = Depends(get_current_student),
        db: Session = Depends(get_db)
):
    pending_count = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.roll_number == current_student.roll_number,  # type: ignore
        AttendanceCorrection.correction_status == "Pending"
    ).count()

    return {"pending_count": pending_count}


@router.get("/corrections")
def get_corrections(
        current_student: Student = Depends(get_current_student),
        db: Session = Depends(get_db)
):
    corrections = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.roll_number == current_student.roll_number  # type: ignore
    ).all()

    # Build a list of correction dictionaries with the desired keys
    corrections_list = [
        {
            "correction_id": c.correction_id,
            "report_id": c.report_id,
            "course_code": c.course_code,
            "request_time": c.request_time,
            "status": c.correction_status,
            "feedback": c.feedback or ""
        }
        for c in corrections
    ]

    return {"corrections": corrections_list}
