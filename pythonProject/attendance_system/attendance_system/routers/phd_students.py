from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta, timezone, date, time
import random
import os
# from ..tasks import delete_expired_otps
from ..utils.auth import (
    send_email,
    hash_password,
    get_current_phd_student
)
from ..database import get_db
from ..models import (
    OTPVerification,
    PhDStudent,
    AttendanceReport,
    AttendanceCorrection,
    DynamicScheduleUpdate,
    ValidPhDStudent,
    ClassSchedule,
    ClassScheduleAccess,
    Section,
    Student,
    StudentSection
)
from ..schemas import (
    OTPVerifyRequest,
    PhDStudentRegister,
    MarkAttendanceRequest,
    ApproveCorrectionRequest,
    BulkApproveCorrectionRequest,
    CancelClassRequest,
    AttendanceReportResponse,
    AttendanceRecord,
    UnifiedScheduleUpdateRequest,
    PhDOTPRequest,
    GenerateReportResponse
)

router = APIRouter(tags=["phd_students"])

OTP_EXPIRY_MINUTES = float(os.getenv("OTP_EXPIRY_MINUTES", 5))


def phd_has_access(phd_id: int, section_id: int, course_code: str, db: Session):
    return db.query(ClassScheduleAccess).filter(
        and_(
            ClassScheduleAccess.phd_student_id == phd_id,
            ClassSchedule.section_id == section_id,
            ClassSchedule.course_code == course_code
        )
    ).join(ClassSchedule).first() is not None


@router.post("/register/request_otp")
def request_otp(
    phd: PhDOTPRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    valid_entry = db.query(ValidPhDStudent).filter(
        (ValidPhDStudent.employee_id == phd.phd_id) &
        (ValidPhDStudent.email == phd.email) &
        ValidPhDStudent.is_verified
    ).first()

    if not valid_entry:
        raise HTTPException(status_code=400, detail="Invalid PhD student ID or email")

    existing_phd = db.query(PhDStudent).filter(
        (PhDStudent.employee_id == phd.phd_id) |
        (PhDStudent.email == phd.email)
    ).first()

    if existing_phd:
        raise HTTPException(status_code=400, detail="PhD student already registered")

    existing_otp = db.query(OTPVerification).filter(
        OTPVerification.email == phd.email,
        OTPVerification.user_type == "phd"
    ).first()

    if existing_otp:
        if existing_otp.expiry_time > datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="OTP already sent. Try again later.")
        db.delete(existing_otp)
        db.commit()

    otp_code = random.randint(100000, 999999)
    expiry_time = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)

    otp_entry = OTPVerification(
        email=phd.email,
        otp_code=str(otp_code),
        expiry_time=expiry_time,
        user_type="phd"
    )
    db.add(otp_entry)
    db.commit()

    background_tasks.add_task(send_email, phd.email, f"Your OTP for PhD student registration is {otp_code}")

    return {"message": "OTP sent to registered email"}


@router.post("/register/verify_otp")
def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    current_time = datetime.now(timezone.utc)

    otp_entry = db.query(OTPVerification).filter(
        OTPVerification.email == request.email,
        OTPVerification.otp_code == request.otp,
        OTPVerification.user_type == "phd",
        OTPVerification.expiry_time > current_time
    ).order_by(OTPVerification.expiry_time.desc()).first()

    if not otp_entry:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    otp_entry.is_verified = True
    db.commit()

    return {"message": "OTP verified successfully"}


@router.post("/register")
def register_phd(phd: PhDStudentRegister, db: Session = Depends(get_db)):
    valid_entry = db.query(ValidPhDStudent).filter(
        ValidPhDStudent.employee_id == phd.phd_id,
        ValidPhDStudent.email == phd.email
    ).first()
    if not valid_entry:
        raise HTTPException(status_code=400, detail="Invalid PhD student ID or email")

    existing_phd = db.query(PhDStudent).filter(
        (PhDStudent.employee_id == phd.phd_id) |
        (PhDStudent.email == phd.email) |
        (PhDStudent.username == phd.username)
    ).first()
    if existing_phd:
        raise HTTPException(status_code=400, detail="PhD student already exists")

    verified_otp = db.query(OTPVerification).filter(
        OTPVerification.email == phd.email,
        OTPVerification.user_type == "phd",
        OTPVerification.is_verified
    ).first()
    if not verified_otp:
        raise HTTPException(status_code=400, detail="OTP verification required")

    hashed_password = hash_password(phd.password)
    new_phd = PhDStudent(
        employee_id=phd.phd_id,
        full_name=phd.name,
        email=phd.email,
        username=phd.username,
        password=hashed_password
    )

    db.add(new_phd)
    db.flush()

    verified_otp.phd_id = new_phd.employee_id
    db.delete(verified_otp)
    db.commit()

    for assignment in phd.teaching_assignments:
        batch = assignment.batch
        for course_code in assignment.course_code:
            for section_name in assignment.sections:
                section = db.query(Section).filter(
                    and_(
                        Section.section_name == section_name,
                        Section.course_code == course_code,
                        Section.batch == batch
                    )
                ).first()

                if not section:
                    continue

                schedules = db.query(ClassSchedule).filter(
                    and_(
                        ClassSchedule.section_id == section.section_id,
                        ClassSchedule.course_code == course_code
                    )
                ).all()

                for schedule in schedules:
                    access = ClassScheduleAccess(
                        schedule_id=schedule.schedule_id,
                        phd_student_id=new_phd.employee_id
                    )
                    db.add(access)

    db.commit()

    return {"message": "PhD student registered successfully", "phd_id": new_phd.employee_id}


@router.post("/mark_attendance")
def mark_attendance(
    request: MarkAttendanceRequest,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    if not phd_has_access(current_phd.employee_id, request.section_id, request.course_code, db):
        raise HTTPException(status_code=403, detail="Unauthorized: You are not assigned to this section or course")

    existing_record = db.query(AttendanceReport).filter(
        AttendanceReport.roll_number == request.roll_number,
        AttendanceReport.class_time == request.class_time,
        AttendanceReport.section_id == request.section_id,
        AttendanceReport.course_code == request.course_code
    ).first()

    if existing_record:
        existing_record.is_absent = not request.is_present
        existing_record.marked_by_phd = current_phd.employee_id
    else:
        new_record = AttendanceReport(
            roll_number=request.roll_number,
            class_time=request.class_time,
            is_absent=not request.is_present,
            marked_by_phd=current_phd.employee_id,
            section_id=request.section_id,
            course_code=request.course_code
        )
        db.add(new_record)

    db.commit()
    return {"message": "Attendance marked successfully"}


@router.get("/view_attendance/{section_id}", response_model=List[AttendanceReportResponse])
def view_attendance(
    section_id: int,
    course_code: str,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    if not phd_has_access(current_phd.employee_id, section_id, course_code, db):
        raise HTTPException(status_code=403, detail="Unauthorized")

    records = db.query(AttendanceReport).filter(
        AttendanceReport.section_id == section_id,
        AttendanceReport.course_code == course_code
    ).all()

    grouped = defaultdict(list)
    for record in records:
        grouped[record.roll_number].append(record)

    response = []
    for roll_number, records in grouped.items():
        attendance_records = [
            AttendanceRecord(
                class_time=record.class_time,
                is_absent=record.is_absent,
                course_code=record.course_code
            ) for record in records
        ]
        response.append({
            "section_id": section_id,
            "course_code": course_code,
            "roll_number": roll_number,
            "student_attendance": attendance_records
        })

    return response


@router.get("/pending_corrections", operation_id="get_phd_pending_corrections")
def get_pending_corrections(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    phd_sections = db.query(ClassSchedule.section_id, ClassSchedule.course_code).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()
    phd_section_ids = [section.section_id for section in phd_sections]
    phd_course_codes = [section.course_code for section in phd_sections]

    pending_requests = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.section_id.in_(phd_section_ids),
        AttendanceCorrection.course_code.in_(phd_course_codes),
        AttendanceCorrection.correction_status == "Pending"
    ).all()

    return {"pending_corrections": pending_requests}


@router.post("/approve_correction")
def approve_correction(
    request: ApproveCorrectionRequest,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    correction_request = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.correction_id == request.correction_id
    ).first()

    if not correction_request:
        raise HTTPException(status_code=404, detail="Correction request not found")

    if not phd_has_access(
        current_phd.employee_id,
        correction_request.section_id,
        correction_request.course_code,
        db
    ):
        raise HTTPException(status_code=403, detail="Unauthorized: You cannot approve this correction")

    correction_request.correction_status = "Approved" if request.approval_status else "Rejected"
    correction_request.feedback = request.feedback
    correction_request.processed_by_phd = current_phd.employee_id

    if request.approval_status:
        attendance_record = db.query(AttendanceReport).filter(
            AttendanceReport.report_id == correction_request.report_id  # type: ignore
        ).first()

        if attendance_record:
            attendance_record.is_absent = False
            attendance_record.marked_by_phd = current_phd.employee_id
            attendance_record.correction_applied = True

    db.commit()
    return {"message": "Correction request approved successfully and attendance updated."}


@router.post("/bulk_approve_correction")
def bulk_approve_correction(
    request: BulkApproveCorrectionRequest,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    phd_sections = db.query(ClassSchedule.section_id, ClassSchedule.course_code).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    phd_section_ids = [s.section_id for s in phd_sections]
    phd_course_codes = [s.course_code for s in phd_sections]

    corrections_to_approve = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.correction_id.in_(request.correction_ids),
        AttendanceCorrection.section_id.in_(phd_section_ids),
        AttendanceCorrection.course_code.in_(phd_course_codes),
        AttendanceCorrection.correction_status == "Pending"
    ).all()

    if not corrections_to_approve:
        raise HTTPException(status_code=404, detail="No matching correction requests found.")

    for correction in corrections_to_approve:
        correction.correction_status = "Approved" if request.approval_status else "Rejected"
        correction.feedback = request.feedback
        correction.processed_by_phd = current_phd.employee_id

        if request.approval_status:
            attendance_record = db.query(AttendanceReport).filter(
                AttendanceReport.report_id == correction.report_id  # type: ignore
            ).first()

            if attendance_record:
                attendance_record.is_absent = False
                attendance_record.marked_by_phd = current_phd.employee_id
                attendance_record.correction_applied = True

    db.commit()
    return {"message": "Bulk correction requests processed successfully, and attendance updated."}


@router.post("/update_schedule")
def update_schedule(
    request: UnifiedScheduleUpdateRequest,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
) -> dict:

    original_date = request.original_date
    day_of_week = original_date.strftime("%A")

    base_schedule = db.query(ClassSchedule).filter(
        ClassSchedule.section_id == request.section_id,
        ClassSchedule.course_code == request.course_code,
        ClassSchedule.day_of_week == day_of_week
    ).first()
    if not base_schedule:
        raise HTTPException(status_code=404, detail="Base schedule not found.")

    if not phd_has_access(current_phd.employee_id, request.section_id, request.course_code, db):
        raise HTTPException(status_code=403, detail="Unauthorized: You cannot update this schedule.")

    if request.update_type == "CANCELLED":
        dsu = DynamicScheduleUpdate(
            schedule_id=base_schedule.schedule_id,
            update_type="CANCELLED",
            original_date=request.original_date,
            reason=request.reason,
            updated_by_phd=current_phd.employee_id
        )
        db.add(dsu)
        db.commit()
        return {"message": "Class cancelled successfully."}

    elif request.update_type == "RESCHEDULED":
        if not request.new_time or not request.new_location:
            raise HTTPException(status_code=400, detail="New time and location are required for rescheduling.")

        new_date = request.new_time.date()
        new_start_time = request.new_time.time()
        original_duration = (datetime.combine(date.min, base_schedule.end_time) -
                             datetime.combine(date.min, base_schedule.start_time))
        new_end_time = (datetime.combine(date.min, new_start_time) + original_duration).time()

        conflict_base = db.query(ClassSchedule).filter(
            ClassSchedule.classroom == request.new_location,
            ClassSchedule.day_of_week == new_date.strftime("%A"),
            or_(
                and_(ClassSchedule.start_time <= new_start_time, ClassSchedule.end_time > new_start_time),
                and_(ClassSchedule.start_time < new_end_time, ClassSchedule.end_time >= new_end_time)
            )
        ).first()

        conflict_dsu = db.query(DynamicScheduleUpdate).filter(
            DynamicScheduleUpdate.schedule_id != base_schedule.schedule_id,  # type: ignore
            DynamicScheduleUpdate.new_date == new_date,
            DynamicScheduleUpdate.new_classroom == request.new_location,
            or_(
                and_(DynamicScheduleUpdate.new_start_time <= new_start_time,
                     DynamicScheduleUpdate.new_end_time > new_start_time),
                and_(DynamicScheduleUpdate.new_start_time < new_end_time,
                     DynamicScheduleUpdate.new_end_time >= new_end_time)
            )
        ).first()

        if conflict_base or conflict_dsu:
            raise HTTPException(status_code=400,
                                detail="The new time and location conflict with another scheduled class.")

        dsu = DynamicScheduleUpdate(
            schedule_id=base_schedule.schedule_id,
            update_type="RESCHEDULED",
            original_date=request.original_date,
            new_date=new_date,
            new_start_time=new_start_time,
            new_end_time=new_end_time,
            new_classroom=request.new_location,
            reason=request.reason,
            updated_by_phd=current_phd.employee_id
        )
        db.add(dsu)
        db.commit()
        return {"message": "Schedule rescheduled successfully."}

    else:
        raise HTTPException(status_code=400, detail="Invalid update type. Use 'CANCELLED' or 'RESCHEDULED'.")


@router.delete("/cancel_class")
def cancel_class(
    request: CancelClassRequest,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    if not phd_has_access(current_phd.employee_id, request.section_id, request.course_code, db):
        raise HTTPException(status_code=403, detail="Unauthorized: You cannot cancel this class")

    dsu = db.query(DynamicScheduleUpdate).filter(
        DynamicScheduleUpdate.schedule_id.in_(
            db.query(ClassSchedule.schedule_id).filter(
                ClassSchedule.section_id == request.section_id,
                ClassSchedule.course_code == request.course_code
            )
        ),
        DynamicScheduleUpdate.update_type.in_(["TIME CHANGE", "TIME_LOC CHANGE", "LOCATION CHANGE", "CANCELLED"])
    ).order_by(DynamicScheduleUpdate.update_timestamp.desc()).first()

    if dsu:
        effective_day = dsu.new_day_of_week
        effective_start = dsu.new_start_time
        req_day = request.class_time.strftime("%A")
        req_time = request.class_time.time()
        if effective_day != req_day or effective_start != req_time:
            class_schedule = db.query(ClassSchedule).filter(
                ClassSchedule.section_id == request.section_id,
                ClassSchedule.course_code == request.course_code,
                ClassSchedule.start_time == req_time,
                ClassSchedule.day_of_week == req_day
            ).first()
        else:
            class_schedule = db.query(ClassSchedule).filter(
                ClassSchedule.schedule_id == dsu.schedule_id  # type: ignore
            ).first()
    else:
        class_schedule = db.query(ClassSchedule).filter(
            ClassSchedule.section_id == request.section_id,
            ClassSchedule.course_code == request.course_code,
            ClassSchedule.start_time == request.class_time.time(),
            ClassSchedule.day_of_week == request.class_time.strftime("%A")
        ).first()

    if not class_schedule:
        raise HTTPException(status_code=404, detail="Class not found")

    update = DynamicScheduleUpdate(
        schedule_id=class_schedule.schedule_id,
        updated_by_phd=current_phd.employee_id,
        update_type="CANCELLED",
        reason=request.reason
    )
    db.add(update)
    db.commit()
    return {"message": "Class canceled successfully"}


@router.get("/generate_report/{section_id}", response_model=GenerateReportResponse)
def generate_report(
    section_id: int,
    course_code: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    if not phd_has_access(current_phd.employee_id, section_id, course_code, db):
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: You cannot generate reports for this section or course"
        )

    start_datetime = (
        datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc)
        if start_date else None
    )
    end_datetime = (
        datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc)
        if end_date else None
    )

    query = db.query(AttendanceReport).filter(
        AttendanceReport.section_id == section_id,
        AttendanceReport.course_code == course_code
    )
    if start_datetime:
        query = query.filter(AttendanceReport.class_time >= start_datetime)
    if end_datetime:
        query = query.filter(AttendanceReport.class_time <= end_datetime)
    records = query.all()

    unique_class_times = {record.class_time for record in records}
    total_classes = len(unique_class_times)

    attendance_by_student = defaultdict(list)
    for record in records:
        attendance_by_student[record.roll_number].append(record)

    student_summaries = []
    for roll_number, student_records in attendance_by_student.items():
        present_count = sum(1 for r in student_records if not r.is_absent)
        absent_count = total_classes - present_count
        attendance_percentage = (present_count / total_classes) * 100 if total_classes > 0 else 0
        student = db.query(Student).filter(Student.roll_number == roll_number).first()  # type: ignore
        student_name = student.full_name if student else "Unknown"

        student_summaries.append({
            "roll_number": roll_number,
            "student_name": student_name,
            "total_classes": total_classes,
            "present": present_count,
            "absent": absent_count,
            "attendance_percentage": round(attendance_percentage, 2)
        })

    attendance_details = [
        {
            "roll_number": record.roll_number,
            "class_time": record.class_time.isoformat(),
            "is_absent": record.is_absent,
            "marked_by_phd": record.marked_by_phd
        }
        for record in records
    ]

    return {
        "section_id": section_id,
        "course_code": course_code,
        "total_classes": total_classes,
        "student_summaries": student_summaries,
        "attendance_details": attendance_details
    }


@router.get("/class_schedules")
def get_class_schedules(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    """
    Returns all class schedules that the current PhD student has access to,
    including details such as schedule_id, course_code, section_id, day_of_week,
    start_time, end_time, section_name, and classroom.
    """
    schedules = db.query(ClassSchedule).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    result = []
    for sch in schedules:
        result.append({
            "schedule_id": sch.schedule_id,
            "course_code": sch.course_code,
            "section_id": sch.section_id,
            "day_of_week": sch.day_of_week,
            "start_time": sch.start_time.strftime("%H:%M") if sch.start_time else None,
            "end_time": sch.end_time.strftime("%H:%M") if sch.end_time else None,
            "section_name": sch.section.section_name if sch.section else "N/A",
            "location": sch.classroom
        })
    total = len(result)
    return {"schedules": result, "total": total}


@router.get("/attendance_summary")
def get_attendance_summary(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    total_records = db.query(AttendanceReport).filter(
        AttendanceReport.marked_by_phd == current_phd.employee_id  # type: ignore
    ).count()
    return {"total_records": total_records}


@router.get("/pending_corrections")
def get_pending_corrections(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    phd_sections = db.query(ClassSchedule.section_id, ClassSchedule.course_code).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    phd_section_ids = [section.section_id for section in phd_sections]
    phd_course_codes = [section.course_code for section in phd_sections]

    pending_requests = db.query(AttendanceCorrection).filter(
        AttendanceCorrection.section_id.in_(phd_section_ids),
        AttendanceCorrection.course_code.in_(phd_course_codes),
        AttendanceCorrection.correction_status == "Pending"
    ).all()

    result = [
        {
            "correction_id": c.correction_id,
            "report_id": c.report_id,
            "course_code": c.course_code,
            "request_time": c.request_time.isoformat() if c.request_time else None,
            "correction_status": c.correction_status,
            "reason": c.reason,
            "feedback": c.feedback or ""
        }
        for c in pending_requests
    ]

    return {"pending_corrections": result}


@router.get("/students/{section_id}")
def get_students(
    section_id: int,
    db: Session = Depends(get_db),
    current_phd: PhDStudent = Depends(get_current_phd_student)
):
    access = db.query(ClassScheduleAccess).join(ClassSchedule).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id,  # type: ignore
        ClassSchedule.section_id == section_id
    ).first()
    if not access:
        raise HTTPException(status_code=403, detail="Unauthorized: You do not have access to this section.")

    students = db.query(Student).join(StudentSection).filter(
        StudentSection.section_id == section_id
    ).all()

    result = [
        {"roll_number": s.roll_number, "full_name": s.full_name, "email": s.email}
        for s in students
    ]
    return {"students": result}


@router.get("/assigned_sections")
def get_assigned_sections(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    """
    Returns a unique list of sections (with section_id, section_name, and course_code)
    assigned to the current PhD student.
    """
    schedules = db.query(ClassSchedule).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    unique_sections = {}
    for schedule in schedules:
        key = schedule.section_id
        if key not in unique_sections:
            unique_sections[key] = {
                "section_id": schedule.section_id,
                "section_name": schedule.section.section_name if schedule.section else "N/A",
                "course_code": schedule.course_code
            }

    return {"schedules": list(unique_sections.values())}


@router.get("/attendance_records")
def get_attendance_records(
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    """
    Returns all attendance records for sections the current PhD student has access to.
    """
    phd_sections = db.query(ClassSchedule.section_id, ClassSchedule.course_code).join(
        ClassScheduleAccess
    ).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    phd_section_ids = [s.section_id for s in phd_sections]
    phd_course_codes = [s.course_code for s in phd_sections]

    records = db.query(AttendanceReport).filter(
        AttendanceReport.section_id.in_(phd_section_ids),
        AttendanceReport.course_code.in_(phd_course_codes)
    ).all()

    result = []
    for record in records:
        student_name = record.student.full_name if record.student else ""
        result.append({
            "report_id": record.report_id,
            "roll_number": record.roll_number,
            "student_name": student_name,
            "class_time": record.class_time.isoformat(),
            "course_code": record.course_code,
            "is_absent": record.is_absent,
            "section_id": record.section_id
        })

    return {"records": result}


@router.get("/effective_schedules/{week_start_date}")
def get_effective_schedules(
    week_start_date: date,
    current_phd: PhDStudent = Depends(get_current_phd_student),
    db: Session = Depends(get_db)
):
    week_end_date = week_start_date + timedelta(days=6)

    original_schedules = db.query(ClassSchedule).join(ClassScheduleAccess).filter(
        ClassScheduleAccess.phd_student_id == current_phd.employee_id  # type: ignore
    ).all()

    dsu_entries = db.query(DynamicScheduleUpdate).filter(
        DynamicScheduleUpdate.schedule_id.in_([sch.schedule_id for sch in original_schedules]),
        or_(
            and_(DynamicScheduleUpdate.original_date >= week_start_date,
                 DynamicScheduleUpdate.original_date <= week_end_date),
            and_(DynamicScheduleUpdate.new_date >= week_start_date, DynamicScheduleUpdate.new_date <= week_end_date)
        )
    ).all()

    excluded_dates = set()
    for dsu in dsu_entries:
        if dsu.update_type in ["CANCELLED", "RESCHEDULED"] and dsu.original_date:
            excluded_dates.add((dsu.schedule_id, dsu.original_date))

    rescheduled_entries = [
        dsu for dsu in dsu_entries
        if dsu.update_type == "RESCHEDULED" and dsu.new_date and week_start_date <= dsu.new_date <= week_end_date
    ]

    effective_schedules = []

    for sch in original_schedules:
        week_dates = [
            week_start_date + timedelta(days=i)
            for i in range(7)
            if (week_start_date + timedelta(days=i)).strftime("%A") == sch.day_of_week
        ]
        for d in week_dates:
            if (sch.schedule_id, d) not in excluded_dates:
                effective_schedules.append({
                    "date": d.isoformat(),
                    "course_code": sch.course_code,
                    "section_id": sch.section_id,
                    "section_name": sch.section.section_name if sch.section else "N/A",
                    "start_time": sch.start_time.strftime("%H:%M"),
                    "end_time": sch.end_time.strftime("%H:%M"),
                    "location": sch.classroom,
                    "is_updated": False
                })

    for dsu in rescheduled_entries:
        effective_schedules.append({
            "date": dsu.new_date.isoformat(),
            "course_code": dsu.class_schedule.course_code if dsu.class_schedule else "N/A",
            "section_id": dsu.class_schedule.section_id if dsu.class_schedule else "N/A",
            "section_name": dsu.class_schedule.section.section_name if dsu.class_schedule and dsu.class_schedule.section
            else "N/A",
            "start_time": dsu.new_start_time.strftime("%H:%M") if dsu.new_start_time else "N/A",
            "end_time": dsu.new_end_time.strftime("%H:%M") if dsu.new_end_time else "N/A",
            "location": dsu.new_classroom,
            "is_updated": True
        })

    effective_schedules.sort(key=lambda x: (x["date"], x["start_time"]))

    return {"schedules": effective_schedules}
