from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Optional, Tuple, cast
import logging
from collections import defaultdict

from ..database import get_db
from ..models import MLAttendanceLog, ClassSchedule, DynamicScheduleUpdate, AttendanceReport, EntryExitTime, \
    StudentSection, Student
from ..schemas import MLAttendanceLogRequest, EntryExitLogRequest

router = APIRouter(tags=["attendance"])

GAP_THRESHOLD_MINUTES: int = 5
PRESENCE_THRESHOLD_RATIO: float = 0.40


def derive_schedule_id(log_timestamp: datetime, classroom_code: str, db: Session) -> Optional[int]:
    log_date = log_timestamp.date()
    log_time = log_timestamp.time()
    dynamic_update = (
        db.query(DynamicScheduleUpdate)
        .join(ClassSchedule, DynamicScheduleUpdate.schedule_id == ClassSchedule.schedule_id)  # type: ignore
        .filter(ClassSchedule.classroom == classroom_code)
        .filter(DynamicScheduleUpdate.update_type.in_(["Time Change", "TIME_LOC CHANGE"]))
        .filter(DynamicScheduleUpdate.new_date == log_date)
        .filter(DynamicScheduleUpdate.new_start_time <= log_time)
        .filter(DynamicScheduleUpdate.new_end_time >= log_time)
        .order_by(DynamicScheduleUpdate.update_timestamp.desc())
        .first()
    )
    if dynamic_update:
        return dynamic_update.schedule_id
    schedule = (
        db.query(ClassSchedule)
        .filter(ClassSchedule.classroom == classroom_code)
        .filter(ClassSchedule.day_of_week == log_timestamp.strftime("%A"))
        .filter(ClassSchedule.start_time <= log_time)
        .filter(ClassSchedule.end_time >= log_time)
        .first()
    )
    if schedule:
        return schedule.schedule_id
    return None


@router.post("/ml_log")
def record_ml_log(log: MLAttendanceLogRequest, db: Session = Depends(get_db)) -> dict:
    new_log = MLAttendanceLog(
        roll_number=log.roll_number,
        classroom_code=log.classroom_code,
        timestamp=log.timestamp
    )
    db.add(new_log)
    db.commit()
    return {"message": "ML log recorded successfully"}


def get_effective_schedule(schedule_instance: ClassSchedule, db: Session) -> Optional[Tuple[time, time, str]]:
    today = datetime.today().date()
    update = (
        db.query(DynamicScheduleUpdate)
        .filter_by(schedule_id=schedule_instance.schedule_id)
        .filter(DynamicScheduleUpdate.new_date == today)
        .order_by(DynamicScheduleUpdate.update_timestamp.desc())
        .first()
    )
    if update:
        if update.update_type.upper() == "CANCELLED":
            return None
        elif update.update_type.upper() in ["TIME CHANGE", "LOCATION CHANGE", "TIME_LOC CHANGE"]:
            effective_start = update.new_start_time
            effective_end = update.new_end_time
            effective_classroom = (
                update.new_classroom) if update.new_classroom is not None else schedule_instance.classroom
            return effective_start, effective_end, effective_classroom
    return schedule_instance.start_time, schedule_instance.end_time, schedule_instance.classroom


def group_logs_by_student(logs: List[MLAttendanceLog]) -> Dict[int, List[datetime]]:
    grouped: Dict[int, List[datetime]] = {}
    for log in logs:
        grouped.setdefault(log.roll_number, []).append(log.timestamp)
    for roll in grouped:
        grouped[roll].sort()
    return grouped


def compute_presence_duration(timestamps: List[datetime]) -> timedelta:
    if not timestamps:
        return timedelta(0)
    total_duration = timedelta(0)
    block_start = timestamps[0]
    block_end = timestamps[0]
    for t in timestamps[1:]:
        if (t - block_end) <= timedelta(minutes=GAP_THRESHOLD_MINUTES):
            block_end = t
        else:
            total_duration += (block_end - block_start) + timedelta(minutes=1)
            block_start = t
            block_end = t
    total_duration += (block_end - block_start) + timedelta(minutes=1)
    return total_duration


@router.post("/process/{schedule_id}")
def process_attendance(schedule_id: int, db: Session = Depends(get_db)) -> dict:
    schedule_instance: Optional[ClassSchedule] = db.query(ClassSchedule).filter(
        ClassSchedule.schedule_id == schedule_id
    ).first()
    if not schedule_instance:
        raise HTTPException(status_code=404, detail="Schedule not found")
    effective_data = get_effective_schedule(schedule_instance, db)
    if not effective_data:
        return {"message": "Class cancelled; attendance not marked."}
    effective_start, effective_end, effective_classroom = effective_data
    today: date = datetime.today().date()
    effective_start_dt: datetime = datetime.combine(today, effective_start)
    effective_end_dt: datetime = datetime.combine(today, effective_end)
    processing_end: datetime = effective_end_dt - timedelta(minutes=15)
    ml_logs: List[MLAttendanceLog] = cast(
        List[MLAttendanceLog],
        db.query(MLAttendanceLog).filter(
            and_(
                MLAttendanceLog.classroom_code == effective_classroom,
                MLAttendanceLog.timestamp >= effective_start_dt,
                MLAttendanceLog.timestamp <= processing_end
            )
        ).all()
    )
    if not ml_logs:
        return {"message": "No ML logs found for this session."}
    grouped_logs = group_logs_by_student(ml_logs)
    class_duration: timedelta = effective_end_dt - effective_start_dt
    threshold_duration: timedelta = class_duration * PRESENCE_THRESHOLD_RATIO
    marked_students: List[int] = []
    for roll, timestamps in grouped_logs.items():
        presence = compute_presence_duration(timestamps)
        condition = and_(
            AttendanceReport.roll_number == roll,
            AttendanceReport.section_id == schedule_instance.section_id,
            AttendanceReport.course_code == schedule_instance.course_code,
            AttendanceReport.class_time == effective_start_dt
        )
        record: Optional[AttendanceReport] = db.query(AttendanceReport).filter(condition).first()
        if presence >= threshold_duration:
            if record:
                record.is_absent = False
            else:
                new_record = AttendanceReport(
                    roll_number=roll,
                    section_id=schedule_instance.section_id,
                    course_code=schedule_instance.course_code,
                    class_time=effective_start_dt,
                    is_absent=False
                )
                db.add(new_record)
            marked_students.append(roll)
        else:
            if record:
                record.is_absent = True
            else:
                new_record = AttendanceReport(
                    roll_number=roll,
                    section_id=schedule_instance.section_id,
                    course_code=schedule_instance.course_code,
                    class_time=effective_start_dt,
                    is_absent=True
                )
                db.add(new_record)
    db.commit()
    logging.info(f"Processed attendance for schedule {schedule_id}. Marked present: {marked_students}")
    return {"message": "Attendance processed", "marked_students": marked_students}


@router.post("/entry_exit")
def record_entry_exit(log: EntryExitLogRequest, db: Session = Depends(get_db)) -> dict:
    schedule_id = derive_schedule_id(log.timestamp, log.classroom_code, db)
    if not schedule_id:
        raise HTTPException(status_code=404, detail="No matching schedule found for the given timestamp and classroom.")
    if log.direction == "entry":
        new_record = EntryExitTime(
            roll_number=log.roll_number,
            schedule_id=schedule_id,
            entry_time=log.timestamp
        )
        db.add(new_record)
        db.commit()
        return {"message": "Entry recorded successfully"}
    elif log.direction == "exit":
        record: Optional[EntryExitTime] = db.query(EntryExitTime).filter(
            EntryExitTime.roll_number == log.roll_number,
            EntryExitTime.schedule_id == schedule_id,
            EntryExitTime.exit_time.is_(None)
        ).order_by(EntryExitTime.entry_exit_id.desc()).first()
        if not record:
            raise HTTPException(status_code=404, detail="No corresponding entry found for exit.")
        record.exit_time = log.timestamp
        db.commit()
        return {"message": "Exit recorded successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid direction.")


@router.post("/process_entry_exit/{schedule_id}")
def process_entry_exit_attendance(schedule_id: int, db: Session = Depends(get_db)) -> dict:
    # Fetch the schedule
    schedule_instance = db.query(ClassSchedule).filter(ClassSchedule.schedule_id == schedule_id).first()
    if not schedule_instance:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Get effective schedule data (assuming this function exists)
    effective_data = get_effective_schedule(schedule_instance, db)
    if not effective_data:
        return {"message": "Class cancelled; attendance not marked."}

    effective_start, effective_end, _ = effective_data
    today = datetime.today().date()
    effective_start_dt = datetime.combine(today, effective_start)
    effective_end_dt = datetime.combine(today, effective_end)
    processing_end_dt = effective_end_dt - timedelta(minutes=15)  # Process up to 15 mins before end
    class_duration = effective_end_dt - effective_start_dt
    threshold_duration = class_duration * PRESENCE_THRESHOLD_RATIO

    # Step 1: Get all students enrolled in the section
    section_id = schedule_instance.section_id
    enrolled_students = (db.query(Student).join(StudentSection).filter
                         (StudentSection.section_id == section_id).all())  # type: ignore
    enrolled_roll_numbers = {student.roll_number for student in enrolled_students}

    # Step 2: Process entry and exit logs to determine present students
    records = db.query(EntryExitTime).filter(
        EntryExitTime.schedule_id == schedule_id,
        EntryExitTime.entry_time.is_not(None)
    ).all()

    in_times = defaultdict(timedelta)
    for rec in records:
        start = rec.entry_time
        end = rec.exit_time if rec.exit_time and rec.exit_time < processing_end_dt else processing_end_dt
        duration = end - start if end else timedelta(0)
        in_times[rec.roll_number] += duration

    present_students = {roll for roll, total_in in in_times.items() if total_in >= threshold_duration}

    # Step 3: Mark attendance for ALL enrolled students
    class_time = effective_start_dt  # Assuming class_time is the start time
    course_code = schedule_instance.course_code

    for roll in enrolled_roll_numbers:
        is_absent = roll not in present_students  # Absent if not in present_students
        # Check if a record already exists
        record = db.query(AttendanceReport).filter(
            and_(
                AttendanceReport.roll_number == roll,
                AttendanceReport.section_id == section_id,
                AttendanceReport.course_code == course_code,
                AttendanceReport.class_time == class_time
            )
        ).first()

        if record:
            # Update existing record
            record.is_absent = is_absent
        else:
            # Create new record
            new_record = AttendanceReport(
                roll_number=roll,
                section_id=section_id,
                course_code=course_code,
                class_time=class_time,
                is_absent=is_absent
            )
            db.add(new_record)
        db.commit()

    # Step 4: Clean up processed entry/exit records
    db.query(EntryExitTime).filter(EntryExitTime.schedule_id == schedule_id).delete()
    db.commit()

    return {
        "message": "Attendance processed based on entry/exit logs",
        "marked_present": list(present_students),
        "marked_absent": list(enrolled_roll_numbers - present_students)
    }
