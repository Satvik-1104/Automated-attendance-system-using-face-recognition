from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, Time, TIMESTAMP, Text, text, JSON, Date
from sqlalchemy.orm import relationship
import datetime
from .database import Base


class Student(Base):
    __tablename__ = 'students'

    roll_number = Column(Integer, primary_key=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    branch = Column(String(3), nullable=False)
    semester = Column(Integer, nullable=False)
    batch = Column(Integer, nullable=False)
    username = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    photo_folder = Column(String(255), nullable=True)

    # Relationships
    student_sections = relationship('StudentSection', back_populates='student', cascade="all, delete-orphan")
    attendance_reports = relationship('AttendanceReport', back_populates='student', cascade="all, delete-orphan")
    entry_exit_times = relationship('EntryExitTime', back_populates='student', cascade="all, delete-orphan")
    corrections = relationship("AttendanceCorrection", back_populates="student", cascade="all, delete-orphan")
    # valid_student = relationship("ValidStudent", back_populates="student")
    otp_verifications = relationship(
        "OTPVerification",
        back_populates="student",
        cascade="all, delete-orphan",
        foreign_keys="[OTPVerification.student_id]"
    )


class Section(Base):
    __tablename__ = 'sections'

    section_id = Column(Integer, primary_key=True)
    section_name = Column(String(10), nullable=False)
    batch = Column(Integer, nullable=False)
    course_code = Column(String(10), nullable=True)

    student_sections = relationship('StudentSection', back_populates='section', cascade="all, delete-orphan")
    class_schedules = relationship('ClassSchedule', back_populates='section', cascade="all, delete-orphan")
    attendance_reports = relationship('AttendanceReport', back_populates='section', cascade="all, delete-orphan")
    correction_requests = relationship(
        "AttendanceCorrection",
        back_populates="section",
        cascade="all, delete-orphan"
    )


class StudentSection(Base):
    __tablename__ = 'student_sections'

    roll_number = Column(
        Integer,
        ForeignKey('students.roll_number', ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        index=True
    )
    section_id = Column(
        Integer,
        ForeignKey('sections.section_id', ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        index=True
    )

    student = relationship('Student', back_populates='student_sections')
    section = relationship('Section', back_populates='student_sections')


class AttendanceReport(Base):
    __tablename__ = 'attendance_reports'

    report_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    roll_number = Column(Integer, ForeignKey('students.roll_number', ondelete="CASCADE"), nullable=False, index=True)
    section_id = Column(Integer, ForeignKey('sections.section_id', ondelete="CASCADE"), nullable=False, index=True)
    course_code = Column(String(10), nullable=False)
    class_time = Column(DateTime, nullable=False)
    is_absent = Column(Boolean, default=True)
    correction_applied = Column(Boolean, default=False)
    marked_by_faculty = Column(Integer, ForeignKey("faculty.employee_id", ondelete="SET NULL"), nullable=True)
    marked_by_phd = Column(Integer, ForeignKey("phd_students.employee_id", ondelete="SET NULL"), nullable=True)

    student = relationship('Student', back_populates='attendance_reports')
    section = relationship('Section', back_populates='attendance_reports')
    corrections = relationship("AttendanceCorrection", back_populates="attendance_report")
    faculty_marker = relationship("Faculty", foreign_keys=[marked_by_faculty])
    phd_marker = relationship("PhDStudent", foreign_keys=[marked_by_phd])


class AttendanceCorrection(Base):
    __tablename__ = "attendance_corrections"

    correction_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    roll_number = Column(Integer, ForeignKey("students.roll_number", ondelete="CASCADE"), nullable=False)
    report_id = Column(Integer, ForeignKey("attendance_reports.report_id", ondelete="CASCADE"), nullable=False)
    section_id = Column(Integer, ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False)
    course_code = Column(String(10), nullable=False)
    request_time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    correction_status = Column(String(20), default="Pending")
    reason = Column(Text, nullable=False)
    supporting_image = Column(String(255), nullable=True)
    processed_by_faculty = Column(Integer, ForeignKey("faculty.employee_id"), nullable=True)
    processed_by_phd = Column(Integer, ForeignKey("phd_students.employee_id"), nullable=True)
    feedback = Column(Text, nullable=True)

    student = relationship("Student", back_populates="corrections")
    attendance_report = relationship("AttendanceReport", back_populates="corrections")
    section = relationship("Section", back_populates="correction_requests")
    faculty = relationship("Faculty", back_populates="approved_corrections", foreign_keys=[processed_by_faculty])
    phd_student = relationship("PhDStudent", back_populates="processed_corrections", foreign_keys=[processed_by_phd])


class DynamicScheduleUpdate(Base):
    __tablename__ = 'dynamic_schedule_updates'

    update_id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_id = Column(
        Integer,
        ForeignKey('class_schedules.schedule_id', ondelete="CASCADE"),
        nullable=True,  # Allow null for one-time schedules (optional extension)
        index=True
    )
    update_type = Column(String(20), nullable=False)  # e.g., "CANCELLED", "RESCHEDULED"
    original_date = Column(Date, nullable=True)  # Date of the original schedule instance
    new_date = Column(Date, nullable=True)  # New date for rescheduled classes
    new_start_time = Column(Time, nullable=True)
    new_end_time = Column(Time, nullable=True)
    new_classroom = Column(String(50), nullable=True)
    reason = Column(String(255), nullable=True)
    update_timestamp = Column(DateTime, default=datetime.datetime.now)
    updated_by_faculty = Column(Integer, ForeignKey("faculty.employee_id", ondelete="SET NULL"), nullable=True)
    updated_by_phd = Column(Integer, ForeignKey("phd_students.employee_id", ondelete="SET NULL"), nullable=True)

    class_schedule = relationship('ClassSchedule', back_populates='dynamic_schedule_updates')
    faculty = relationship("Faculty", back_populates="schedule_updates", foreign_keys=[updated_by_faculty])
    phd_student = relationship("PhDStudent", back_populates="schedule_updates", foreign_keys=[updated_by_phd])


class Faculty(Base):
    __tablename__ = 'faculty'

    employee_id = Column(Integer, ForeignKey('valid_faculty.employee_id'), primary_key=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    username = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)

    # Relationship with ValidFaculty
    valid_faculty = relationship("ValidFaculty", back_populates="faculty")
    schedule_access = relationship("ClassScheduleAccess", back_populates="faculty", cascade="all, delete-orphan")
    approved_corrections = relationship(
        "AttendanceCorrection",
        back_populates="faculty",
        foreign_keys=[AttendanceCorrection.processed_by_faculty]
    )
    marked_attendance = relationship(
        "AttendanceReport",
        back_populates="faculty_marker",
        foreign_keys=[AttendanceReport.marked_by_faculty]
    )
    schedule_updates = relationship(
        "DynamicScheduleUpdate",
        back_populates="faculty",
        foreign_keys=[DynamicScheduleUpdate.updated_by_faculty]
    )
    otp_verifications = relationship(
        "OTPVerification",
        back_populates="faculty",
        cascade="all, delete-orphan",
        foreign_keys="[OTPVerification.faculty_id]"
    )


class PhDStudent(Base):
    __tablename__ = 'phd_students'

    employee_id = Column(Integer, ForeignKey('valid_phd_students.employee_id'), primary_key=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    username = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)

    # Relationship with ValidPhDStudent
    valid_phd_student = relationship("ValidPhDStudent", back_populates="phd_student")
    schedule_access = relationship("ClassScheduleAccess", back_populates="phd_student", cascade="all, delete-orphan")
    processed_corrections = relationship(
        "AttendanceCorrection",
        back_populates="phd_student",
        foreign_keys=[AttendanceCorrection.processed_by_phd]
    )
    marked_attendance = relationship(
        "AttendanceReport",
        back_populates="phd_marker",
        foreign_keys=[AttendanceReport.marked_by_phd]
    )
    schedule_updates = relationship(
        "DynamicScheduleUpdate",
        back_populates="phd_student",
        foreign_keys=[DynamicScheduleUpdate.updated_by_phd]
    )
    otp_verifications = relationship(
        "OTPVerification",
        back_populates="phd_student",
        cascade="all, delete-orphan",
        foreign_keys="[OTPVerification.phd_id]"
    )


class ClassSchedule(Base):
    __tablename__ = 'class_schedules'

    schedule_id = Column(Integer, primary_key=True, autoincrement=True)
    section_id = Column(Integer, ForeignKey('sections.section_id', ondelete="CASCADE"), nullable=False, index=True)
    course_code = Column(String(10), nullable=False)
    day_of_week = Column(String(10), nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    classroom = Column(String(50), nullable=False)

    section = relationship('Section', back_populates='class_schedules')
    entry_exit_times = relationship('EntryExitTime', back_populates='class_schedule', cascade="all, delete-orphan")
    dynamic_schedule_updates = relationship(
        'DynamicScheduleUpdate',
        back_populates='class_schedule',
        cascade="all, delete-orphan"
    )
    access_entries = relationship("ClassScheduleAccess", back_populates="class_schedule", cascade="all, delete-orphan")


class ClassScheduleAccess(Base):
    __tablename__ = "class_schedule_access"

    id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_id = Column(
        Integer,
        ForeignKey("class_schedules.schedule_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    faculty_id = Column(Integer,
                        ForeignKey("faculty.employee_id", ondelete="CASCADE"),
                        nullable=True,
                        index=True
                        )
    phd_student_id = Column(
        Integer,
        ForeignKey("phd_students.employee_id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    class_schedule = relationship("ClassSchedule", back_populates="access_entries")
    faculty = relationship("Faculty", back_populates="schedule_access")
    phd_student = relationship("PhDStudent", back_populates="schedule_access")


class EntryExitTime(Base):
    __tablename__ = 'entry_exit_times'

    entry_exit_id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(Integer, ForeignKey('students.roll_number', ondelete="CASCADE"), nullable=False, index=True)
    schedule_id = Column(
        Integer,
        ForeignKey('class_schedules.schedule_id', ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    overlapping_classes = Column(JSON)

    student = relationship('Student', back_populates='entry_exit_times')
    class_schedule = relationship('ClassSchedule', back_populates='entry_exit_times')


# class ValidStudent(Base):
#     __tablename__ = "valid_students"

#     roll_number = Column(Integer, primary_key=True)
#     email = Column(String(100), nullable=False, unique=True)
#     is_verified = Column(Boolean, default=False)

#     # Relationship with Student
#     student = relationship("Student", back_populates="valid_student", uselist=False)


class ValidFaculty(Base):
    __tablename__ = "valid_faculty"

    employee_id = Column(Integer, primary_key=True)
    email = Column(String(100), nullable=False, unique=True)
    is_verified = Column(Boolean, default=False)

    # Relationship with Faculty
    faculty = relationship("Faculty", back_populates="valid_faculty", uselist=False)


class ValidPhDStudent(Base):
    __tablename__ = "valid_phd_students"

    employee_id = Column(Integer, primary_key=True)
    email = Column(String(100), nullable=False, unique=True)
    is_verified = Column(Boolean, default=False)

    # Relationship with PhDStudent
    phd_student = relationship("PhDStudent", back_populates="valid_phd_student", uselist=False)


class OTPVerification(Base):
    __tablename__ = "otp_verifications"

    otp_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), nullable=False)  # Remove ForeignKey constraint
    otp_code = Column(String(6), nullable=False)
    expiry_time = Column(TIMESTAMP(timezone=True), nullable=False)
    user_type = Column(String(20), nullable=False)
    purpose = Column(String(20), default="registration")
    student_id = Column(Integer, ForeignKey('students.roll_number', ondelete="CASCADE"), nullable=True)
    faculty_id = Column(Integer, ForeignKey('faculty.employee_id', ondelete="CASCADE"), nullable=True)
    phd_id = Column(Integer, ForeignKey('phd_students.employee_id', ondelete="CASCADE"), nullable=True)
    is_verified = Column(Boolean, default=False)

    # Relationships
    student = relationship("Student", back_populates="otp_verifications")
    faculty = relationship("Faculty", back_populates="otp_verifications")
    phd_student = relationship("PhDStudent", back_populates="otp_verifications")


class MLAttendanceLog(Base):
    __tablename__ = "ml_attendance_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(Integer, nullable=False)
    classroom_code = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
