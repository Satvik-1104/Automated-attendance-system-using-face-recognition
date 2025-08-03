from pydantic import BaseModel, EmailStr, field_validator, Field
from datetime import datetime, date
from typing import Optional, List  # , Dict, Union
import pytz

ist = pytz.timezone('Asia/Kolkata')


class LoginRequest(BaseModel):
    username: str
    password: str


class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class StudentRegister(BaseModel):
    roll_number: int
    full_name: str
    email: EmailStr
    branch: str
    semester: int
    batch: int
    username: str
    password: str
    sections_selected: List[str]
    # role: str = "student"


class UploadPhotoRequest(BaseModel):
    roll_number: int


class TeachingAssignment(BaseModel):
    batch: int
    course_code: List[str]
    sections: List[str]


class FacultyRegister(BaseModel):
    faculty_id: int
    name: str
    email: EmailStr
    username: str
    password: str
    department: str
    teaching_assignments: List[TeachingAssignment]
    # role: str = "faculty"


class PhDStudentRegister(BaseModel):
    phd_id: int
    name: str
    email: EmailStr
    username: str
    password: str
    department: str
    teaching_assignments: List[TeachingAssignment]
    # role: str = "phd_student"


class AttendanceRecord(BaseModel):
    class_time: datetime
    is_absent: bool
    course_code: str


class MarkAttendanceRequest(BaseModel):
    roll_number: int
    section_id: int
    class_time: datetime  # We expect class_time to already be in IST
    is_present: bool
    course_code: str

    @field_validator("class_time", mode="before")
    @classmethod
    def validate_ist_timezone(cls, value):
        """Ensure class_time is in IST, but don't reapply timezone if already set."""
        if isinstance(value, str):  # Convert ISO 8601 format string to datetime
            value = datetime.fromisoformat(value)

        # If timezone is already set (like +05:30), don't modify it
        if value.tzinfo is not None:
            return value  # Already has a timezone, return as is

        # Otherwise, assume it's naive and add IST
        return value.replace(tzinfo=ist)


class ViewAttendanceRequest(BaseModel):
    section_id: int


class AttendanceCorrectionRequest(BaseModel):
    roll_number: int
    report_id: int  # ID of the AttendanceReport to correct
    course_code: str
    reason: str
    supporting_image: Optional[str] = None


class AttendanceCorrectionResponse(BaseModel):
    correction_id: int
    roll_number: int
    report_id: int
    course_code: str
    request_time: datetime  # Will auto-convert to UTC
    correction_status: str
    reason: str
    supporting_image: Optional[str]


class ApproveCorrectionRequest(BaseModel):
    correction_id: int
    approval_status: bool
    feedback: Optional[str] = None


class BulkApproveCorrectionRequest(BaseModel):
    correction_ids: List[int]
    approval_status: bool
    feedback: Optional[str] = None


class UpdateScheduleRequest(BaseModel):
    section_id: int
    course_code: str
    old_time: datetime
    new_time: datetime


class MoveClassRequest(BaseModel):
    section_id: int
    course_code: str
    old_location: str
    new_location: str


class CancelClassRequest(BaseModel):
    section_id: int
    course_code: str
    class_time: datetime
    reason: Optional[str] = None


class AttendanceReportResponse(BaseModel):
    section_id: int
    course_code: str
    roll_number: int
    student_attendance: List[AttendanceRecord]


class AttendanceAnalyticsResponse(BaseModel):
    section_id: int
    total_classes: int
    total_present: int
    total_absent: int
    attendance_percentage: float


class AttendanceCorrectionDecision(BaseModel):
    correction_id: int
    approve: bool
    feedback: Optional[str] = None


class AttendanceReportRequest(BaseModel):
    section: str


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str


class StudentOTPRequest(BaseModel):
    roll_number: int
    email: EmailStr


class FacultyOTPRequest(BaseModel):
    faculty_id: int
    email: EmailStr


class PhDOTPRequest(BaseModel):
    phd_id: int
    email: EmailStr


class MLAttendanceLogRequest(BaseModel):
    roll_number: int
    classroom_code: str
    timestamp: datetime

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value):
        # If the value is a string, convert it to a datetime object
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            return value.replace(tzinfo=ist)
        return value


class EntryExitLogRequest(BaseModel):
    roll_number: int
    classroom_code: str
    timestamp: datetime
    direction: str = Field(..., description="Must be either 'entry' or 'exit'")

    @field_validator("direction", mode="before")
    def validate_direction(cls, value):
        if isinstance(value, str):
            value = value.lower()
        if value not in {"entry", "exit"}:
            raise ValueError("Direction must be either 'entry' or 'exit'")
        return value


class UnifiedScheduleUpdateRequest(BaseModel):
    update_type: str  # "CANCELLED" or "RESCHEDULED"
    course_code: str
    section_id: int
    original_date: date  # The specific date of the original schedule to update
    new_time: datetime | None = None  # New datetime for rescheduling (optional for "CANCELLED")
    new_location: str | None = None  # New location (optional for "CANCELLED")
    reason: str

    @field_validator("new_time", mode="before")
    @classmethod
    def ensure_timezone(cls, value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            ist = pytz.timezone("Asia/Kolkata")
            value = value.replace(tzinfo=ist)
        return value


class AttendanceRecordResponse(BaseModel):
    roll_number: int
    class_time: str  # ISO formatted datetime string
    is_absent: bool
    marked_by_faculty: int | None  # It may be null if not marked


class StudentAttendanceSummary(BaseModel):
    roll_number: int
    student_name: str
    total_classes: int
    present: int
    absent: int
    attendance_percentage: float


class GenerateReportResponse(BaseModel):
    section_id: int
    course_code: str
    total_classes: int
    student_summaries: List[StudentAttendanceSummary]
    attendance_details: List[dict]
