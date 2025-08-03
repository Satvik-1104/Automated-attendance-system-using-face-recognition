// src/pages/FacultyAttendanceReports.jsx
import BaseAttendanceReports from "./BaseAttendanceReports";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const FacultyAttendanceReports = () => {
  const facultyApiEndpoints = {
    assignedSections: `${BASE_URL}/faculty/assigned_sections`,
    attendanceRecords: `${BASE_URL}/faculty/attendance_records`
  };

  return (
    <BaseAttendanceReports 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty - Attendance Reports"
      role="faculty"
    />  
  );
};

export default FacultyAttendanceReports;