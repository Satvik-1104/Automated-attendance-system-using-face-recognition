// src/pages/MarkAttendance.jsx
import BaseMarkAttendance from "./BaseMarkAttendance";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const MarkAttendance = () => {
  const facultyApiEndpoints = {
    assignedSections: `${BASE_URL}/faculty/assigned_sections`,
    students: `${BASE_URL}/faculty/students`,
    markAttendance: `${BASE_URL}/faculty/mark_attendance`
  };

  return (
    <BaseMarkAttendance 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty - Mark Attendance"
      role="faculty"
    />
  );
};

export default MarkAttendance;