// src/pages/PhDMarkAttendance.jsx
import BaseMarkAttendance from "./BaseMarkAttendance";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhDMarkAttendance = () => {
  const phdApiEndpoints = {
    assignedSections: `${BASE_URL}/phd_students/assigned_sections`,
    students: `${BASE_URL}/phd_students/students`,
    markAttendance: `${BASE_URL}/phd_students/mark_attendance`
  };

  return (
    <BaseMarkAttendance 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD - Mark Attendance"
      role="phd"
    />
  );
};

export default PhDMarkAttendance;