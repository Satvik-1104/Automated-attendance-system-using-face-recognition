// src/pages/FacultyHome.jsx
import BaseHome from "./BaseHome";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const FacultyHome = () => {
  const facultyApiEndpoints = {
    classSchedules: `${BASE_URL}/faculty/class_schedules`,
    pendingCorrections: `${BASE_URL}/faculty/pending_corrections`,
    attendanceSummary: `${BASE_URL}/faculty/attendance_summary`
  };

  return (
    <BaseHome 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty Dashboard"
      role="faculty"
    />
  );
};

export default FacultyHome;