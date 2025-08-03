// src/pages/PhDHome.jsx
import BaseHome from "./BaseHome";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhDHome = () => {
  const phdApiEndpoints = {
    classSchedules: `${BASE_URL}/phd_students/class_schedules`,
    pendingCorrections: `${BASE_URL}/phd_students/pending_corrections`,
    attendanceSummary: `${BASE_URL}/phd_students/attendance_summary`
  };

  return (
    <BaseHome 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD Dashboard"
      role="phd"
    />
  );
};

export default PhDHome;