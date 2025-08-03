// src/pages/PhDAttendanceReports.jsx
import BaseAttendanceReports from "./BaseAttendanceReports";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;
const PhDAttendanceReports = () => {
  const phdApiEndpoints = {
    assignedSections: `${BASE_URL}/phd_students/assigned_sections`,
    attendanceRecords: `${BASE_URL}/phd_students/attendance_records`
  };

  return (
    <BaseAttendanceReports 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD - Attendance Reports"
      role="phd"
    />
  );
};

export default PhDAttendanceReports;