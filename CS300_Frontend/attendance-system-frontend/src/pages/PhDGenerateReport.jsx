// src/pages/PhDGenerateReport.jsx
import BaseGenerateReport from "./BaseGenerateReport";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhDGenerateReport = () => {
  const phdApiEndpoints = {
    assignedSections: `${BASE_URL}/phd_students/assigned_sections`,
    generateReport: `${BASE_URL}/phd_students/generate_report`
  };

  return (
    <BaseGenerateReport 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD - Generate Attendance Report"
      role="phd"
    />
  );
};

export default PhDGenerateReport;