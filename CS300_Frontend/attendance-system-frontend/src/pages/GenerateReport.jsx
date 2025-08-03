// src/pages/GenerateReport.jsx
import BaseGenerateReport from "./BaseGenerateReport";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const GenerateReport = () => {
  const facultyApiEndpoints = {
    assignedSections: `${BASE_URL}/faculty/assigned_sections`,
    generateReport: `${BASE_URL}/faculty/generate_report`
  };

  return (
    <BaseGenerateReport 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty - Generate Attendance Report"
      role="faculty"
    />
  );
};

export default GenerateReport;