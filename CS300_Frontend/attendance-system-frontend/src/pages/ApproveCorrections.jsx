// src/pages/ApproveCorrections.jsx
import BaseApproveCorrections from "./BaseApproveCorrections";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const ApproveCorrections = () => {
  const facultyApiEndpoints = {
    pendingCorrections: `${BASE_URL}/faculty/pending_corrections`,
    approveCorrection: `${BASE_URL}/faculty/approve_correction`,
    uploadBase: `${BASE_URL}/uploads`
  };

  return (
    <BaseApproveCorrections 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty - Approve Corrections"
      role="faculty"
    />
  );
};

export default ApproveCorrections;