// src/pages/PhDApproveCorrections.jsx
import BaseApproveCorrections from "./BaseApproveCorrections";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhDApproveCorrections = () => {
  const phdApiEndpoints = {
    pendingCorrections: `${BASE_URL}/phd_students/pending_corrections`,
    approveCorrection: `${BASE_URL}/phd_students/approve_correction`,
    uploadBase: `${BASE_URL}/uploads`
  };

  return (
    <BaseApproveCorrections 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD - Approve Corrections"
      role="phd"
    />
  );
};

export default PhDApproveCorrections;