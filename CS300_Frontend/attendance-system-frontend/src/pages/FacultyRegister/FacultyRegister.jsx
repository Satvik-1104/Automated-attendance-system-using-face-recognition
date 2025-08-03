// src/pages/FacultyRegister.jsx
import React, { useState } from "react";
import FacultyInitialRegistration from "./FacultyInitialRegistration";
import OTPVerification from "../../components/OTPVerification";
import FacultyCompleteRegistration from "./FacultyCompleteRegistration";
import FacultyAcknowledgement from "./FacultyAcknowledgement";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const FacultyRegister = () => {
  const [step, setStep] = useState("initial");
  const [facultyId, setFacultyId] = useState("");
  const [email, setEmail] = useState("");

  const handleInitialSubmit = (id, mail) => {
    setFacultyId(id);
    setEmail(mail);
    setStep("otp");
  };

  const handleOTPSubmit = () => {
    setStep("complete");
  };

  const handleCompleteSubmit = () => {
    setStep("acknowledgement");
  };

  return (
    <div style={styles.container}>
      {step === "initial" && <FacultyInitialRegistration onSubmit={handleInitialSubmit} />}
      {step === "otp" && (
        <OTPVerification
          email={email}
          onSubmit={handleOTPSubmit}
          apiEndpoint={`${BASE_URL}/faculty/register/verify_otp`}
        />
      )}
      {step === "complete" && (
        <FacultyCompleteRegistration
          facultyId={facultyId}
          email={email}
          onSubmit={handleCompleteSubmit}
        />
      )}
      {step === "acknowledgement" && <FacultyAcknowledgement />}
    </div>
  );
};

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f9fa",
  },
};

export default FacultyRegister;