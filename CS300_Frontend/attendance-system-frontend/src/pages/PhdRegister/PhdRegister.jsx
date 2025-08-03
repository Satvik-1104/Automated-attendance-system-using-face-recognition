// src/pages/PhdRegister.jsx
import React, { useState } from "react";
import PhdInitialRegistration from "./PhdInitialRegistration";
import OTPVerification from "../../components/OTPVerification";
import PhdCompleteRegistration from "./PhdCompleteRegistration";
import PhdAcknowledgement from "./PhdAcknowledgement";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhdRegister = () => {
  const [step, setStep] = useState("initial");
  const [phdId, setPhdId] = useState("");
  const [email, setEmail] = useState("");

  const handleInitialSubmit = (id, mail) => {
    setPhdId(id);
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
      {step === "initial" && <PhdInitialRegistration onSubmit={handleInitialSubmit} />}
      {step === "otp" && (
        <OTPVerification
          email={email}
          onSubmit={handleOTPSubmit}
          apiEndpoint={`${BASE_URL}/phd/register/verify_otp`}
        />
      )}
      {step === "complete" && (
        <PhdCompleteRegistration phdId={phdId} email={email} onSubmit={handleCompleteSubmit} />
      )}
      {step === "acknowledgement" && <PhdAcknowledgement />}
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

export default PhdRegister;