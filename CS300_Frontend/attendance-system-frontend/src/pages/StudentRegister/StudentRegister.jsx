// src/pages/StudentRegister/StudentRegister.jsx
import React, { useState } from "react";
import InitialRegistration from "./InitialRegistration";
import OTPVerification from "./OTPVerification";
import CompleteRegistration from "./CompleteRegistration";
import Acknowledgement from "./Acknowledgement";

const StudentRegister = () => {
  const [step, setStep] = useState("initial");
  const [rollNumber, setRollNumber] = useState("");
  const [email, setEmail] = useState("");

  const handleInitialSubmit = (roll, mail) => {
    setRollNumber(roll);
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
    <div style={{ minHeight: "100vh", display: "flex", justifyContent: "center", alignItems: "center", backgroundColor: "#f8f9fa" }}>
      {step === "initial" && <InitialRegistration onSubmit={handleInitialSubmit} />}
      {step === "otp" && <OTPVerification rollNumber={rollNumber} email={email} onSubmit={handleOTPSubmit} />}
      {step === "complete" && (
        <CompleteRegistration rollNumber={rollNumber} email={email} onSubmit={handleCompleteSubmit} />
      )}  
      {step === "acknowledgement" && <Acknowledgement />}
    </div>
  );
};

export default StudentRegister;