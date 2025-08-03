// src/pages/ForgotPassword/index.jsx
import { useState } from "react";
import RequestPasswordReset from "./RequestPasswordReset";
import VerifyOTP from "./VerifyOTP";
import ResetPassword from "./ResetPassword";

const ForgotPassword = () => {
  const [step, setStep] = useState(1);
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");

  const handleRequestSubmit = (submittedEmail) => {
    setEmail(submittedEmail);
    setStep(2);
  };

  const handleOTPSubmit = (submittedOtp) => {
    setOtp(submittedOtp);
    setStep(3);
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return <RequestPasswordReset onSubmit={handleRequestSubmit} />;
      case 2:
        return <VerifyOTP email={email} onSubmit={handleOTPSubmit} />;
      case 3:
        return <ResetPassword email={email} otp={otp} />;
      default:
        return <RequestPasswordReset onSubmit={handleRequestSubmit} />;
    }
  };

  return (
    <div style={styles.wrapper}>
      {renderStep()}
    </div>
  );
};

const styles = {
  wrapper: {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f9fa",
  },
};

export default ForgotPassword;