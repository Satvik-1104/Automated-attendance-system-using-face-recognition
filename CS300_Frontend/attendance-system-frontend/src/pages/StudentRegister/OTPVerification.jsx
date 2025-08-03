// src/pages/StudentRegister/OTPVerification.jsx
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const OTPVerification = ({ rollNumber, email, onSubmit }) => {
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [loading, setLoading] = useState(false);
  const [resendDisabled, setResendDisabled] = useState(true);
  const [timer, setTimer] = useState(30);
  const inputRefs = useRef([]);

  useEffect(() => {
    inputRefs.current[0]?.focus();
    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setResendDisabled(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleOtpChange = (text, index) => {
    if (!/^\d*$/.test(text)) return;
    const newOtp = [...otp];
    newOtp[index] = text;
    setOtp(newOtp);
    if (text && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleVerifyOTP = async () => {
    const otpValue = otp.join("");
    if (otpValue.length !== 6) {
      alert("Please enter the complete 6-digit OTP");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(`${BASE_URL}/students/register/verify_otp`, {
        email: email,
        otp: otpValue,
      });
      if (response.data.message) {
        onSubmit();
      }
    } catch (error) {
      alert("Invalid OTP: " + (error.response?.data?.detail || "Please try again"));
    } finally {
      setLoading(false);
    }
  };

  const handleResendOTP = async () => {
    setResendDisabled(true);
    setTimer(30);
    try {
      await axios.post(`${BASE_URL}/students/register/request_otp`, {
        roll_number: rollNumber,
        email: email,
      });
      alert("New OTP has been sent to your email");
      const interval = setInterval(() => {
        setTimer((prev) => {
          if (prev <= 1) {
            clearInterval(interval);
            setResendDisabled(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (error) {
      alert("Failed to resend OTP: " + (error.response?.data?.detail || "Please try again"));
      setResendDisabled(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Verify OTP</h2>
      <p style={styles.subtitle}>Enter the 6-digit code sent to your email</p>
      <p style={styles.emailText}>{email}</p>
      <div style={styles.otpContainer}>
        {otp.map((digit, index) => (
          <input
            key={index}
            type="text"
            value={digit}
            onChange={(e) => handleOtpChange(e.target.value, index)}
            maxLength={1}
            ref={(el) => (inputRefs.current[index] = el)}
            style={styles.otpInput}
          />
        ))}
      </div>
      <button onClick={handleVerifyOTP} disabled={loading} style={styles.button}>
        {loading ? "Verifying..." : "Verify OTP"}
      </button>
      <div style={styles.resendContainer}>
        <span style={styles.resendText}>Didn't receive the code? </span>
        {resendDisabled ? (
          <span style={styles.timerText}>Resend in {timer}s</span>
        ) : (
          <button onClick={handleResendOTP} style={styles.resendLink}>
            Resend OTP
          </button>
        )}
      </div>
    </div>
  );
};

const styles = {
  card: {
    backgroundColor: "#ffffff",
    borderRadius: "15px",
    padding: "20px",
    width: "100%",
    maxWidth: "400px",
    boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
  },
  title: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333",
    textAlign: "center",
    marginBottom: "10px",
  },
  subtitle: {
    fontSize: "14px",
    color: "#777",
    textAlign: "center",
    marginBottom: "20px",
  },
  emailText: {
    fontSize: "16px",
    color: "#4a56e2",
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: "20px",
  },
  otpContainer: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: "20px",
  },
  otpInput: {
    width: "40px",
    height: "40px",
    textAlign: "center",
    fontSize: "20px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
  },
  button: {
    backgroundColor: "#4a56e2",
    color: "#fff",
    padding: "15px",
    borderRadius: "10px",
    textAlign: "center",
    fontSize: "16px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
    width: "100%",
  },
  resendContainer: {
    marginTop: "20px",
    textAlign: "center",
  },
  resendText: {
    color: "#777",
    fontSize: "14px",
  },
  timerText: {
    color: "#999",
    fontSize: "14px",
  },
  resendLink: {
    color: "#4a56e2",
    fontSize: "14px",
    fontWeight: "bold",
    background: "none",
    border: "none",
    cursor: "pointer",
  },
};

export default OTPVerification;