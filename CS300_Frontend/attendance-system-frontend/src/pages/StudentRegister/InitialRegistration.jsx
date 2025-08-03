// src/pages/StudentRegister/InitialRegistration.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom"; // Added import
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const InitialRegistration = ({ onSubmit }) => {
  const [rollNumber, setRollNumber] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  const validateEmail = () => {
    console.log("Validating email:", email);
    if (!email.trim()) return false;
    const emailPattern = /^[a-zA-Z]+\.[a-zA-Z]+(\d{2})b@iiitg\.ac\.in$/;
    const match = email.match(emailPattern);
    if (!match) {
      console.log("Email validation failed: Pattern mismatch");
      return false;
    }
    const emailYear = match[1];
    const rollYear = rollNumber.substring(0, 2);
    console.log("Extracted email year:", emailYear, "Roll year:", rollYear);
    return emailYear === rollYear;
  };

  const handleRequestOTP = async () => {
    console.log("Requesting OTP with Roll Number:", rollNumber, "Email:", email);
    if (!rollNumber.trim() || isNaN(parseInt(rollNumber))) {
      alert("Please enter a valid roll number");
      console.log("Invalid Roll Number");
      return;
    }
    if (!validateEmail()) {
      alert("Please enter a valid campus email that matches your roll number year");
      console.log("Invalid Email");
      return;
    }

    setLoading(true);
    try {
      console.log("Sending OTP request to backend...");
      const response = await axios.post(`${BASE_URL}/students/register/request_otp`, {
        roll_number: parseInt(rollNumber),
        email: email,
      });
      console.log("Response from server:", response.data);
      if (response.data.message) {
        console.log("OTP request successful, proceeding to next step");
        onSubmit(rollNumber, email);
      }
    } catch (error) {
      console.error("Error requesting OTP:", error.response?.data?.detail || error.message);
      alert("Error requesting OTP: " + (error.response?.data?.detail || "Please try again"));
    } finally {
      setLoading(false);
      console.log("OTP request process completed");
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Student Registration</h2>
      <p style={styles.subtitle}>Enter your credentials to get started</p>
      <div style={styles.inputContainer}>
        <input
          type="text"
          placeholder="Roll Number"
          value={rollNumber}
          onChange={(e) => {
            console.log("Roll number changed to:", e.target.value);
            setRollNumber(e.target.value);
          }}
          style={styles.input}
          maxLength={8}
        />
      </div>
      <div style={styles.inputContainer}>
        <input
          type="email"
          placeholder="Campus Email"
          value={email}
          onChange={(e) => {
            console.log("Email changed to:", e.target.value);
            setEmail(e.target.value);
          }}
          style={styles.input}
        />
      </div>
      <p style={styles.infoText}>
        We'll send a one-time password (OTP) to your campus email address to verify your identity.<br/>
        This whole process may not work with our campus wi-fi because, you know why.
      </p>
      <button onClick={handleRequestOTP} disabled={loading} style={styles.button}>
        {loading ? "Loading..." : "Request OTP"}
      </button>
      <p style={styles.loginText}>
        Already have an account? <Link to="/login" style={styles.loginLink}>Login here</Link>
      </p>
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
  inputContainer: {
    marginBottom: "15px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
    padding: "5px 10px",
  },
  input: {
    width: "100%",
    height: "40px",
    border: "none",
    fontSize: "16px",
    outline: "none",
  },
  infoText: {
    fontSize: "12px",
    color: "#777",
    marginBottom: "20px",
    textAlign: "center",
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
  },
  loginText: {
    marginTop: "20px",
    textAlign: "center",
    fontSize: "14px",
    color: "#777",
  },
  loginLink: {
    color: "#4a56e2",
    textDecoration: "none",
    fontWeight: "bold",
  },
};

export default InitialRegistration;