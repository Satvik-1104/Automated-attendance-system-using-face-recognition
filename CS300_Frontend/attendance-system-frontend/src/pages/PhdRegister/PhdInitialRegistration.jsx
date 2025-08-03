// src/pages/PhdRegister/PhdInitialRegistration.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhdInitialRegistration = ({ onSubmit }) => {
  const [phdId, setPhdId] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRequestOTP = async () => {
    if (!phdId.trim() || !email.trim()) {
      alert("Please enter PhD ID and email");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(`${BASE_URL}/phd/register/request_otp`, {
        phd_id: phdId,
        email: email,
      });
      if (response.data.message) {
        onSubmit(phdId, email);
      }
    } catch (error) {
      alert("Error requesting OTP: " + (error.response?.data?.detail || "Please try again"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>PhD Student Registration</h2>
      <p style={styles.subtitle}>Enter your credentials to get started</p>
      <div style={styles.inputContainer}>
        <input
          type="text"
          placeholder="PhD ID"
          value={phdId}
          onChange={(e) => setPhdId(e.target.value)}
          style={styles.input}
        />
      </div>
      <div style={styles.inputContainer}>
        <input
          type="email"
          placeholder="Campus Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={styles.input}
        />
      </div>
      <p style={styles.infoText}>
        We'll send a one-time password (OTP) to your campus email address to verify your identity.
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

export default PhdInitialRegistration;