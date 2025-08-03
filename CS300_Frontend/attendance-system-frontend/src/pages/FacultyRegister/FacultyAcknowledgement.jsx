// src/pages/FacultyRegister/FacultyAcknowledgement.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

const FacultyAcknowledgement = () => {
  const navigate = useNavigate();

  const handleLogin = () => {
    navigate("/login");
  };

  return (
    <div style={styles.card}>
      <div style={styles.checkIcon}>✔</div>
      <h2 style={styles.title}>Registration Complete!</h2>
      <p style={styles.message}>
        Thank you for completing your registration. Your account has been successfully created.
      </p>
      <button onClick={handleLogin} style={styles.button}>
        Go to Login
      </button>
    </div>
  );
};

const styles = {
  card: {
    backgroundColor: "#ffffff",
    borderRadius: "20px",
    padding: "30px",
    width: "100%",
    maxWidth: "500px",
    boxShadow: "0 5px 15px rgba(0,0,0,0.1)",
    textAlign: "center",
  },
  checkIcon: {
    fontSize: "80px",
    color: "#4CAF50",
    marginBottom: "20px",
  },
  title: {
    fontSize: "28px",
    fontWeight: "bold",
    color: "#333",
    marginBottom: "20px",
  },
  message: {
    fontSize: "16px",
    color: "#555",
    marginBottom: "25px",
    lineHeight: "24px",
  },
  button: {
    backgroundColor: "#4a56e2",
    color: "#fff",
    padding: "14px 30px",
    borderRadius: "30px",
    fontSize: "16px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
  },
};

export default FacultyAcknowledgement;