// src/pages/StudentRegister/Acknowledgement.jsx
import React from "react";
import { useNavigate } from "react-router-dom"; // Added import

const Acknowledgement = () => {
  const navigate = useNavigate(); // Added hook

  const handleLogin = () => {
    navigate("/login"); // Updated navigation
  };

  return (
    <div style={styles.card}>
      <div style={styles.checkIcon}>✔</div>
      <h2 style={styles.title}>Registration Complete!</h2>
      <p style={styles.status}>
        Status: <span style={styles.statusBadge}>Under Review</span>
      </p>
      <p style={styles.message}>
        Thank you for completing your registration. Your profile information and face data have been successfully submitted and are currently under review.
      </p>
      <p style={styles.instruction}>
        You will recieve an email regarding your registration status.<br/>
        If you have felt inconvinience during the whole process, we apologize but we don't give a damn, live with it.
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
  status: {
    fontSize: "18px",
    color: "#666",
    marginBottom: "25px",
  },
  statusBadge: {
    backgroundColor: "#FFF9C4",
    padding: "8px 15px",
    borderRadius: "20px",
    border: "1px solid #FBC02D",
    color: "#F57F17",
    fontWeight: "bold",
  },
  message: {
    fontSize: "16px",
    color: "#555",
    marginBottom: "20px",
    lineHeight: "24px",
  },
  instruction: {
    fontSize: "15px",
    color: "#666",
    marginBottom: "25px",
    fontStyle: "italic",
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

export default Acknowledgement;