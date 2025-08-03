// src/pages/ForgotPassword/ResetPassword.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const ResetPassword = ({ email, otp }) => {
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const validatePassword = () => {
    if (newPassword.length < 8) {
      setMessage("Password must be at least 8 characters long");
      return false;
    }
    if (newPassword !== confirmPassword) {
      setMessage("Passwords do not match");
      return false;
    }
    return true;
  };

  const handleResetPassword = async () => {
    if (!validatePassword()) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${BASE_URL}/auth/reset-password`, {
        email: email,
        otp: otp,
        new_password: newPassword
      });
      
      if (response.data.message) {
        setMessage("Password reset successfully! Redirecting to login...");
        setTimeout(() => {
          navigate("/login");
        }, 2000);
      }
    } catch (error) {
      setMessage(error.response?.data?.detail || "Failed to reset password. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Reset Password</h2>
      <p style={styles.subtitle}>Create a new password for your account</p>
      
      <div style={styles.inputContainer}>
        <input
          type="password"
          placeholder="New Password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          style={styles.input}
        />
      </div>
      
      <div style={styles.inputContainer}>
        <input
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          style={styles.input}
        />
      </div>
      
      <p style={styles.infoText}>
        Password must be at least 8 characters long
      </p>
      
      {message && <p style={message.includes("successfully") ? styles.successMessage : styles.errorMessage}>
        {message}
      </p>}
      
      <button 
        onClick={handleResetPassword} 
        disabled={loading} 
        style={styles.button}
      >
        {loading ? "Resetting..." : "Reset Password"}
      </button>
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
  errorMessage: {
    fontSize: "14px",
    color: "#d32f2f",
    marginBottom: "15px",
    textAlign: "center",
  },
  successMessage: {
    fontSize: "14px",
    color: "#43a047",
    marginBottom: "15px",
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
    width: "100%",
  },
};

export default ResetPassword;