import { useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const RequestPasswordReset = ({ onSubmit }) => {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleRequestReset = async () => {
    if (!email.trim()) {
      setMessage("Please enter your email address");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${BASE_URL}/auth/forgot-password`, null, {
        params: { email }
      });
      
      if (response.data.message) {
        setMessage("If the email exists, an OTP has been sent to your inbox");
        // Move to next step after 1.5 seconds
        setTimeout(() => {
          onSubmit(email);
        }, 1500);
      }
    } catch (error) {
      setMessage(error.response?.data?.detail || "Failed to request password reset. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Forgot Password</h2>
      <p style={styles.subtitle}>Enter your email to receive a reset code</p>
      
      <div style={styles.inputContainer}>
        <input
          type="email"
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={styles.input}
        />
      </div>
      
      {message && <p style={styles.message}>{message}</p>}
      
      <button 
        onClick={handleRequestReset} 
        disabled={loading} 
        style={styles.button}
      >
        {loading ? "Sending..." : "Request Reset Code"}
      </button>
      
      <p style={styles.loginText}>
        Remember your password? <Link to="/login" style={styles.link}>Login here</Link>
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
  message: {
    fontSize: "14px",
    color: "#4a56e2",
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
  loginText: {
    marginTop: "20px",
    textAlign: "center",
    fontSize: "14px",
    color: "#777",
  },
  link: {
    color: "#4a56e2",
    textDecoration: "none",
    fontWeight: "bold",
  },
};

export default RequestPasswordReset;