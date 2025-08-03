// src/pages/Login.jsx
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom"; // Add useNavigate
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import { jwtDecode } from "jwt-decode";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

console.log("BASE_URL", BASE_URL); // Log the BASE_URL to check if it's correct
const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const { login } = useAuth();
  const navigate = useNavigate(); // For redirecting after login

  // Determine the registration path based on the mode
  const getRegisterPath = () => {
    const mode = import.meta.env.MODE;
    switch (mode) {
      case "faculty":
        return "/register/faculty";
      case "phd":
        return "/register/phd";
      case "student":
      default:
        return "/register/student";
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);

      const response = await axios.post(
        `${BASE_URL}/auth/token`,
        formData,
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
        }
      );
      const { access_token } = response.data;
      const decodedToken = jwtDecode(access_token);
      const role = decodedToken.role;
      login(access_token, role);

      // Redirect based on role after successful login
      switch (role) {
        case "student":
          navigate("/student/dashboard");
          break;
        case "faculty":
          navigate("/faculty/dashboard");
          break;
        case "phd_student":
          navigate("/phd/dashboard");
          break;
        default:
          setError("Unknown role");
      }
    } catch (err) {
      setError("Invalid credentials");
    }
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.card}>
        <h1 style={styles.title}>Login</h1>
        <form onSubmit={handleSubmit}>
          <div style={styles.inputContainer}>
            <label style={styles.label}>Username:</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          <div style={styles.inputContainer}>
            <label style={styles.label}>Password:</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={styles.input}
            />
          </div>
          <div style={styles.forgotPasswordContainer}>
            <Link to="/forgot-password" style={styles.forgotPasswordLink}>
              Forgot Password?
            </Link>
          </div>
          {error && <p style={styles.error}>{error}</p>}
          <button type="submit" style={styles.button}>Login</button>
        </form>
        <p style={styles.registerText}>
          Don't have an account?{" "}
          <Link to={getRegisterPath()} style={styles.registerLink}>
            Register here
          </Link>
        </p>
      </div>
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
    marginBottom: "20px",
  },
  inputContainer: {
    marginBottom: "15px",
  },
  label: {
    fontSize: "14px",
    color: "#555",
    marginBottom: "5px",
    display: "block",
  },
  input: {
    width: "95%",
    padding: "10px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
    fontSize: "16px",
    outline: "none",
  },
  forgotPasswordContainer: {
    textAlign: "right",
    marginBottom: "20px",
  },
  forgotPasswordLink: {
    fontSize: "14px",
    color: "#4a56e2",
    textDecoration: "none",
  },
  error: {
    color: "red",
    textAlign: "center",
    marginBottom: "15px",
    fontSize: "14px",
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
  registerText: {
    marginTop: "20px",
    textAlign: "center",
    fontSize: "14px",
    color: "#777",
  },
  registerLink: {
    color: "#4a56e2",
    textDecoration: "none",
    fontWeight: "bold",
  },
};

export default Login;