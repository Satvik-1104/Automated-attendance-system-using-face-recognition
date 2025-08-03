import React, { useState } from "react";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhdCompleteRegistration = ({ phdId, email, onSubmit }) => {
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [department, setDepartment] = useState(""); // New state for department
  const [teachingAssignments, setTeachingAssignments] = useState([
    { batch: "", course_codes: "", sections: "" },
  ]);
  const [loading, setLoading] = useState(false);

  const addAssignment = () => {
    setTeachingAssignments([...teachingAssignments, { batch: "", course_codes: "", sections: "" }]);
  };

  const handleAssignmentChange = (index, field, value) => {
    const newAssignments = [...teachingAssignments];
    newAssignments[index][field] = value;
    setTeachingAssignments(newAssignments);
  };

  const handleRegister = async () => {
    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }
    if (teachingAssignments.some((a) => !a.batch || !a.course_codes || !a.sections)) {
      alert("Please fill in all teaching assignment fields");
      return;
    }
    if (!department) {
      alert("Please enter your department");
      return;
    }
    setLoading(true);
    try {
      const assignments = teachingAssignments.map((a) => ({
        batch: parseInt(a.batch),
        course_code: a.course_codes.split(",").map((c) => c.trim()),
        sections: a.sections.split(",").map((s) => s.trim()),
      }));
      const response = await axios.post(`${BASE_URL}/phd/register`, {
        phd_id: phdId,
        name: name,
        email: email,
        username: username,
        password: password,
        department: department, // Add department to payload
        teaching_assignments: assignments,
      });
      if (response.data.message) {
        onSubmit();
      }
    } catch (error) {
      alert("Registration failed: " + (error.response?.data?.detail || "Please try again"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Complete PhD Registration</h2>
      <input
        style={styles.input}
        placeholder="Full Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <input
        style={styles.input}
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
      <input
        style={styles.input}
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <input
        style={styles.input}
        type="password"
        placeholder="Confirm Password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
      />
      <input // New input for department
        style={styles.input}
        placeholder="Department"
        value={department}
        onChange={(e) => setDepartment(e.target.value)}
      />
      <h3 style={styles.subtitle}>Teaching Assignments</h3>
      {teachingAssignments.map((assignment, index) => (
        <div key={index} style={styles.assignmentContainer}>
          <input
            style={styles.input}
            placeholder="Batch (e.g., 2022)"
            value={assignment.batch}
            onChange={(e) => handleAssignmentChange(index, "batch", e.target.value)}
          />
          <input
            style={styles.input}
            placeholder="Course Codes (comma-separated)"
            value={assignment.course_codes}
            onChange={(e) => handleAssignmentChange(index, "course_codes", e.target.value)}
          />
          <input
            style={styles.input}
            placeholder="Sections (comma-separated)"
            value={assignment.sections}
            onChange={(e) => handleAssignmentChange(index, "sections", e.target.value)}
          />
        </div>
      ))}
      <button onClick={addAssignment} style={styles.addButton}>
        Add Another Assignment
      </button>
      <button onClick={handleRegister} disabled={loading} style={styles.button}>
        {loading ? "Registering..." : "Complete Registration"}
      </button>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: "#ffffff",
    borderRadius: "15px",
    padding: "20px",
    width: "100%",
    maxWidth: "600px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
  },
  title: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333",
    textAlign: "center",
    marginBottom: "20px",
  },
  subtitle: {
    fontSize: "18px",
    color: "#777",
    marginBottom: "15px",
  },
  input: {
    width: "100%",
    padding: "12px",
    marginBottom: "15px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
    fontSize: "16px",
    boxSizing: "border-box",
  },
  assignmentContainer: {
    marginBottom: "20px",
  },
  addButton: {
    backgroundColor: "#5d6497",
    color: "#fff",
    padding: "10px",
    borderRadius: "10px",
    textAlign: "center",
    fontSize: "14px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
    marginBottom: "20px",
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

export default PhdCompleteRegistration;