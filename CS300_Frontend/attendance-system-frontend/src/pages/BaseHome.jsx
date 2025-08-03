// src/components/BaseHome.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './EmployeeHome.css';

const BaseHome = ({ 
  apiEndpoints, 
  pageTitle = "Dashboard Home",
  role = "student" 
}) => {
  const { token } = useAuth();
  const [stats, setStats] = useState({
    scheduledClasses: 0,
    pendingCorrections: 0,
    attendanceMarked: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const scheduleResponse = await axios.get(
          apiEndpoints.classSchedules,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const scheduledClasses = scheduleResponse.data.total || 0;

        const correctionsResponse = await axios.get(
          apiEndpoints.pendingCorrections,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const pendingCorrections = correctionsResponse.data.pending_corrections.length || 0;

        const attendanceResponse = await axios.get(
          apiEndpoints.attendanceSummary,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const attendanceMarked = attendanceResponse.data.total_records || 0;

        setStats({
          scheduledClasses,
          pendingCorrections,
          attendanceMarked,
        });
        setLoading(false);
      } catch (err) {
        setError("Failed to load summary data");
        setLoading(false);
      }
    };
    fetchSummary();
  }, [token]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
  <div className="employee-home-container">
    <h1 className="employee-page-title">{pageTitle}</h1>
    <div className="employee-stats-grid">
      <div className="employee-stat-card scheduled-classes">
        <div className="status-indicator"></div>
        <h3>Scheduled Classes</h3>
        <p className="employee-stat-value">{stats.scheduledClasses}</p>
      </div>
      <div className="employee-stat-card pending-corrections">
        <div className="status-indicator"></div>
        <h3>Pending Corrections</h3>
        <p className="employee-stat-value">{stats.pendingCorrections}</p>
      </div>
      <div className="employee-stat-card attendance-marked">
        <div className="status-indicator"></div>
        <h3>Attendance Marked</h3>
        <p className="employee-stat-value">{stats.attendanceMarked}</p>
      </div>
    </div>
  </div>
);
};

export default BaseHome;