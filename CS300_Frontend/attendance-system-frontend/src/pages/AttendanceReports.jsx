import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './AttendanceReports.css';

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const AttendanceReports = () => {
  const { token } = useAuth();
  const [attendance, setAttendance] = useState([]);
  const [filteredAttendance, setFilteredAttendance] = useState([]);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch attendance records when the component mounts
  useEffect(() => {
    const fetchAttendance = async () => {
      try {
        const response = await axios.get(
          `${BASE_URL}/students/my_attendance`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        setAttendance(response.data.attendance_records);
        setFilteredAttendance(response.data.attendance_records);
        setLoading(false);
      } catch (err) {
        setError("Failed to load attendance records");
        setLoading(false);
      }
    };
    fetchAttendance();
  }, [token]);

  // Update filtered attendance when the filter changes
  useEffect(() => {
    if (filter === "all") {
      setFilteredAttendance(attendance);
    } else if (filter === "present") {
      setFilteredAttendance(
        attendance.filter((record) => !record.is_absent)
      );
    } else if (filter === "absent") {
      setFilteredAttendance(
        attendance.filter((record) => record.is_absent)
      );
    }
  }, [filter, attendance]);

  // Handle filter change
  const handleFilterChange = (e) => {
    setFilter(e.target.value);
  };

  // Render attendance status with color coding
  const renderAttendanceStatus = (isAbsent) => {
    const statusClass = isAbsent ? "student-status-absent" : "student-status-present";
    return (
      <span className={`student-attendance-status ${statusClass}`}>
        {isAbsent ? "Absent" : "Present"}
      </span>
    );
  };

  // Conditional rendering for loading and error states
  if (loading) return <div className="student-loading">Loading...</div>;
  if (error) return <div className="student-error-message">{error}</div>;

  return (
    <div className="student-attendance-reports-container">
      <h1 className="student-page-title">Attendance Reports</h1>

      {/* Filter Dropdown */}
      <div className="student-filter-section">
        <label htmlFor="filter">Filter by: </label>
        <select
          id="filter"
          value={filter}
          onChange={handleFilterChange}
          className="form-control"
        >
          <option value="all">All Records</option>
          <option value="present">Present Classes</option>
          <option value="absent">Absent Classes</option>
        </select>
      </div>

      {/* Attendance Table */}
      <div className="student-attendance-table-container">
        {filteredAttendance.length === 0 ? (
          <div className="student-no-records">No attendance records found.</div>
        ) : (
          <table className="student-attendance-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Course Code</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredAttendance.map((record) => (
                <tr
                  key={record.report_id}
                  className={record.is_absent ? "student-absent-row" : "student-present-row"}
                >
                  <td>
                    {new Date(record.class_time).toLocaleDateString()}
                  </td>
                  <td>{record.course_code}</td>
                  <td>
                    {renderAttendanceStatus(record.is_absent)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Summary Statistics */}
      <div className="student-attendance-summary">
        <div className="student-summary-stats">
          <div className="student-stat-item">
            <span className="student-stat-label">Total Classes</span>
            <span className="student-stat-value">{attendance.length}</span>
          </div>
          <div className="student-stat-item">
            <span className="student-stat-label">Present</span>
            <span className="student-stat-value student-present-count">
              {attendance.filter(record => !record.is_absent).length}
            </span>
          </div>
          <div className="student-stat-item">
            <span className="student-stat-label">Absent</span>
            <span className="student-stat-value student-absent-count">
              {attendance.filter(record => record.is_absent).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AttendanceReports;