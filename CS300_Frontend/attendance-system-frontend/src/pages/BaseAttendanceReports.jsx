// src/components/BaseAttendanceReports.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './EmployeeAttendanceReports.css';

const BaseAttendanceReports = ({ 
  apiEndpoints, 
  pageTitle = "Attendance Reports",
  role = "student" 
}) => {
  const { token } = useAuth();
  const [sections, setSections] = useState([]);
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [filteredRecords, setFilteredRecords] = useState([]);
  const [selectedSection, setSelectedSection] = useState("");
  const [courseCodeFilter, setCourseCodeFilter] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch sections and attendance records on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch assigned sections
        const sectionsResponse = await axios.get(
          apiEndpoints.assignedSections,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setSections(sectionsResponse.data.schedules || []);

        // Fetch all attendance records
        const attendanceResponse = await axios.get(
          apiEndpoints.attendanceRecords,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setAttendanceRecords(attendanceResponse.data.records || []);
        setFilteredRecords(attendanceResponse.data.records || []);
        setLoading(false);
      } catch (err) {
        setError("Failed to load data");
        setLoading(false);
      }
    };
    fetchData();
  }, [token]);

  // Filter records based on section, course code, and date range
  useEffect(() => {
    let filtered = attendanceRecords;

    if (selectedSection) {
      filtered = filtered.filter((record) => record.section_id === parseInt(selectedSection));
    }
    if (courseCodeFilter) {
      filtered = filtered.filter((record) =>
        record.course_code.toLowerCase().includes(courseCodeFilter.toLowerCase())
      );
    }
    if (startDate) {
      filtered = filtered.filter(
        (record) => new Date(record.class_time) >= new Date(startDate)
      );
    }
    if (endDate) {
      filtered = filtered.filter(
        (record) => new Date(record.class_time) <= new Date(endDate)
      );
    }

    setFilteredRecords(filtered);
  }, [selectedSection, courseCodeFilter, startDate, endDate, attendanceRecords]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="attendance-reports-container">
      <h1 className="page-title">{pageTitle}</h1>

      {/* Filters */}
      <div className="filters-container">
        <div className="filter-group">
          <label>Section</label>
          <select
            value={selectedSection}
            onChange={(e) => setSelectedSection(e.target.value)}
            className="form-control"
          >
            <option value="">All Sections</option>
            {sections.map((section) => (
              <option key={section.section_id} value={section.section_id}>
                {section.section_name} ({section.course_code})
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Course Code</label>
          <input
            type="text"
            value={courseCodeFilter}
            onChange={(e) => setCourseCodeFilter(e.target.value)}
            placeholder="e.g., CS101"
            className="form-control"
          />
        </div>

        <div className="filter-group">
          <label>Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="form-control"
          />
        </div>

        <div className="filter-group">
          <label>End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="form-control"
          />
        </div>
      </div>

      {/* Attendance Table */}
      {filteredRecords.length === 0 ? (
        <div className="no-records">No attendance records found.</div>
      ) : (
        <div className="table-container">
          <table className="attendance-table">
            <thead>
              <tr>
                <th>Roll Number</th>
                <th>Name</th>
                <th>Date</th>
                <th>Course Code</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((record) => (
                <tr 
                  key={record.report_id} 
                  className={record.is_absent ? 'absent-row' : 'present-row'}
                >
                  <td>{record.roll_number}</td>
                  <td>{record.student_name}</td>
                  <td>{new Date(record.class_time).toLocaleDateString()}</td>
                  <td>{record.course_code}</td>
                  <td>
                    <span className={`status-badge ${record.is_absent ? 'absent' : 'present'}`}>
                      {record.is_absent ? "Absent" : "Present"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default BaseAttendanceReports;