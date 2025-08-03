// src/components/BaseGenerateReport.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import Papa from "papaparse";
import { useAuth } from "../context/AuthContext";
import './GenerateReport.css';

const BaseGenerateReport = ({ 
  apiEndpoints,
  pageTitle = "Generate Attendance Report",
  role = "faculty"
}) => {
  const { token } = useAuth();
  const [assignedSectionCourses, setAssignedSectionCourses] = useState([]);
  const [selectedSectionCourse, setSelectedSectionCourse] = useState(null);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Fetch assigned sections and courses on mount
  useEffect(() => {
    const fetchAssigned = async () => {
      try {
        const response = await axios.get(apiEndpoints.assignedSections, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setAssignedSectionCourses(response.data.schedules || []);
      } catch (err) {
        setError("Failed to load assigned sections");
      }
    };
    fetchAssigned();
  }, [token]);

  // Handle report generation
  const handleGenerateReport = async () => {
    if (!selectedSectionCourse) {
      setError("Please select a section and course");
      return;
    }
    setLoading(true);
    setError("");
    const { section_id, course_code } = selectedSectionCourse;
    const params = new URLSearchParams();
    if (startDate)
      params.append("start_date", startDate.toISOString().split("T")[0]);
    if (endDate)
      params.append("end_date", endDate.toISOString().split("T")[0]);
    params.append("course_code", course_code);

    try {
      const response = await axios.get(
        `${apiEndpoints.generateReport}/${section_id}?${params.toString()}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setReportData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to generate report");
    } finally {
      setLoading(false);
    }
  };

  // Handle CSV export for the student summary table
  const handleExportCSV = () => {
    if (!reportData) return;
    const csvData = reportData.student_summaries.map((summary) => ({
      "Roll Number": summary.roll_number,
      "Student Name": summary.student_name,
      "Total Classes": summary.total_classes,
      "Present": summary.present,
      "Absent": summary.absent,
      "Attendance %": summary.attendance_percentage.toFixed(2),
    }));
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute(
      "download",
      `attendance_report_${reportData.section_id}_${reportData.course_code}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="generate-report-container">
      <h1 className="page-title">{pageTitle}</h1>
      
      <div className="report-filters">
        <div className="filter-group">
          <label>Select Section and Course</label>
          <select
            value={
              selectedSectionCourse
                ? `${selectedSectionCourse.section_id}-${selectedSectionCourse.course_code}`
                : ""
            }
            onChange={(e) => {
              const [sectionId, courseCode] = e.target.value.split("-");
              setSelectedSectionCourse({
                section_id: sectionId,
                course_code: courseCode,
              });
            }}
            className="form-control"
          >
            <option value="">-- Select --</option>
            {assignedSectionCourses.map((sc) => (
              <option
                key={`${sc.section_id}-${sc.course_code}`}
                value={`${sc.section_id}-${sc.course_code}`}
              >
                Section {sc.section_name} - Course {sc.course_code}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Start Date</label>
          <DatePicker 
            selected={startDate} 
            onChange={(date) => setStartDate(date)}
            className="form-control"
            placeholderText="Select start date"
          />
        </div>

        <div className="filter-group">
          <label>End Date</label>
          <DatePicker 
            selected={endDate} 
            onChange={(date) => setEndDate(date)}
            className="form-control"
            placeholderText="Select end date"
          />
        </div>

        <button
          onClick={handleGenerateReport}
          className="btn btn-primary generate-btn"
          disabled={loading}
        >
          {loading ? "Generating..." : "Generate Report"}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {reportData && (
        <div className="report-details">
          <div className="report-header">
            <h2>
              Report for Section {reportData.section_id} - Course{" "}
              {reportData.course_code}
            </h2>
            <p>Total Unique Classes: {reportData.total_classes}</p>
          </div>

          <div className="report-section">
            <h3>Student Attendance Summary</h3>
            <div className="table-container">
            <table className="table attendance-summary">
              <thead>
                <tr>
                  <th>Roll Number</th>
                  <th>Student Name</th>
                  <th>Total Classes</th>
                  <th>Present</th>
                  <th>Absent</th>
                  <th>Attendance %</th>
                </tr>
              </thead>
              <tbody>
                {reportData.student_summaries.map((summary, index) => (
                  <tr
                    key={index}
                    className={
                      summary.attendance_percentage < 75 
                        ? "low-attendance-row" 
                        : ""
                    }
                  >
                    <td>{summary.roll_number}</td>
                    <td>{summary.student_name}</td>
                    <td>{summary.total_classes}</td>
                    <td>{summary.present}</td>
                    <td>{summary.absent}</td>
                    <td>{summary.attendance_percentage.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>

            <button 
              onClick={handleExportCSV} 
              className="btn btn-secondary export-btn"
            >
              Export Summary to CSV
            </button>
          </div>

          <div className="report-section">
            <h3>Detailed Attendance Records</h3>
            <div className="table-container">
            <table className="table detailed-records">
              <thead>
                <tr>
                  <th>Roll Number</th>
                  <th>Class Time</th>
                  <th>Present</th>
                  <th>Marked By</th>
                </tr>
              </thead>
              <tbody>
                {reportData.attendance_details.map((record, index) => (
                  <tr key={index}>
                    <td>{record.roll_number}</td>
                    <td>{new Date(record.class_time).toLocaleString()}</td>
                    <td>{record.is_absent ? "No" : "Yes"}</td>
                    <td>{record.marked_by_faculty || record.marked_by_phd || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BaseGenerateReport;