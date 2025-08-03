// src/components/BaseMarkAttendance.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './MarkAttendance.css';

const BaseMarkAttendance = ({ 
  apiEndpoints, 
  pageTitle = "Mark Attendance",
  role = "faculty" 
}) => {
  const { token } = useAuth();
  const [sections, setSections] = useState([]);
  const [selectedSection, setSelectedSection] = useState("");
  const [courseCode, setCourseCode] = useState("");
  const [classTime, setClassTime] = useState("");
  const [students, setStudents] = useState([]);
  const [filteredStudents, setFilteredStudents] = useState([]);
  const [rollFilter, setRollFilter] = useState("");
  const [attendance, setAttendance] = useState({});
  const [updateStatus, setUpdateStatus] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Fetch sections assigned to the user
  useEffect(() => {
    const fetchSections = async () => {
      try {
        const response = await axios.get(apiEndpoints.assignedSections, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSections(response.data.schedules || []);
        setLoading(false);
      } catch (err) {
        setError("Failed to load sections");
        setLoading(false);
      }
    };
    fetchSections();
  }, [token]);

  // Fetch students for the selected section
  const fetchStudents = async () => {
    if (!selectedSection || !courseCode || !classTime) return;
    try {
      setLoading(true);
      const response = await axios.get(
        `${apiEndpoints.students}/${selectedSection}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const studentList = response.data.students || [];
      setStudents(studentList);
      setFilteredStudents(studentList);
      
      // Initialize attendance and updateStatus states
      const initialAttendance = studentList.reduce((acc, student) => {
        acc[student.roll_number] = false;
        return acc;
      }, {});
      const initialUpdateStatus = studentList.reduce((acc, student) => {
        acc[student.roll_number] = false;
        return acc;
      }, {});
      
      setAttendance(initialAttendance);
      setUpdateStatus(initialUpdateStatus);
      setLoading(false);
    } catch (err) {
      setError("Failed to load students");
      setLoading(false);
    }
  };

  // Filter students by roll number
  const handleFilter = (e) => {
    const filterValue = e.target.value.toLowerCase();
    setRollFilter(filterValue);
    const filtered = students.filter((student) =>
      student.roll_number.toString().toLowerCase().includes(filterValue)
    );
    setFilteredStudents(filtered);
  };

  // Handle attendance toggle
  const toggleAttendance = (rollNumber) => {
    setAttendance((prev) => ({
      ...prev,
      [rollNumber]: !prev[rollNumber],
    }));
  };

  // Handle update status toggle
  const toggleUpdateStatus = (rollNumber) => {
    setUpdateStatus((prev) => ({
      ...prev,
      [rollNumber]: !prev[rollNumber],
    }));
  };

  // Submit attendance
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedSection || !courseCode || !classTime || students.length === 0) {
      setError("Please select a section, course code, class time, and fetch students");
      return;
    }

    try {
      const attendancePromises = filteredStudents
        .filter((student) => updateStatus[student.roll_number])
        .map((student) =>
          axios.post(
            apiEndpoints.markAttendance,
            {
              roll_number: parseInt(student.roll_number),
              section_id: parseInt(selectedSection),
              class_time: new Date(classTime).toISOString(),
              is_present: attendance[student.roll_number],
              course_code: courseCode,
            },
            { headers: { Authorization: `Bearer ${token}` } }
          )
        );

      if (attendancePromises.length === 0) {
        setError("No students selected for update");
        return;
      }

      await Promise.all(attendancePromises);
      setSuccess("Attendance marked successfully for selected students!");
      setError("");
      
      // Reset form
      setAttendance({});
      setUpdateStatus({});
      setFilteredStudents([]);
      setStudents([]);
      setSelectedSection("");
      setCourseCode("");
      setClassTime("");
      setRollFilter("");
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to mark attendance");
      setSuccess("");
    }
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="mark-attendance-container">
      <h1 className="page-title">{pageTitle}</h1>
      
      {success && <div className="success-message">{success}</div>}

      <div className="attendance-form">
        {/* Section Selection */}
        <div className="form-group">
          <label>Section</label>
          <select
            value={selectedSection}
            onChange={(e) => setSelectedSection(e.target.value)}
            className="form-control"
          >
            <option value="">-- Select Section --</option>
            {sections.map((section) => (
              <option key={section.section_id} value={section.section_id}>
                {section.section_name} ({section.course_code})
              </option>
            ))}
          </select>
        </div>

        {/* Course Code Input */}
        <div className="form-group">
          <label>Course Code</label>
          <input
            type="text"
            value={courseCode}
            onChange={(e) => setCourseCode(e.target.value)}
            placeholder="e.g., CS101"
            className="form-control"
          />
        </div>

        {/* Class Time Input */}
        <div className="form-group">
          <label>Class Time</label>
          <input
            type="datetime-local"
            value={classTime}
            onChange={(e) => setClassTime(e.target.value)}
            className="form-control"
          />
        </div>

        {/* Fetch Students Button */}
        <button
          onClick={fetchStudents}
          className="btn btn-primary fetch-students-btn"
          disabled={!selectedSection || !courseCode || !classTime}
        >
          Fetch Students
        </button>

        {/* Students List */}
        {filteredStudents.length > 0 && (
          <div className="students-section">
            {/* Roll Number Filter */}
            <div className="form-group roll-filter">
              <label>Filter by Roll Number</label>
              <input
                type="text"
                value={rollFilter}
                onChange={handleFilter}
                placeholder="Enter roll number"
                className="form-control"
              />
            </div>

            <form onSubmit={handleSubmit} className="attendance-form">
              <div class="attendance-table-container">
                <table className="table attendance-table">
                <thead>
                  <tr>
                    <th>Roll Number</th>
                    <th>Name</th>
                    <th>Update</th>
                    <th>Present</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStudents.map((student) => (
                    <tr key={student.roll_number}>
                      <td>{student.roll_number}</td>
                      <td>{student.full_name}</td>
                      <td className="checkbox-cell">
                        <input
                          type="checkbox"
                          checked={updateStatus[student.roll_number] || false}
                          onChange={() => toggleUpdateStatus(student.roll_number)}
                        />
                      </td>
                      <td className="checkbox-cell">
                        <input
                          type="checkbox"
                          checked={attendance[student.roll_number] || false}
                          onChange={() => toggleAttendance(student.roll_number)}
                          disabled={!updateStatus[student.roll_number]}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>

              <p className="attendance-hint">
                (Tick "Present" to mark as present, leave unticked to mark as absent)
              </p>

              <button
                type="submit"
                className="btn btn-success submit-attendance-btn"
              >
                Submit Attendance for Selected Students
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
};

export default BaseMarkAttendance;