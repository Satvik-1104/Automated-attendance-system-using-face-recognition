// src/pages/RequestCorrections.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './RequestCorrections.css';

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const RequestCorrections = () => {
  const { token } = useAuth();
  const [attendance, setAttendance] = useState([]);
  const [selectedReportId, setSelectedReportId] = useState("");
  const [reason, setReason] = useState("");
  const [supportingImage, setSupportingImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");


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
        setLoading(false);
      } catch (err) {
        setError("Failed to load attendance records");
        setLoading(false);
      }
    };
    fetchAttendance();
  }, [token]);

  // Handle image file selection
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSupportingImage(file);

    // Create image preview
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset previous messages
    setError("");
    setSuccess("");

    if (!selectedReportId) {
      setError("Please select an attendance record");
      return;
    }

    // Find the selected attendance record
    const selectedRecord = attendance.find(
      (record) => record.report_id === parseInt(selectedReportId)
    );
    
    if (!selectedRecord) {
      setError("Invalid attendance record selected");
      return;
    }

    const formData = new FormData();
    formData.append("report_id", selectedReportId);
    formData.append("course_code", selectedRecord.course_code);
    formData.append("reason", reason);
    
    if (supportingImage) {
      formData.append("supporting_image", supportingImage);
    }

    try {
      await axios.post(
        `${BASE_URL}/students/request_correction`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );
      
      // Success handling
      setSuccess("Correction request submitted successfully");
      setSelectedReportId("");
      setReason("");
      setSupportingImage(null);
      setImagePreview(null);
      
      // Reset file input
      if (document.getElementById('image')) {
        document.getElementById('image').value = '';
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to submit correction request");
    }
  };

  // Render attendance record label
  const renderAttendanceRecordLabel = (record) => {
    const date = new Date(record.class_time).toLocaleDateString();
    const status = record.is_absent ? "Absent" : "Present";
    return `${date} - ${record.course_code} - ${status}`;
  };

  if (loading) return <div className="loading">Loading...</div>;

  return (
    <div className="request-corrections-container">
      <h1 className="page-title">Request Corrections</h1>
      
      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}

      <form onSubmit={handleSubmit} className="correction-form">
        {/* Attendance Record Selection */}
        <div className="form-group">
          <label htmlFor="report">Select Attendance Record</label>
          <select
            id="report"
            value={selectedReportId}
            onChange={(e) => setSelectedReportId(e.target.value)}
            className="form-control"
            required
          >
            <option value="">-- Select a record --</option>
            {attendance.map((record) => (
              <option 
                key={record.report_id} 
                value={record.report_id}
              >
                {renderAttendanceRecordLabel(record)}
              </option>
            ))}
          </select>
        </div>

        {/* Reason Input */}
        <div className="form-group">
          <label htmlFor="reason">Reason for Correction</label>
          <textarea
            id="reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="form-control"
            placeholder="Provide detailed explanation for the correction request"
            required
          />
        </div>

        {/* Supporting Image Upload */}
        <div className="form-group">
          <label htmlFor="image">Supporting Image (optional)</label>
          <input
            type="file"
            id="image"
            accept="image/jpeg, image/png"
            onChange={handleImageChange}
            className="form-control-file"
          />
          
          {/* Image Preview */}
          {imagePreview && (
            <div className="image-preview">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="preview-image"
              />
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div className="form-actions">
          <button 
            type="submit" 
            className="btn btn-primary req-correction-submit-btn"
          >
            Submit Correction Request
          </button>
        </div>
      </form>
    </div>
  );
};

export default RequestCorrections;