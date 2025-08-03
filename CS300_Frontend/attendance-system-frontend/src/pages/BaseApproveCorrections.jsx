// src/components/BaseApproveCorrectionss.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './ApproveCorrections.css';

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const BaseApproveCorrections = ({ 
  apiEndpoints, 
  pageTitle = "Approve Corrections",
  role = "faculty" 
}) => {
  const { token } = useAuth();
  const [corrections, setCorrections] = useState([]);
  const [selectedCorrections, setSelectedCorrections] = useState(new Set());
  const [feedback, setFeedback] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Fetch pending corrections on mount
  useEffect(() => {
    const fetchCorrections = async () => {
      try {
        const response = await axios.get(
          apiEndpoints.pendingCorrections,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setCorrections(response.data.pending_corrections || []);
        setLoading(false);
      } catch (err) {
        const errMsg =
          typeof err.response?.data?.detail === "object"
            ? JSON.stringify(err.response.data.detail)
            : err.response?.data?.detail || "Failed to load correction requests";
        setError(errMsg);
        setLoading(false);
      }
    };
    fetchCorrections();
  }, [token]);

  // Toggle selection of a correction for bulk actions
  const toggleSelection = (correctionId) => {
    setSelectedCorrections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(correctionId)) {
        newSet.delete(correctionId);
      } else {
        newSet.add(correctionId);
      }
      return newSet;
    });
  };

  // Handle bulk approval
  const handleBulkApprove = async () => {
    if (selectedCorrections.size === 0) {
      setError("Please select at least one correction to approve");
      return;
    }

    try {
      const approvalPromises = Array.from(selectedCorrections).map((correctionId) =>
        axios.post(
          apiEndpoints.approveCorrection,
          { correction_id: correctionId, approval_status: true, feedback: "" },
          { headers: { Authorization: `Bearer ${token}` } }
        )
      );
      await Promise.all(approvalPromises);
      setCorrections(corrections.filter((c) => !selectedCorrections.has(c.correction_id)));
      setSuccess(`Approved ${selectedCorrections.size} correction(s) successfully!`);
      setSelectedCorrections(new Set());
      setError("");
    } catch (err) {
      const errMsg =
        typeof err.response?.data?.detail === "object"
          ? JSON.stringify(err.response.data.detail)
          : err.response?.data?.detail || "Failed to approve corrections";
      setError(errMsg);
      setSuccess("");
    }
  };

  // Handle bulk rejection
  const handleBulkReject = async () => {
    if (selectedCorrections.size === 0) {
      setError("Please select at least one correction to reject");
      return;
    }

    try {
      const rejectionPromises = Array.from(selectedCorrections).map((correctionId) =>
        axios.post(
          apiEndpoints.approveCorrection,
          {
            correction_id: correctionId,
            approval_status: false,
            feedback: feedback[correctionId] || "Bulk rejected",
          },
          { headers: { Authorization: `Bearer ${token}` } }
        )
      );
      await Promise.all(rejectionPromises);
      setCorrections(corrections.filter((c) => !selectedCorrections.has(c.correction_id)));
      setSelectedCorrections(new Set());
      setFeedback((prev) => {
        const newFeedback = { ...prev };
        selectedCorrections.forEach((id) => delete newFeedback[id]);
        return newFeedback;
      });
      setSuccess(`Rejected ${selectedCorrections.size} correction(s) successfully!`);
      setError("");
    } catch (err) {
      const errMsg =
        typeof err.response?.data?.detail === "object"
          ? JSON.stringify(err.response.data.detail)
          : err.response?.data?.detail || "Failed to reject corrections";
      setError(errMsg);
      setSuccess("");
    }
  };

  // Handle individual approval
  const handleApprove = async (correctionId) => {
    try {
      await axios.post(
        apiEndpoints.approveCorrection,
        { correction_id: correctionId, approval_status: true, feedback: "" },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setCorrections(corrections.filter((c) => c.correction_id !== correctionId));
      setSuccess("Correction approved successfully!");
      setError("");
    } catch (err) {
      const errMsg =
        typeof err.response?.data?.detail === "object"
          ? JSON.stringify(err.response.data.detail)
          : err.response?.data?.detail || "Failed to approve correction";
      setError(errMsg);
      setSuccess("");
    }
  };

  // Handle individual rejection
  const handleReject = async (correctionId) => {
    const feedbackText = feedback[correctionId] || "";
    try {
      await axios.post(
        apiEndpoints.approveCorrection,
        { correction_id: correctionId, approval_status: false, feedback: feedbackText },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setCorrections(corrections.filter((c) => c.correction_id !== correctionId));
      setFeedback((prev) => {
        const newFeedback = { ...prev };
        delete newFeedback[correctionId];
        return newFeedback;
      });
      setSuccess("Correction rejected successfully!");
      setError("");
    } catch (err) {
      const errMsg =
        typeof err.response?.data?.detail === "object"
          ? JSON.stringify(err.response.data.detail)
          : err.response?.data?.detail || "Failed to reject correction";
      setError(errMsg);
      setSuccess("");
    }
  };

  // Update feedback state
  const handleFeedbackChange = (correctionId, value) => {
    setFeedback((prev) => ({
      ...prev,
      [correctionId]: value,
    }));
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="approve-corrections-container">
      <h1 className="page-title">{pageTitle}</h1>
      
      {success && <div className="success-message">{success}</div>}

      {/* Bulk Action Buttons */}
      <div className="bulk-actions">
        <button
          onClick={handleBulkApprove}
          className="btn btn-primary bulk-approve"
          disabled={selectedCorrections.size === 0}
        >
          Approve Selected
        </button>
        <button
          onClick={handleBulkReject}
          className="btn btn-danger bulk-reject"
          disabled={selectedCorrections.size === 0}
        >
          Reject Selected
        </button>
      </div>

      {/* Corrections Table */}
      {corrections.length === 0 ? (
        <div className="no-corrections">No pending correction requests.</div>
      ) : (
        <div className="corrections-table-container">
          <table className="corrections-table">
            <thead>
              <tr>
                <th>Select</th>
                <th>Roll Number</th>
                <th>Course Code</th>
                <th>Date</th>
                <th>Reason</th>
                <th>Image</th>
                <th>Actions</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {corrections.map((correction) => (
                <tr key={correction.correction_id}>
                  <td className="select-cell">
                    <input
                      type="checkbox"
                      checked={selectedCorrections.has(correction.correction_id)}
                      onChange={() => toggleSelection(correction.correction_id)}
                    />
                  </td>
                  <td>{correction.roll_number}</td>
                  <td>{correction.course_code}</td>
                  <td>{new Date(correction.class_time).toLocaleDateString()}</td>
                  <td>{correction.reason}</td>
                  <td>
                    {correction.supporting_image ? (
                      <a
                        href={`${apiEndpoints.uploadBase}${correction.supporting_image}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="image-link"
                      >
                        View Image
                      </a>
                    ) : (
                      "N/A"
                    )}
                  </td>
                  <td className="actions-cell">
                    <button
                      onClick={() => handleApprove(correction.correction_id)}
                      className="btn btn-success approve-btn"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => handleReject(correction.correction_id)}
                      className="btn btn-danger reject-btn"
                    >
                      Reject
                    </button>
                  </td>
                  <td>
                    <textarea
                      value={feedback[correction.correction_id] || ""}
                      onChange={(e) =>
                        handleFeedbackChange(correction.correction_id, e.target.value)
                      }
                      placeholder="Optional feedback (for rejection)"
                      className="feedback-textarea"
                    />
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

export default BaseApproveCorrections;