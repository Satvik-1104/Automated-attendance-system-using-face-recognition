// src/pages/CorrectionsList.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import './CorrectionsList.css';

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const CorrectionsList = () => {
  const { token } = useAuth();
  const [corrections, setCorrections] = useState([]);
  const [filteredCorrections, setFilteredCorrections] = useState([]);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch correction requests when the component mounts
  useEffect(() => {
    const fetchCorrections = async () => {
      try {
        const response = await axios.get(
          `${BASE_URL}/students/corrections`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        setCorrections(response.data.corrections);
        setFilteredCorrections(response.data.corrections);
        setLoading(false);
      } catch (err) {
        setError("Failed to load correction requests");
        setLoading(false);
      }
    };
    fetchCorrections();
  }, [token]);

  // Update filtered corrections when the filter changes
  useEffect(() => {
    if (filter === "all") {
      setFilteredCorrections(corrections);
    } else {
      setFilteredCorrections(
        corrections.filter((correction) => correction.status.toLowerCase() === filter)
      );
    }
  }, [filter, corrections]);

  // Render status badge with appropriate color
  const renderStatusBadge = (status) => {
    const statusClass = status.toLowerCase();
    return (
      <span className={`status-badge ${statusClass}`}>
        {status}
      </span>
    );
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="corrections-list-container">
      <h1 className="page-title">Corrections List</h1>

      {/* Filter Dropdown */}
      <div className="filter-section">
        <label htmlFor="filter">Filter by: </label>
        <select
          id="filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="form-control"
        >
          <option value="all">All Corrections</option>
          <option value="pending">Pending Corrections</option>
          <option value="approved">Approved Corrections</option>
          <option value="rejected">Rejected Corrections</option>
        </select>
      </div>

      {/* Corrections Table */}
      {filteredCorrections.length === 0 ? (
        <div className="no-corrections">No correction requests found.</div>
      ) : (
        <div className="corrections-table-container">
          <table className="corrections-table">
            <thead>
              <tr>
                <th>Report ID</th>
                <th>Course Code</th>
                <th>Request Time</th>
                <th>Status</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {filteredCorrections.map((correction) => (
                <tr key={correction.correction_id}>
                  <td>{correction.report_id}</td>
                  <td>{correction.course_code}</td>
                  <td>{new Date(correction.request_time).toLocaleString()}</td>
                  <td>
                    {renderStatusBadge(correction.status)}
                  </td>
                  <td>{correction.feedback || "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CorrectionsList;