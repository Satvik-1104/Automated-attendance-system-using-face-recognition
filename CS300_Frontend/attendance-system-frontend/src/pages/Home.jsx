import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";

// Create a separate CSS file for Home component styles
import './Home.css';

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const Home = () => {
  const { token } = useAuth();
  const [stats, setStats] = useState({
    total_absent: 0,
    total_attended: 0,
    pending_corrections: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch data when the component mounts
  useEffect(() => {
    const fetchSummary = async () => {
      try {
        // Fetch attendance summary
        const attendanceResponse = await axios.get(
          `${BASE_URL}/students/attendance/summary`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const { total_absent, total_attended } = attendanceResponse.data;

        // Fetch pending corrections
        const correctionsResponse = await axios.get(
          `${BASE_URL}/students/corrections/pending`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const { pending_count } = correctionsResponse.data;

        // Update state with fetched data
        setStats({
          total_absent,
          total_attended,
          pending_corrections: pending_count,
        });
        setLoading(false);
      } catch (err) {
        setError("Failed to load summary data");
        setLoading(false);
      }
    };
    fetchSummary();
  }, [token]);

  // Stat card component for reusability
  const StatCard = ({ title, value, className }) => (
    <div className={`student-stat-card ${className}`}>
      <h3 className="student-stat-card-title">{title}</h3>
      <p className="student-stat-card-value">{value}</p>
    </div>
  );

  // Conditional rendering for loading and error states
  if (loading) return <div className="student-loading">Loading...</div>;
  if (error) return <div className="student-error">Loading...</div>;

  // Render the home page with stats cards
  return (
    <div className="student-home-container">
      <h1 className="student-page-title">Dashboard Overview</h1>
      <div className="student-stats-grid">
        <StatCard
          title="Classes Absent"
          value={stats.total_absent}
          className="student-absent-classes"
        />
        <StatCard
          title="Pending Corrections"
          value={stats.pending_corrections}
          className="student-pending-corrections"
        />
        <StatCard
          title="Classes Attended"
          value={stats.total_attended}
          className="student-attended-classes"
        />
      </div>
    </div>
  );
};

export default Home;