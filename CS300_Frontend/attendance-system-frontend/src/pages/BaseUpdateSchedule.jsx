// src/components/BaseUpdateSchedule.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../context/AuthContext";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import moment from "moment";
import './UpdateSchedule.css';

const BaseUpdateSchedule = ({ 
  apiEndpoints, 
  pageTitle = "Update Schedule",
  role = "faculty" 
}) => {
  const { token } = useAuth();
  const [effectiveSchedules, setEffectiveSchedules] = useState([]);
  const [filteredSchedules, setFilteredSchedules] = useState([]);
  const [selectedWeekStart, setSelectedWeekStart] = useState(new Date());
  const [sections, setSections] = useState([]);
  const [courseCodeFilter, setCourseCodeFilter] = useState("");
  const [sectionFilter, setSectionFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [editingSchedule, setEditingSchedule] = useState(null);
  const [newSchedule, setNewSchedule] = useState({
    update_type: "",
    course_code: "",
    section_id: "",
    original_date: "",
    new_date: "",
    new_start_time: "",
    new_end_time: "",
    new_location: "",
    reason: "",
  });

  // Fetch data when component mounts or week changes
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Fetch sections
        const sectionsResponse = await axios.get(
          apiEndpoints.assignedSections,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setSections(sectionsResponse.data.schedules || []);

        // Fetch effective schedules
        const weekStartStr = moment(selectedWeekStart).format("YYYY-MM-DD");
        const effectiveResponse = await axios.get(
          `${apiEndpoints.effectiveSchedules}/${weekStartStr}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const fetchedSchedules = effectiveResponse.data.schedules || [];

        const weekStart = moment(selectedWeekStart).startOf("day");
        const weekEnd = moment(weekStart).add(6, "days").endOf("day");

        const uniqueSchedules = [];
        const seen = new Set();
        fetchedSchedules.forEach((schedule) => {
          const scheduleDate = moment(schedule.date);
          const uniqueKey = `${schedule.date}-${schedule.course_code}-${schedule.section_id}-${schedule.start_time}-${schedule.end_time}`;
          if (
            scheduleDate.isBetween(weekStart, weekEnd, null, "[]") &&
            !seen.has(uniqueKey)
          ) {
            seen.add(uniqueKey);
            uniqueSchedules.push(schedule);
          }
        });

        setEffectiveSchedules(uniqueSchedules);
        setFilteredSchedules(uniqueSchedules);
        setLoading(false);
      } catch (err) {
        setError("Failed to load data");
        setLoading(false);
      }
    };
    fetchData();
  }, [token, selectedWeekStart, apiEndpoints]);

  // Apply course code and section filters
  useEffect(() => {
    let filtered = [...effectiveSchedules];
    if (courseCodeFilter) {
      filtered = filtered.filter((schedule) =>
        schedule.course_code
          .toLowerCase()
          .includes(courseCodeFilter.toLowerCase())
      );
    }
    if (sectionFilter) {
      filtered = filtered.filter(
        (schedule) => schedule.section_id.toString() === sectionFilter
      );
    }
    setFilteredSchedules(filtered);
  }, [courseCodeFilter, sectionFilter, effectiveSchedules]);

  // Handle starting the update process
  const startEditing = (schedule) => {
    setEditingSchedule(schedule);
    setNewSchedule({
      update_type: "",
      course_code: schedule.course_code,
      section_id: schedule.section_id.toString(),
      original_date: schedule.date,
      new_date: "",
      new_start_time: schedule.start_time,
      new_end_time: schedule.end_time,
      new_location: schedule.location,
      reason: "",
    });
    setError("");
    setSuccess("");
  };

  // Handle form input changes
  const handleNewScheduleChange = (field, value) => {
    setNewSchedule((prev) => ({ ...prev, [field]: value }));
  };

  // Submit the DSU request
  const handleUpdateSchedule = async () => {
    // Validation
    if (
      !newSchedule.update_type ||
      !newSchedule.course_code ||
      !newSchedule.section_id ||
      !newSchedule.original_date ||
      !newSchedule.reason
    ) {
      setError("Please fill in all required fields (update type, original date, reason)");
      return;
    }

    if (
      newSchedule.update_type === "RESCHEDULED" &&
      (!newSchedule.new_date || !newSchedule.new_start_time || !newSchedule.new_location)
    ) {
      setError("Please fill in new date, start time, and location for rescheduling");
      return;
    }

    try {
      const payload = {
        update_type: newSchedule.update_type,
        course_code: newSchedule.course_code,
        section_id: parseInt(newSchedule.section_id),
        original_date: newSchedule.original_date,
        reason: newSchedule.reason,
      };

      if (newSchedule.update_type === "RESCHEDULED") {
        const newTime = new Date(
          `${newSchedule.new_date}T${newSchedule.new_start_time}:00Z`
        ).toISOString();
        payload.new_time = newTime;
        payload.new_location = newSchedule.new_location;
      }

      // Submit update request
      await axios.post(
        apiEndpoints.updateSchedule,
        payload,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Refresh effective schedules
      const weekStartStr = moment(selectedWeekStart).format("YYYY-MM-DD");
      const effectiveResponse = await axios.get(
        `${apiEndpoints.effectiveSchedules}/${weekStartStr}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const fetchedSchedules = effectiveResponse.data.schedules || [];
      const weekStart = moment(selectedWeekStart).startOf("day");
      const weekEnd = moment(weekStart).add(6, "days").endOf("day");

      const uniqueSchedules = [];
      const seen = new Set();
      fetchedSchedules.forEach((schedule) => {
        const scheduleDate = moment(schedule.date);
        const uniqueKey = `${schedule.date}-${schedule.course_code}-${schedule.section_id}-${schedule.start_time}-${schedule.end_time}`;
        if (
          scheduleDate.isBetween(weekStart, weekEnd, null, "[]") &&
          !seen.has(uniqueKey)
        ) {
          seen.add(uniqueKey);
          uniqueSchedules.push(schedule);
        }
      });

      // Update state
      setEffectiveSchedules(uniqueSchedules);
      setFilteredSchedules(uniqueSchedules);
      setEditingSchedule(null);
      setNewSchedule({
        update_type: "",
        course_code: "",
        section_id: "",
        original_date: "",
        new_date: "",
        new_start_time: "",
        new_end_time: "",
        new_location: "",
        reason: "",
      });
      setSuccess("Schedule update request submitted successfully!");
      setError("");
    } catch (err) {
      const errorMessage =
        err.response?.data?.detail || "Failed to submit schedule update request";
      setError(errorMessage);
      setSuccess("");
    }
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error && !success) return <div className="error-message">{error}</div>;

  return (
    <div className="update-schedule-container">
      <h1 className="page-title">{pageTitle}</h1>
      
      {success && <div className="success-message">{success}</div>}
      {error && <div className="error-message">{error}</div>}

      {/* Week Selector */}
      <div className="week-selector">
        <label>Select Week Starting: </label>
        <DatePicker
          selected={selectedWeekStart}
          onChange={(date) => setSelectedWeekStart(date)}
          dateFormat="yyyy-MM-dd"
          placeholderText="Select week start date"
          className="form-control"
        />
      </div>

      {/* Filters */}
      <h2 className="section-title">Effective Schedule for Selected Week</h2>
      <div className="filters">
        <div className="filter-group">
          <label>Filter by Course Code: </label>
          <input
            type="text"
            value={courseCodeFilter}
            onChange={(e) => setCourseCodeFilter(e.target.value)}
            placeholder="e.g., CS101"
            className="form-control"
          />
        </div>
        <div className="filter-group">
          <label>Filter by Section: </label>
          <select
            value={sectionFilter}
            onChange={(e) => setSectionFilter(e.target.value)}
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
      </div>

      {/* Effective Schedule Table */}
      {filteredSchedules.length === 0 ? (
        <div className="no-schedules">No schedules found for this week.</div>
      ) : (
        <div className="schedules-table-container">
          <table className="schedules-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Day</th>
                <th>Course Code</th>
                <th>Section</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Location</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredSchedules.map((schedule) => (
                <tr
                  key={`${schedule.date}-${schedule.course_code}-${schedule.section_id}-${schedule.start_time}-${schedule.end_time}`}
                >
                  <td>{schedule.date}</td>
                  <td>{moment(schedule.date).format("dddd")}</td>
                  <td>{schedule.course_code}</td>
                  <td>{schedule.section_name}</td>
                  <td>{schedule.start_time}</td>
                  <td>{schedule.end_time}</td>
                  <td>{schedule.location}</td>
                  <td>{schedule.is_updated ? "Updated" : "Original"}</td>
                  <td>
                    <button
                      onClick={() => startEditing(schedule)}
                      className="btn btn-primary request-update-btn"
                    >
                      Request Update
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* DSU Request Form */}
      {editingSchedule && (
        <div className="schedule-update-form">
          <h2>Request Schedule Update</h2>
          <div className="form-group">
            <label>Update Type: </label>
            <select
              value={newSchedule.update_type}
              onChange={(e) => handleNewScheduleChange("update_type", e.target.value)}
              className="form-control"
            >
              <option value="">-- Select Type --</option>
              <option value="CANCELLED">Cancel Class</option>
              <option value="RESCHEDULED">Reschedule Class</option>
            </select>
          </div>
          <div className="form-group">
            <label>Course Code: </label>
            <input
              type="text"
              value={newSchedule.course_code}
              disabled
              className="form-control"
            />
          </div>
          <div className="form-group">
            <label>Section: </label>
            <input
              type="text"
              value={newSchedule.section_id}
              disabled
              className="form-control"
            />
          </div>
          <div className="form-group">
            <label>Original Date: </label>
            <input
              type="date"
              value={newSchedule.original_date}
              onChange={(e) => handleNewScheduleChange("original_date", e.target.value)}
              className="form-control"
            />
          </div>
          {newSchedule.update_type === "RESCHEDULED" && (
            <>
              <div className="form-group">
                <label>New Date: </label>
                <input
                  type="date"
                  value={newSchedule.new_date}
                  onChange={(e) => handleNewScheduleChange("new_date", e.target.value)}
                  className="form-control"
                />
              </div>
              <div className="form-group">
                <label>New Start Time: </label>
                <input
                  type="time"
                  value={newSchedule.new_start_time}
                  onChange={(e) => handleNewScheduleChange("new_start_time", e.target.value)}
                  className="form-control"
                />
              </div>
              <div className="form-group">
                <label>New End Time: </label>
                <input
                  type="time"
                  value={newSchedule.new_end_time}
                  onChange={(e) => handleNewScheduleChange("new_end_time", e.target.value)}
                  className="form-control"
                />
              </div>
              <div className="form-group">
                <label>New Location: </label>
                <input
                  type="text"
                  value={newSchedule.new_location}
                  onChange={(e) => handleNewScheduleChange("new_location", e.target.value)}
                  className="form-control"
                />
              </div>
            </>
          )}
          <div className="form-group">
            <label>Reason: </label>
            <textarea
              value={newSchedule.reason}
              onChange={(e) => handleNewScheduleChange("reason", e.target.value)}
              placeholder="Enter reason for update"
              className="form-control reason-textarea"
            />
          </div>
          <div className="form-actions">
            <button
              onClick={handleUpdateSchedule}
              className="btn btn-success submit-btn"
            >
              Submit Update
            </button>
            <button
              onClick={() => setEditingSchedule(null)}
              className="btn btn-danger cancel-btn"
            >
              Cancel
            </button>
            </div>
        </div>
      )}
    </div>
  );
};

export default BaseUpdateSchedule;