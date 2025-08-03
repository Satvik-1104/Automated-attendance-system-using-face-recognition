// src/pages/UpdateSchedule.jsx
import BaseUpdateSchedule from "./BaseUpdateSchedule";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const UpdateSchedule = () => {
  const facultyApiEndpoints = {
    assignedSections: `${BASE_URL}/faculty/assigned_sections`,
    effectiveSchedules: `${BASE_URL}/faculty/effective_schedules`,
    updateSchedule: `${BASE_URL}/faculty/update_schedule`
  };

  return (
    <BaseUpdateSchedule 
      apiEndpoints={facultyApiEndpoints}
      pageTitle="Faculty - Update Schedule"
      role="faculty"
    />
  );
};

export default UpdateSchedule;