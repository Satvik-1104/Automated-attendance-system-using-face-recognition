// src/pages/PhDUpdateSchedule.jsx
import BaseUpdateSchedule from "./BaseUpdateSchedule";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PhDUpdateSchedule = () => {
  const phdApiEndpoints = {
    assignedSections: `${BASE_URL}/phd_students/assigned_sections`,
    effectiveSchedules: `${BASE_URL}/phd_students/effective_schedules`,
    updateSchedule: `${BASE_URL}/phd_students/update_schedule`
  };

  return (
    <BaseUpdateSchedule 
      apiEndpoints={phdApiEndpoints}
      pageTitle="PhD - Update Schedule"
      role="phd"
    />
  );
};

export default PhDUpdateSchedule;