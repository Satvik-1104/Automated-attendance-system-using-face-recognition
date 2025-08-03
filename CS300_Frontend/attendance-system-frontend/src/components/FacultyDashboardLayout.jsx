// src/components/FacultyDashboardLayout.jsx
import BaseDashboardLayout from "./BaseDashboardLayout";

const FacultyDashboardLayout = () => {
  const facultyNavItems = [
    { 
      label: "Home", 
      path: "/faculty/dashboard/home",
      icon: "🏠"
    },
    { 
      label: "Mark Attendance", 
      path: "/faculty/dashboard/mark-attendance",
      icon: "📝"
    },
    { 
      label: "Attendance Reports", 
      path: "/faculty/dashboard/attendance-reports",
      icon: "📊"
    },
    { 
      label: "Approve Corrections", 
      path: "/faculty/dashboard/approve-corrections",
      icon: "✅"
    },
    { 
      label: "Update Schedule", 
      path: "/faculty/dashboard/update-schedule",
      icon: "🕒"
    },
    { 
      label: "Generate Report", 
      path: "/faculty/dashboard/generate-report",
      icon: "📋"
    }
  ];

  return (
    <BaseDashboardLayout 
      dashboardTitle="Faculty Dashboard" 
      navItems={facultyNavItems} 
    />
  );
};

export default FacultyDashboardLayout;