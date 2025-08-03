// src/components/DashboardLayout.jsx
import BaseDashboardLayout from "./BaseDashboardLayout";

const DashboardLayout = () => {
  const studentNavItems = [
    { 
      label: "Home", 
      path: "/student/dashboard/home",
      icon: "🏠"
    },
    { 
      label: "Attendance Reports", 
      path: "/student/dashboard/attendance-reports",
      icon: "📊"
    },
    { 
      label: "Request Corrections", 
      path: "/student/dashboard/request-corrections",
      icon: "✏️"
    },
    { 
      label: "Corrections List", 
      path: "/student/dashboard/corrections-list",
      icon: "📋"
    }
  ];

  return (
    <BaseDashboardLayout 
      dashboardTitle="Student Dashboard" 
      navItems={studentNavItems} 
    />
  );
};

export default DashboardLayout;