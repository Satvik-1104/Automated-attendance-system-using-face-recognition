// src/components/PhDDashboardLayout.jsx
import BaseDashboardLayout from "./BaseDashboardLayout";

const PhDDashboardLayout = () => {
  const phdNavItems = [
    { 
      label: "Home", 
      path: "/phd/dashboard/home",
      icon: "🏠"
    },
    { 
      label: "Mark Attendance", 
      path: "/phd/dashboard/mark-attendance",
      icon: "📝"
    },
    { 
      label: "Attendance Reports", 
      path: "/phd/dashboard/attendance-reports",
      icon: "📊"
    },
    { 
      label: "Approve Corrections", 
      path: "/phd/dashboard/approve-corrections",
      icon: "✅"
    },
    { 
      label: "Update Schedule", 
      path: "/phd/dashboard/update-schedule",
      icon: "🕒"
    },
    { 
      label: "Generate Report", 
      path: "/phd/dashboard/generate-report",
      icon: "📋"
    }
  ];

  return (
    <BaseDashboardLayout 
      dashboardTitle="PhD Dashboard" 
      navItems={phdNavItems} 
    />
  );
};

export default PhDDashboardLayout;