// src/AppRouter.jsx
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Login from "./pages/Login";
import ForgotPassword from "./pages/ForgotPassword";
import StudentRegister from "./pages/StudentRegister/StudentRegister";
import FacultyRegister from "./pages/FacultyRegister/FacultyRegister";
import PhdRegister from "./pages/PhdRegister/PhdRegister";
import DashboardLayout from "./components/DashboardLayout";
import FacultyDashboardLayout from "./components/FacultyDashboardLayout";
import PhDDashboardLayout from "./components/PhDDashboardLayout";
import Home from "./pages/Home";
import AttendanceReports from "./pages/AttendanceReports";
import RequestCorrections from "./pages/RequestCorrections";
import CorrectionsList from "./pages/CorrectionsList";
import FacultyHome from "./pages/FacultyHome";
import MarkAttendance from "./pages/MarkAttendance";
import FacultyAttendanceReports from "./pages/FacultyAttendanceReports";
import ApproveCorrections from "./pages/ApproveCorrections";
import UpdateSchedule from "./pages/UpdateSchedule";
import GenerateReport from "./pages/GenerateReport";
import PhDHome from "./pages/PhDHome";
import PhDMarkAttendance from "./pages/PhDMarkAttendance";
import PhDAttendanceReports from "./pages/PhDAttendanceReports";
import PhDApproveCorrections from "./pages/PhDApproveCorrections";
import PhDUpdateSchedule from "./pages/PhDUpdateSchedule";
import PhDGenerateReport from "./pages/PhDGenerateReport";
import ProtectedRoute from "./components/ProtectedRoute";

// Determine the initial page based on the mode
const getInitialPage = () => {
  const mode = import.meta.env.MODE;
  switch (mode) {
    case "faculty":
      return <FacultyRegister />;
    case "phd":
      return <PhdRegister />;
    case "student":
    default:
      return <StudentRegister />;
  }
};

const AppRouter = () => {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/register/student" element={<StudentRegister />} />
          <Route path="/register/faculty" element={<FacultyRegister />} />
          <Route path="/register/phd" element={<PhdRegister />} />

          {/* Student Dashboard (Protected) */}
          <Route
            path="/student/dashboard"
            element={
              <ProtectedRoute allowedRoles={["student"]}>
                <DashboardLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Home />} />
            <Route path="home" element={<Home />} />
            <Route path="attendance-reports" element={<AttendanceReports />} />
            <Route path="request-corrections" element={<RequestCorrections />} />
            <Route path="corrections-list" element={<CorrectionsList />} />
          </Route>

          {/* Faculty Dashboard (Protected) */}
          <Route
            path="/faculty/dashboard"
            element={
              <ProtectedRoute allowedRoles={["faculty"]}>
                <FacultyDashboardLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<FacultyHome />} />
            <Route path="home" element={<FacultyHome />} />
            <Route path="mark-attendance" element={<MarkAttendance />} />
            <Route path="attendance-reports" element={<FacultyAttendanceReports />} />
            <Route path="approve-corrections" element={<ApproveCorrections />} />
            <Route path="update-schedule" element={<UpdateSchedule />} />
            <Route path="generate-report" element={<GenerateReport />} />
          </Route>

          {/* PhD Dashboard (Protected) */}
          <Route
            path="/phd/dashboard"
            element={
              <ProtectedRoute allowedRoles={["phd_student"]}>
                <PhDDashboardLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<PhDHome />} />
            <Route path="home" element={<PhDHome />} />
            <Route path="mark-attendance" element={<PhDMarkAttendance />} />
            <Route path="attendance-reports" element={<PhDAttendanceReports />} />
            <Route path="approve-corrections" element={<PhDApproveCorrections />} />
            <Route path="update-schedule" element={<PhDUpdateSchedule />} />
            <Route path="generate-report" element={<PhDGenerateReport />} />
          </Route>

          {/* Dynamic Root Route */}
          <Route path="/" element={getInitialPage()} />
          <Route path="*" element={<h1>404 - Page Not Found</h1>} />
        </Routes>
      </AuthProvider>
    </Router>
  );
};

export default AppRouter;