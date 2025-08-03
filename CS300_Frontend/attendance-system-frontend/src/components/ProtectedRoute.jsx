// src/components/ProtectedRoute.jsx
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext"; // Import useAuth as a named export

const ProtectedRoute = ({ children, allowedRoles }) => {
  const { token, role } = useAuth(); // Use useAuth hook instead of useContext
  if (!token || !allowedRoles.includes(role)) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

export default ProtectedRoute;