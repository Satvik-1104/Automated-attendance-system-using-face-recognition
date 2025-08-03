// src/components/BaseDashboardLayout.jsx
import React, { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import './DashboardLayout.css';

const BaseDashboardLayout = ({ 
  dashboardTitle, 
  navItems 
}) => {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="dashboard-layout">
      {/* Mobile Menu Toggle */}
      <button 
        className="mobile-menu-toggle"
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        aria-label="Toggle Navigation Menu"
      >
        {isMobileMenuOpen ? '✕' : '☰'}
      </button>

      {/* Sidebar */}
      <div className={`dashboard-sidebar ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
        <h2 className="dashboard-title">{dashboardTitle}</h2>
        <nav className="dashboard-nav">
          <ul>
            {navItems.map((item) => (
              <li key={item.path}>
                <NavLink 
                  to={item.path}
                  className={({ isActive }) => 
                    `dashboard-nav-link ${isActive ? 'active' : ''}`
                  }
                >
                  {item.icon && <span className="nav-icon">{item.icon}</span>}
                  {item.label}
                </NavLink>
              </li>
            ))}
            
            <li className="logout-section">
              <button 
                onClick={handleLogout}
                className="logout-button"
              >
                Logout
              </button>
            </li>
          </ul>
        </nav>
      </div>

      {/* Main Content Area */}
      <main className="dashboard-content">
        <Outlet />
      </main>
    </div>
  );
};

export default BaseDashboardLayout;