"use client"

import { useState } from "react"
import { Menu, X, User, LogOut, Settings, ChevronDown } from "lucide-react"
import { Link, useNavigate } from "react-router-dom"

const Header = ({ isAuthenticated, user, onLogout, loading }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false)
  const navigate = useNavigate()

  const handleLogout = async () => {
    await onLogout()
    navigate('/')
  }

  const navigateToProfile = () => {
    setIsProfileMenuOpen(false)
    navigate('/profile')
  }

  const navigateToDashboard = () => {
    setIsProfileMenuOpen(false)
    navigate('/dashboard')
  }

  if (loading) {
    return <div>Loading...</div> // Or your loading component
  }

  return (
    <header className="sticky top-0 z-50 bg-white shadow-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center">
          <Link to="/" className="text-blue-600 font-bold text-2xl">
            Career<span className="text-blue-900">Connect</span>
          </Link>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-8">
          <a href="#features" className="text-gray-600 font-medium relative after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-blue-600 hover:text-blue-600 after:transition-all after:duration-300 hover:after:w-full">
            Features
          </a>
          <a href="#how-it-works" className="text-gray-600 font-medium relative after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-blue-600 hover:text-blue-600 after:transition-all after:duration-300 hover:after:w-full">
            How It Works
          </a>
          <a href="#for-employers" className="text-gray-600 font-medium relative after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-blue-600 hover:text-blue-600 after:transition-all after:duration-300 hover:after:w-full">
            For Employers
          </a>
          <a href="#for-job-seekers" className="text-gray-600 font-medium relative after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-blue-600 hover:text-blue-600 after:transition-all after:duration-300 hover:after:w-full">
            For Job Seekers
          </a>

          {isAuthenticated ? (
            <div className="relative">
              <button
                onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                className="flex items-center space-x-2 text-gray-700 hover:text-blue-600"
              >
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white">
                  {user?.fullName?.charAt(0) || <User size={16} />}
                </div>
                <span>{user?.fullName || "User"}</span>
                <ChevronDown className="ml-1 h-4 w-4" />
              </button>

              {isProfileMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
                  <Link
                    to="/dashboard"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={navigateToDashboard}
                  >
                    Dashboard
                  </Link>
                  <Link
                    to="/profile"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={navigateToProfile}
                  >
                    <User size={16} className="inline mr-2" />
                    My Profile
                  </Link>
                  <Link
                    to="/settings"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setIsProfileMenuOpen(false)}
                  >
                    <Settings size={16} className="inline mr-2" />
                    Settings
                  </Link>
                  <button
                    onClick={() => {
                      setIsProfileMenuOpen(false)
                      handleLogout()
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <LogOut size={16} className="inline mr-2" />
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="flex space-x-3">
              <Link
                to="/signin"
                className="px-4 py-2 bg-white text-black border border-black rounded-md transition-all duration-300 hover:bg-black hover:text-white"
              >
                Log In
              </Link>
              <Link
                to="/signup"
                className="px-4 py-2 bg-white text-black border border-black rounded-md transition-all duration-300 hover:bg-black hover:text-white"
              >
                Sign Up
              </Link>
            </div>
          )}
        </nav>

        {/* Mobile Menu Button */}
        <button className="md:hidden text-gray-700" onClick={() => setIsMenuOpen(!isMenuOpen)}>
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden bg-white py-4 px-4 shadow-md">
          <nav className="flex flex-col space-y-4">
            <a
              href="#features"
              className="text-gray-700 hover:text-blue-600 font-medium"
              onClick={() => setIsMenuOpen(false)}
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="text-gray-700 hover:text-blue-600 font-medium"
              onClick={() => setIsMenuOpen(false)}
            >
              How It Works
            </a>
            <a
              href="#for-employers"
              className="text-gray-700 hover:text-blue-600 font-medium"
              onClick={() => setIsMenuOpen(false)}
            >
              For Employers
            </a>
            <a
              href="#for-job-seekers"
              className="text-gray-700 hover:text-blue-600 font-medium"
              onClick={() => setIsMenuOpen(false)}
            >
              For Job Seekers
            </a>

            {isAuthenticated ? (
              <>
                <Link
                  to="/dashboard"
                  className="text-gray-700 hover:text-blue-600 font-medium"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Dashboard
                </Link>
                <Link
                  to="/profile"
                  className="text-gray-700 hover:text-blue-600 font-medium"
                  onClick={() => setIsMenuOpen(false)}
                >
                  My Profile
                </Link>
                <button
                  onClick={() => {
                    setIsMenuOpen(false)
                    handleLogout()
                  }}
                  className="text-left text-gray-700 hover:text-blue-600 font-medium"
                >
                  Sign Out
                </button>
              </>
            ) : (
              <div className="flex flex-col space-y-2">
                <Link
                  to="/signin"
                  className="px-4 py-2 text-blue-600 border border-blue-600 rounded-md hover:bg-blue-50 transition-colors text-center"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Log In
                </Link>
                <Link
                  to="/signup"
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-center"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Sign Up
                </Link>
              </div>
            )}
          </nav>
        </div>
      )}
    </header>
  )
}

export default Header;