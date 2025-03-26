"use client"

import { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import {
  ArrowLeft,
  Briefcase,
  Building,
  Calendar,
  Clock,
  FileText,
  AlertCircle,
  CheckCircle,
  XCircle,
  User,
  Eye,
} from "lucide-react"

const MyApplications = () => {
  const [applications, setApplications] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState("all") // all, pending, reviewed, interviewed, rejected, hired

  useEffect(() => {
    fetchApplications()
  }, [])

  const fetchApplications = async () => {
    try {
      setLoading(true)
      const response = await fetch("http://localhost:5000/api/applications", {
        credentials: "include",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        }
      })

      if (!response.ok) {
        throw new Error("Failed to fetch applications")
      }

      const data = await response.json()
      setApplications(data)
    } catch (err) {
      console.error("Error fetching applications:", err)
      setError(err.message || "An error occurred while fetching your applications")
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (status) => {
    switch (status) {
      case "pending":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
            <Clock className="h-3 w-3 mr-1" />
            Pending
          </span>
        )
      case "reviewed":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            <Eye className="h-3 w-3 mr-1" />
            Reviewed
          </span>
        )
      case "interviewed":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
            <User className="h-3 w-3 mr-1" />
            Interviewed
          </span>
        )
      case "rejected":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            <XCircle className="h-3 w-3 mr-1" />
            Rejected
          </span>
        )
      case "hired":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle className="h-3 w-3 mr-1" />
            Hired
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
            {status}
          </span>
        )
    }
  }

  const filteredApplications = applications.filter((app) => {
    if (filter === "all") return true
    return app.status === filter
  })

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center">
            <Link to="/dashboard" className="text-blue-600 hover:text-blue-800 mr-4">
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">My Applications</h1>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Filter tabs */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setFilter("all")}
              className={`px-4 py-2 rounded-md text-sm ${
                filter === "all" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              All Applications
            </button>
            <button
              onClick={() => setFilter("pending")}
              className={`px-4 py-2 rounded-md text-sm flex items-center ${
                filter === "pending" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              <Clock className="h-4 w-4 mr-2" />
              Pending
            </button>
            <button
              onClick={() => setFilter("reviewed")}
              className={`px-4 py-2 rounded-md text-sm flex items-center ${
                filter === "reviewed" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              <Eye className="h-4 w-4 mr-2" />
              Reviewed
            </button>
            <button
              onClick={() => setFilter("interviewed")}
              className={`px-4 py-2 rounded-md text-sm flex items-center ${
                filter === "interviewed" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              <User className="h-4 w-4 mr-2" />
              Interviewed
            </button>
            <button
              onClick={() => setFilter("rejected")}
              className={`px-4 py-2 rounded-md text-sm flex items-center ${
                filter === "rejected" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              <XCircle className="h-4 w-4 mr-2" />
              Rejected
            </button>
            <button
              onClick={() => setFilter("hired")}
              className={`px-4 py-2 rounded-md text-sm flex items-center ${
                filter === "hired" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
              }`}
            >
              <CheckCircle className="h-4 w-4 mr-2" />
              Hired
            </button>
          </div>
        </div>

        {/* Applications list */}
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
            <p className="mt-2 text-gray-600">Loading applications...</p>
          </div>
        ) : error ? (
          <div className="bg-white rounded-lg shadow-sm p-8 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-500 mb-4">{error}</p>
            <button
              onClick={fetchApplications}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        ) : applications.length === 0 ? (
          <div className="bg-white rounded-lg shadow-sm p-8 text-center">
            <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Applications Yet</h3>
            <p className="text-gray-600 mb-6">You haven't applied to any jobs yet.</p>
            <Link
              to="/jobs"
              className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors inline-flex items-center"
            >
              Browse Jobs
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredApplications.map((application) => (
              <div
                key={application.id}
                className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex flex-col md:flex-row md:items-start md:justify-between">
                  <div>
                    <Link
                      to={`/jobs/${application.job_id}`}
                      className="text-lg font-semibold text-gray-900 hover:text-blue-600"
                    >
                      {application.jobTitle}
                    </Link>
                    <div className="mt-1 flex items-center text-gray-600">
                      <Building className="h-4 w-4 mr-1" />
                      {application.company}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        <Briefcase className="h-3 w-3 mr-1" />
                        {application.location}
                      </div>
                      <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        <Calendar className="h-3 w-3 mr-1" />
                        Applied: {new Date(application.created_at).toLocaleDateString()}
                      </div>
                      {getStatusBadge(application.status)}
                    </div>
                  </div>
                  <div className="mt-4 md:mt-0">
                    <Link
                      to={`/applications/${application.id}`}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      View Details
                    </Link>
                  </div>
                </div>

                {/* Additional information based on status */}
                {application.status === "rejected" && application.feedback && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-100 rounded-md">
                    <p className="text-sm font-medium text-red-800">Feedback:</p>
                    <p className="text-sm text-red-700">{application.feedback}</p>
                    <p className="text-xs text-red-600 mt-1">
                      Updated: {new Date(application.updatedAt).toLocaleString()}
                    </p>
                  </div>
                )}

                {application.status === "reviewed" && (
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-100 rounded-md">
                    <p className="text-sm font-medium text-blue-800">Your application has been reviewed</p>
                    <p className="text-xs text-blue-600 mt-1">
                      Updated: {new Date(application.updatedAt).toLocaleString()}
                    </p>
                  </div>
                )}

                {application.status === "interviewed" && (
                  <div className="mt-4 p-3 bg-purple-50 border border-purple-100 rounded-md">
                    <p className="text-sm font-medium text-purple-800">You have been selected for an interview</p>
                    <p className="text-xs text-purple-600 mt-1">
                      Updated: {new Date(application.updatedAt).toLocaleString()}
                    </p>
                  </div>
                )}

                {application.status === "hired" && (
                  <div className="mt-4 p-3 bg-green-50 border border-green-100 rounded-md">
                    <p className="text-sm font-medium text-green-800">Congratulations! You have been hired</p>
                    <p className="text-xs text-green-600 mt-1">
                      Updated: {new Date(application.updatedAt).toLocaleString()}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

export default MyApplications

