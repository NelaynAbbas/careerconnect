"use client"

import { useState, useEffect } from "react"
import { Link, useParams } from "react-router-dom"
import {
  ArrowLeft,
  User,
  Mail,
  Phone,
  FileText,
  Calendar,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Download,
  MessageSquare,
  UserCheck,
  Briefcase,
} from "lucide-react"

const JobApplications = () => {
  const { id: jobId } = useParams()
  const [job, setJob] = useState(null)
  const [applications, setApplications] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState("all") // all, pending, reviewed, interviewed, rejected, hired
  const [selectedApplication, setSelectedApplication] = useState(null)
  const [feedbackModal, setFeedbackModal] = useState(false)
  const [feedback, setFeedback] = useState("")
  const [actionLoading, setActionLoading] = useState(false)

  useEffect(() => {
    fetchJobAndApplications()
  }, [jobId])

  const fetchJobAndApplications = async () => {
    try {
      setLoading(true)

      // Fetch job details
      const jobResponse = await fetch(`/api/jobs/${jobId}`, {
        credentials: "include",
      })

      if (!jobResponse.ok) {
        throw new Error("Failed to fetch job details")
      }

      const jobData = await jobResponse.json()
      setJob(jobData)

      // Fetch applications for this job
      const applicationsResponse = await fetch(`/api/jobs/${jobId}/applications`, {
        credentials: "include",
      })

      if (!applicationsResponse.ok) {
        throw new Error("Failed to fetch applications")
      }

      const applicationsData = await applicationsResponse.json()
      setApplications(applicationsData)
    } catch (err) {
      console.error("Error fetching data:", err)
      setError(err.message || "An error occurred while fetching data")
    } finally {
      setLoading(false)
    }
  }

  const updateApplicationStatus = async (applicationId, newStatus, feedbackText = "") => {
    try {
      setActionLoading(true)

      const response = await fetch(`/api/applications/${applicationId}/status`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          status: newStatus,
          feedback: feedbackText,
        }),
        credentials: "include",
      })

      if (!response.ok) {
        throw new Error("Failed to update application status")
      }

      // Update local state
      setApplications(
        applications.map((app) =>
          app.id === applicationId ? { ...app, status: newStatus, feedback: feedbackText || app.feedback } : app,
        ),
      )

      // Close feedback modal if open
      setFeedbackModal(false)
      setFeedback("")
      setSelectedApplication(null)
    } catch (err) {
      console.error("Error updating application status:", err)
      alert(err.message || "An error occurred while updating the application")
    } finally {
      setActionLoading(false)
    }
  }

  const openFeedbackModal = (application) => {
    setSelectedApplication(application)
    setFeedback(application.feedback || "")
    setFeedbackModal(true)
  }

  const filteredApplications = applications.filter((app) => {
    if (filter === "all") return true
    return app.status === filter
  })

  // For demo purposes, let's create some mock data
  const mockJob = {
    id: Number(jobId),
    title: "Senior Frontend Developer",
    company: "TechCorp Inc.",
    location: "New York, NY",
    type: "Full-time",
    postedDate: "2023-05-15",
    applicationsCount: 5,
  }

  const mockApplications = [
    {
      id: 1,
      jobId: Number(jobId),
      applicantId: 101,
      applicantName: "John Smith",
      applicantEmail: "john.smith@example.com",
      applicantPhone: "+1 (555) 123-4567",
      resumeUrl: "/uploads/resume1.pdf",
      coverLetter:
        "I am excited to apply for this position with my 8 years of experience in frontend development. I have worked extensively with React, TypeScript, and modern frontend frameworks. My previous role at TechGiant involved leading a team of 5 developers to rebuild our customer-facing application, resulting in a 40% improvement in load times and a significant increase in user engagement.",
      status: "pending",
      appliedDate: "2023-05-20T10:30:00Z",
    },
    {
      id: 2,
      jobId: Number(jobId),
      applicantId: 102,
      applicantName: "Emily Johnson",
      applicantEmail: "emily.johnson@example.com",
      applicantPhone: "+1 (555) 234-5678",
      resumeUrl: "/uploads/resume2.pdf",
      coverLetter:
        "With 6 years of experience in frontend development and a focus on user experience, I believe I would be a great fit for this role. I've led the development of several high-traffic web applications and have a strong background in optimizing performance and accessibility.",
      status: "reviewed",
      appliedDate: "2023-05-19T14:45:00Z",
    },
    {
      id: 3,
      jobId: Number(jobId),
      applicantId: 103,
      applicantName: "Michael Chen",
      applicantEmail: "michael.chen@example.com",
      applicantPhone: "+1 (555) 345-6789",
      resumeUrl: "/uploads/resume3.pdf",
      coverLetter:
        "I have been following your company for years and am impressed with your innovative approach to frontend development. With my 7 years of experience and expertise in React, Next.js, and state management libraries, I am confident I can contribute significantly to your team.",
      status: "interviewed",
      appliedDate: "2023-05-18T09:15:00Z",
      interviewDate: "2023-05-25T13:00:00Z",
      interviewNotes: "Strong technical skills, good cultural fit. Moving to second round.",
    },
    {
      id: 4,
      jobId: Number(jobId),
      applicantId: 104,
      applicantName: "Sarah Williams",
      applicantEmail: "sarah.williams@example.com",
      applicantPhone: "+1 (555) 456-7890",
      resumeUrl: "/uploads/resume4.pdf",
      coverLetter:
        "I am a frontend developer with 5 years of experience specializing in building responsive and accessible web applications. I have a passion for clean code and user-centric design, and I'm excited about the possibility of bringing my skills to your team.",
      status: "rejected",
      appliedDate: "2023-05-17T11:20:00Z",
      feedback: "Good candidate but looking for someone with more experience in state management libraries.",
    },
    {
      id: 5,
      jobId: Number(jobId),
      applicantId: 105,
      applicantName: "David Rodriguez",
      applicantEmail: "david.rodriguez@example.com",
      applicantPhone: "+1 (555) 567-8901",
      resumeUrl: "/uploads/resume5.pdf",
      coverLetter:
        "As a senior frontend developer with 10 years of experience across various industries, I bring a wealth of knowledge in building scalable and maintainable web applications. I'm particularly interested in your company's focus on innovative user experiences.",
      status: "hired",
      appliedDate: "2023-05-16T16:30:00Z",
      offerDetails: "Start date: June 15, 2023. Salary: $130,000/year.",
    },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Link to="/jobs/manage" className="text-blue-600 hover:text-blue-800 mr-4">
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <h1 className="text-2xl font-bold text-gray-900">Job Applications</h1>
            </div>
            <Link
              to={`/jobs/${jobId}/candidates`}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center"
            >
              <User className="h-5 w-5 mr-2" />
              Find Matching Candidates
            </Link>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
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
              onClick={fetchJobAndApplications}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        ) : (
          <>
            {/* Job Summary */}
            <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-2">{mockJob.title}</h2>
              <div className="flex flex-wrap gap-3 mb-4">
                <div className="flex items-center text-gray-600">
                  <Briefcase className="h-5 w-5 mr-1 text-gray-500" />
                  <span>{mockJob.company}</span>
                </div>
                <div className="flex items-center text-gray-600">
                  <User className="h-5 w-5 mr-1 text-gray-500" />
                  <span>{mockApplications.length} Applicants</span>
                </div>
                <div className="flex items-center text-gray-600">
                  <Calendar className="h-5 w-5 mr-1 text-gray-500" />
                  <span>Posted: {new Date(mockJob.postedDate).toLocaleDateString()}</span>
                </div>
              </div>
              <div className="flex space-x-3">
                <Link
                  to={`/jobs/${jobId}`}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                >
                  View Job Posting
                </Link>
                <Link
                  to={`/jobs/${jobId}/edit`}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                >
                  Edit Job
                </Link>
              </div>
            </div>

            {/* Filter tabs */}
            <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setFilter("all")}
                  className={`px-4 py-2 rounded-md text-sm ${
                    filter === "all" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  All Applications ({mockApplications.length})
                </button>
                <button
                  onClick={() => setFilter("pending")}
                  className={`px-4 py-2 rounded-md text-sm flex items-center ${
                    filter === "pending" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <Clock className="h-4 w-4 mr-2" />
                  Pending ({mockApplications.filter((a) => a.status === "pending").length})
                </button>
                <button
                  onClick={() => setFilter("reviewed")}
                  className={`px-4 py-2 rounded-md text-sm flex items-center ${
                    filter === "reviewed" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Reviewed ({mockApplications.filter((a) => a.status === "reviewed").length})
                </button>
                <button
                  onClick={() => setFilter("interviewed")}
                  className={`px-4 py-2 rounded-md text-sm flex items-center ${
                    filter === "interviewed" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <UserCheck className="h-4 w-4 mr-2" />
                  Interviewed ({mockApplications.filter((a) => a.status === "interviewed").length})
                </button>
                <button
                  onClick={() => setFilter("rejected")}
                  className={`px-4 py-2 rounded-md text-sm flex items-center ${
                    filter === "rejected" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Rejected ({mockApplications.filter((a) => a.status === "rejected").length})
                </button>
                <button
                  onClick={() => setFilter("hired")}
                  className={`px-4 py-2 rounded-md text-sm flex items-center ${
                    filter === "hired" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <Briefcase className="h-4 w-4 mr-2" />
                  Hired ({mockApplications.filter((a) => a.status === "hired").length})
                </button>
              </div>
            </div>

            {/* Applications list */}
            {mockApplications.length === 0 ? (
              <div className="bg-white rounded-lg shadow-sm p-8 text-center">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Applications Yet</h3>
                <p className="text-gray-600 mb-6">This job posting hasn't received any applications yet.</p>
              </div>
            ) : (
              <div className="space-y-6">
                {mockApplications
                  .filter((app) => filter === "all" || app.status === filter)
                  .map((application) => (
                    <div key={application.id} className="bg-white rounded-lg shadow-sm p-6">
                      <div className="flex flex-col md:flex-row md:items-start">
                        <div className="flex-grow">
                          <div className="flex items-center mb-3">
                            <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl mr-4">
                              {application.applicantName.charAt(0)}
                            </div>
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900">{application.applicantName}</h3>
                              <p className="text-gray-600">
                                Applied: {new Date(application.appliedDate).toLocaleDateString()}
                              </p>
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                            <div className="flex items-center">
                              <Mail className="h-5 w-5 text-gray-500 mr-2" />
                              <a
                                href={`mailto:${application.applicantEmail}`}
                                className="text-blue-600 hover:underline"
                              >
                                {application.applicantEmail}
                              </a>
                            </div>
                            <div className="flex items-center">
                              <Phone className="h-5 w-5 text-gray-500 mr-2" />
                              <a href={`tel:${application.applicantPhone}`} className="text-blue-600 hover:underline">
                                {application.applicantPhone}
                              </a>
                            </div>
                          </div>

                          <div className="mt-4">
                            <h4 className="font-medium text-gray-900 mb-2">Cover Letter</h4>
                            <div className="bg-gray-50 p-4 rounded-md text-gray-700 text-sm">
                              <p>{application.coverLetter}</p>
                            </div>
                          </div>

                          {application.status === "interviewed" && application.interviewNotes && (
                            <div className="mt-4">
                              <h4 className="font-medium text-gray-900 mb-2">Interview Notes</h4>
                              <div className="bg-blue-50 p-4 rounded-md text-blue-700 text-sm">
                                <p>{application.interviewNotes}</p>
                              </div>
                            </div>
                          )}

                          {application.status === "rejected" && application.feedback && (
                            <div className="mt-4">
                              <h4 className="font-medium text-gray-900 mb-2">Rejection Feedback</h4>
                              <div className="bg-red-50 p-4 rounded-md text-red-700 text-sm">
                                <p>{application.feedback}</p>
                              </div>
                            </div>
                          )}

                          {application.status === "hired" && application.offerDetails && (
                            <div className="mt-4">
                              <h4 className="font-medium text-gray-900 mb-2">Offer Details</h4>
                              <div className="bg-green-50 p-4 rounded-md text-green-700 text-sm">
                                <p>{application.offerDetails}</p>
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="mt-6 md:mt-0 md:ml-6 flex flex-col space-y-3">
                          <div className="mb-2">
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                application.status === "pending"
                                  ? "bg-yellow-100 text-yellow-800"
                                  : application.status === "reviewed"
                                    ? "bg-blue-100 text-blue-800"
                                    : application.status === "interviewed"
                                      ? "bg-purple-100 text-purple-800"
                                      : application.status === "rejected"
                                        ? "bg-red-100 text-red-800"
                                        : "bg-green-100 text-green-800"
                              }`}
                            >
                              {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                            </span>
                          </div>

                          <a
                            href={application.resumeUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors flex items-center"
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Download Resume
                          </a>

                          {application.status === "pending" && (
                            <>
                              <button
                                onClick={() => updateApplicationStatus(application.id, "reviewed")}
                                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center"
                              >
                                <CheckCircle className="h-4 w-4 mr-2" />
                                Mark as Reviewed
                              </button>
                              <button
                                onClick={() => openFeedbackModal(application)}
                                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center"
                              >
                                <XCircle className="h-4 w-4 mr-2" />
                                Reject
                              </button>
                            </>
                          )}

                          {application.status === "reviewed" && (
                            <>
                              <button
                                onClick={() => updateApplicationStatus(application.id, "interviewed")}
                                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors flex items-center"
                              >
                                <UserCheck className="h-4 w-4 mr-2" />
                                Move to Interview
                              </button>
                              <button
                                onClick={() => openFeedbackModal(application)}
                                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center"
                              >
                                <XCircle className="h-4 w-4 mr-2" />
                                Reject
                              </button>
                            </>
                          )}

                          {application.status === "interviewed" && (
                            <>
                              <button
                                onClick={() => updateApplicationStatus(application.id, "hired")}
                                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors flex items-center"
                              >
                                <Briefcase className="h-4 w-4 mr-2" />
                                Hire Candidate
                              </button>
                              <button
                                onClick={() => openFeedbackModal(application)}
                                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center"
                              >
                                <XCircle className="h-4 w-4 mr-2" />
                                Reject
                              </button>
                            </>
                          )}

                          <button
                            onClick={() => (window.location.href = `mailto:${application.applicantEmail}`)}
                            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors flex items-center"
                          >
                            <MessageSquare className="h-4 w-4 mr-2" />
                            Contact Applicant
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </>
        )}
      </main>

      {/* Feedback Modal */}
      {feedbackModal && selectedApplication && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Reject Application - {selectedApplication.applicantName}
            </h3>
            <p className="text-gray-600 mb-4">Please provide feedback for the candidate (optional):</p>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 mb-4"
              rows={4}
              placeholder="Explain why this candidate is not a good fit..."
            />
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => {
                  setFeedbackModal(false)
                  setSelectedApplication(null)
                  setFeedback("")
                }}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors"
                disabled={actionLoading}
              >
                Cancel
              </button>
              <button
                onClick={() => updateApplicationStatus(selectedApplication.id, "rejected", feedback)}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center"
                disabled={actionLoading}
              >
                {actionLoading ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 mr-2" />
                    Reject Application
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default JobApplications

