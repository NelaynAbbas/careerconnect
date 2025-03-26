"use client"

import { useState, useEffect } from "react"
import { Link, useParams } from "react-router-dom"
import { ArrowLeft, User, MapPin, Briefcase, AlertCircle, ChevronDown, ChevronUp } from "lucide-react"

const CandidateRecommendations = () => {
  const { id: jobId } = useParams()
  const [job, setJob] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedCandidates, setExpandedCandidates] = useState({})

  useEffect(() => {
    fetchJobAndCandidates()
  }, [jobId])

  const fetchJobAndCandidates = async () => {
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

      // Fetch candidate recommendations
      const candidatesResponse = await fetch(`/api/recommendations/candidates/${jobId}`, {
        credentials: "include",
      })

      if (!candidatesResponse.ok) {
        throw new Error("Failed to fetch candidate recommendations")
      }

      const candidatesData = await candidatesResponse.json()
      setCandidates(candidatesData)
    } catch (err) {
      console.error("Error fetching data:", err)
      setError(err.message || "An error occurred while fetching data")
    } finally {
      setLoading(false)
    }
  }

  const toggleCandidateExpand = (candidateId) => {
    setExpandedCandidates((prev) => ({
      ...prev,
      [candidateId]: !prev[candidateId],
    }))
  }

  // For demo purposes, let's create some mock data
  const mockJob = {
    id: Number(jobId),
    title: "Senior Frontend Developer",
    company: "TechCorp Inc.",
    location: "New York, NY",
    type: "Full-time",
  }

  const mockCandidates = [
    {
      id: 1,
      fullName: "John Smith",
      title: "Frontend Developer",
      location: "New York, NY",
      skills: ["JavaScript", "React", "TypeScript", "HTML", "CSS", "Redux", "Node.js", "GraphQL"],
      matchScore: 92,
    },
    {
      id: 2,
      fullName: "Emily Johnson",
      title: "Senior Software Engineer",
      location: "Remote",
      skills: ["JavaScript", "React", "Angular", "Vue.js", "HTML", "CSS", "Sass", "Webpack"],
      matchScore: 85,
    },
    {
      id: 3,
      fullName: "Michael Chen",
      title: "Frontend Engineer",
      location: "San Francisco, CA",
      skills: ["JavaScript", "React", "TypeScript", "Redux", "Jest", "Cypress", "Tailwind CSS"],
      matchScore: 78,
    },
    {
      id: 4,
      fullName: "Sarah Williams",
      title: "UI Developer",
      location: "Chicago, IL",
      skills: ["JavaScript", "React", "CSS", "Sass", "UI/UX Design", "Figma", "Adobe XD"],
      matchScore: 74,
    },
    {
      id: 5,
      fullName: "David Rodriguez",
      title: "Web Developer",
      location: "Miami, FL",
      skills: ["JavaScript", "React", "Angular", "HTML", "CSS", "Bootstrap", "jQuery", "PHP"],
      matchScore: 67,
    },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center">
            <Link to="/jobs/manage" className="text-blue-600 hover:text-blue-800 mr-4">
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Recommended Candidates</h1>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
            <p className="mt-2 text-gray-600">Finding candidates that match your job requirements...</p>
          </div>
        ) : error ? (
          <div className="bg-white rounded-lg shadow-sm p-8 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-500 mb-4">{error}</p>
            <button
              onClick={fetchJobAndCandidates}
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
                  <MapPin className="h-5 w-5 mr-1 text-gray-500" />
                  <span>{mockJob.location}</span>
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
                  to={`/jobs/${jobId}/applications`}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                >
                  View Applications
                </Link>
              </div>
            </div>

            {/* Candidates list */}
            {mockCandidates.length === 0 ? (
              <div className="bg-white rounded-lg shadow-sm p-8 text-center">
                <User className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Matching Candidates</h3>
                <p className="text-gray-600 mb-6">We couldn't find any candidates that match your job requirements.</p>
              </div>
            ) : (
              <div className="space-y-6">
                <p className="text-gray-700">Showing {mockCandidates.length} candidates sorted by match score</p>

                {mockCandidates.map((candidate) => (
                  <div key={candidate.id} className="bg-white rounded-lg shadow-sm overflow-hidden">
                    <div className="p-6">
                      <div className="flex flex-col md:flex-row md:items-start">
                        <div className="flex-grow">
                          <div className="flex items-center mb-3">
                            <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl mr-4">
                              {candidate.fullName.charAt(0)}
                            </div>
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900">{candidate.fullName}</h3>
                              <p className="text-gray-600">{candidate.title}</p>
                            </div>
                          </div>

                          <div className="flex items-center text-gray-600 mb-3">
                            <MapPin className="h-4 w-4 mr-1 text-gray-500" />
                            <span>{candidate.location}</span>
                          </div>

                          <div className="mb-4">
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Skills</h4>
                            <div className="flex flex-wrap gap-2">
                              {candidate.skills.map((skill, index) => (
                                <span
                                  key={index}
                                  className="bg-blue-100 text-blue-800 px-2.5 py-0.5 rounded-full text-xs font-medium"
                                >
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="mt-6 md:mt-0 md:ml-6 flex flex-col items-center">
                          <div className="flex items-center mb-2">
                            <div className="w-full bg-gray-200 rounded-full h-2.5 w-24 mr-2">
                              <div
                                className={`h-2.5 rounded-full ${
                                  candidate.matchScore >= 80
                                    ? "bg-green-500"
                                    : candidate.matchScore >= 60
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                                style={{ width: `${candidate.matchScore}%` }}
                              ></div>
                            </div>
                            <span className="text-lg font-semibold">{candidate.matchScore}%</span>
                          </div>
                          <span
                            className={`text-sm ${
                              candidate.matchScore >= 80
                                ? "text-green-600"
                                : candidate.matchScore >= 60
                                  ? "text-yellow-600"
                                  : "text-red-600"
                            }`}
                          >
                            {candidate.matchScore >= 80
                              ? "Excellent Match"
                              : candidate.matchScore >= 60
                                ? "Good Match"
                                : "Fair Match"}
                          </span>

                          <button
                            onClick={() => toggleCandidateExpand(candidate.id)}
                            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center"
                          >
                            {expandedCandidates[candidate.id] ? (
                              <>
                                Less Details <ChevronUp className="ml-1 h-4 w-4" />
                              </>
                            ) : (
                              <>
                                More Details <ChevronDown className="ml-1 h-4 w-4" />
                              </>
                            )}
                          </button>
                        </div>
                      </div>

                      {expandedCandidates[candidate.id] && (
                        <div className="mt-4 pt-4 border-t border-gray-200">
                          <div className="space-y-4">
                            <div>
                              <h4 className="text-sm font-medium text-gray-700 mb-2">Why this candidate matches</h4>
                              <ul className="list-disc pl-5 space-y-1 text-gray-600 text-sm">
                                <li>Has {candidate.skills.length} relevant skills for this position</li>
                                <li>
                                  Located in{" "}
                                  {candidate.location === mockJob.location
                                    ? "the same location as the job"
                                    : "a different location"}
                                </li>
                                <li>Current title indicates appropriate experience level</li>
                              </ul>
                            </div>

                            <div className="flex justify-end space-x-3">
                              <a
                                href={`mailto:candidate${candidate.id}@example.com`}
                                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                              >
                                Contact Candidate
                              </a>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default CandidateRecommendations

