"use client"

import { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import { Briefcase, Clock, MapPin, Building, DollarSign, Star, Search, Filter, ChevronDown } from "lucide-react"

const Dashboard = ({ user }) => {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [filters, setFilters] = useState({
    jobType: [],
    location: [],
    salary: "",
  })
  const [showFilters, setShowFilters] = useState(false)

  // Add this inside the Dashboard component
  const [recommendedJobs, setRecommendedJobs] = useState([])
  const [recommendationsLoading, setRecommendationsLoading] = useState(true)

  useEffect(() => {
    // Fetch recommended jobs
    const fetchJobs = async () => {
      try {
        // In a real app, this would be an API call
        // Simulating API call with timeout
        await new Promise((resolve) => setTimeout(resolve, 1000))

        // Mock data
        const mockJobs = [
          {
            id: 1,
            title: "Senior Frontend Developer",
            company: "TechCorp Inc.",
            location: "Remote",
            salary: "$120,000 - $150,000",
            type: "Full-time",
            posted: "2 days ago",
            description: "We are looking for an experienced Frontend Developer with React expertise to join our team.",
            matchScore: 95,
          },
          {
            id: 2,
            title: "Full Stack Engineer",
            company: "InnovateSoft",
            location: "New York, NY",
            salary: "$130,000 - $160,000",
            type: "Full-time",
            posted: "1 week ago",
            description: "Join our team to build scalable web applications using modern technologies.",
            matchScore: 88,
          },
          {
            id: 3,
            title: "UX/UI Designer",
            company: "DesignHub",
            location: "San Francisco, CA",
            salary: "$110,000 - $140,000",
            type: "Full-time",
            posted: "3 days ago",
            description: "Create beautiful and intuitive user interfaces for our products.",
            matchScore: 82,
          },
          {
            id: 4,
            title: "DevOps Engineer",
            company: "CloudTech Solutions",
            location: "Remote",
            salary: "$125,000 - $155,000",
            type: "Contract",
            posted: "5 days ago",
            description: "Manage our cloud infrastructure and CI/CD pipelines.",
            matchScore: 79,
          },
          {
            id: 5,
            title: "Data Scientist",
            company: "DataInsights",
            location: "Boston, MA",
            salary: "$140,000 - $170,000",
            type: "Full-time",
            posted: "1 day ago",
            description: "Analyze large datasets and build machine learning models.",
            matchScore: 75,
          },
        ]

        setJobs(mockJobs)
      } catch (error) {
        console.error("Error fetching jobs:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchJobs()
  }, [])

  useEffect(() => {
    // Fetch job recommendations based on user skills
    const fetchRecommendations = async () => {
      try {
        setRecommendationsLoading(true)
        const response = await fetch("/api/recommendations/jobs", {
          credentials: "include",
        })

        if (!response.ok) {
          throw new Error("Failed to fetch job recommendations")
        }

        const data = await response.json()
        setRecommendedJobs(data)
      } catch (error) {
        console.error("Error fetching recommendations:", error)
      } finally {
        setRecommendationsLoading(false)
      }
    }

    // Only fetch recommendations if user is a job seeker
    if (user?.userType === "jobSeeker") {
      fetchRecommendations()
    }
  }, [user])

  const toggleFilter = (category, value) => {
    setFilters((prev) => {
      const current = [...prev[category]]
      if (current.includes(value)) {
        return {
          ...prev,
          [category]: current.filter((item) => item !== value),
        }
      } else {
        return {
          ...prev,
          [category]: [...current, value],
        }
      }
    })
  }

  const filteredJobs = jobs.filter((job) => {
    // Search term filter
    if (
      searchTerm &&
      !job.title.toLowerCase().includes(searchTerm.toLowerCase()) &&
      !job.company.toLowerCase().includes(searchTerm.toLowerCase())
    ) {
      return false
    }

    // Job type filter
    if (filters.jobType.length > 0 && !filters.jobType.includes(job.type)) {
      return false
    }

    // Location filter
    if (filters.location.length > 0 && !filters.location.some((loc) => job.location.includes(loc))) {
      return false
    }

    return true
  })

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="bg-blue-600 text-white rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-2">Welcome back, {user?.fullName || "there"}!</h2>
          <p className="mb-4">Your profile is 85% complete. Complete your profile to improve your job matches.</p>
          <Link
            to="/profile"
            className="inline-block px-4 py-2 bg-white text-blue-600 rounded-md hover:bg-blue-50 transition-colors"
          >
            Complete Profile
          </Link>
        </div>

        {/* Add this to the Dashboard component for job seekers */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">My Applications</h2>
            <Link to="/applications" className="text-blue-600 hover:text-blue-800">
              View All
            </Link>
          </div>
          {/* Display recent applications summary */}
          <div className="space-y-2">
            {/* This would show the most recent applications */}
            <p className="text-gray-600">You have 5 active job applications</p>
          </div>
        </div>

        {/* Add this to the Dashboard component for employers */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Recent Applications</h2>
          </div>
          {/* Display recent applications received */}
          <div className="space-y-2">
            <p className="text-gray-600">You have received 12 new applications across your job postings</p>
            <Link to="/jobs/manage" className="text-blue-600 hover:text-blue-800">
              View Your Job Postings
            </Link>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-grow">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                placeholder="Search jobs by title or company"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-2 border border-gray-300 rounded-md shadow-sm bg-white text-gray-700 hover:bg-gray-50 transition-colors flex items-center"
            >
              <Filter className="h-5 w-5 mr-2" />
              Filters
              <ChevronDown
                className={`h-4 w-4 ml-2 transition-transform ${showFilters ? "transform rotate-180" : ""}`}
              />
            </button>
          </div>

          {/* Filters */}
          {showFilters && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Job Type Filter */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Job Type</h3>
                  <div className="space-y-2">
                    {["Full-time", "Part-time", "Contract", "Freelance", "Internship"].map((type) => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={filters.jobType.includes(type)}
                          onChange={() => toggleFilter("jobType", type)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">{type}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Location Filter */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Location</h3>
                  <div className="space-y-2">
                    {["Remote", "New York", "San Francisco", "Boston", "Chicago"].map((location) => (
                      <label key={location} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={filters.location.includes(location)}
                          onChange={() => toggleFilter("location", location)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">{location}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Salary Filter */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Salary Range</h3>
                  <select
                    value={filters.salary}
                    onChange={(e) => setFilters({ ...filters, salary: e.target.value })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Any Salary</option>
                    <option value="0-50000">$0 - $50,000</option>
                    <option value="50000-100000">$50,000 - $100,000</option>
                    <option value="100000-150000">$100,000 - $150,000</option>
                    <option value="150000+">$150,000+</option>
                  </select>
                </div>
              </div>

              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => setFilters({ jobType: [], location: [], salary: "" })}
                  className="px-4 py-2 text-sm text-gray-700 hover:text-blue-600"
                >
                  Clear All Filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Job Listings */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Recommended Jobs</h2>
            <Link to="/jobs" className="text-blue-600 hover:text-blue-800">
              View All Jobs
            </Link>
          </div>

          {recommendationsLoading ? (
            <div className="text-center py-8">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
              <p className="mt-2 text-gray-600">Finding jobs that match your skills...</p>
            </div>
          ) : recommendedJobs.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <p className="text-gray-600">
                No job recommendations available. Add more skills to your profile to get personalized job matches.
              </p>
              <Link
                to="/profile"
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors inline-block"
              >
                Update Skills
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {recommendedJobs.slice(0, 3).map((job) => (
                <div key={job.id} className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow">
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{job.title}</h3>
                      <div className="mt-1 flex items-center text-gray-600">
                        <Building className="h-4 w-4 mr-1" />
                        {job.company}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          <Briefcase className="h-3 w-3 mr-1" />
                          {job.type}
                        </div>
                        <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          <MapPin className="h-3 w-3 mr-1" />
                          {job.location}
                        </div>
                        {job.salary && (
                          <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                            <DollarSign className="h-3 w-3 mr-1" />
                            {job.salary}
                          </div>
                        )}
                        <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          <Clock className="h-3 w-3 mr-1" />
                          {new Date(job.postedDate).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <div className="mt-4 md:mt-0 flex flex-col items-end">
                      <div className="flex items-center mb-2">
                        <div className="w-full bg-gray-200 rounded-full h-1.5 mr-2 w-16">
                          <div
                            className={`h-1.5 rounded-full ${
                              job.matchScore >= 80
                                ? "bg-green-500"
                                : job.matchScore >= 60
                                  ? "bg-yellow-500"
                                  : "bg-red-500"
                            }`}
                            style={{ width: `${job.matchScore}%` }}
                          ></div>
                        </div>
                        <Star className="h-4 w-4 text-yellow-400 mr-1" />
                        <span className="text-sm font-medium">{job.matchScore}% Match</span>
                      </div>
                      <Link
                        to={`/jobs/${job.id}`}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                      >
                        View Job
                      </Link>
                    </div>
                  </div>
                </div>
              ))}

              {recommendedJobs.length > 3 && (
                <div className="text-center mt-4">
                  <Link
                    to="/jobs"
                    className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50 transition-colors inline-block"
                  >
                    View More Recommendations
                  </Link>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default Dashboard

