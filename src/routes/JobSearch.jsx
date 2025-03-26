"use client"

import { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import {
  Briefcase,
  MapPin,
  DollarSign,
  Clock,
  Search,
  Filter,
  ChevronDown,
  Star,
  AlertCircle,
  Building,
} from "lucide-react"

const JobSearch = () => {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [filters, setFilters] = useState({
    jobType: [],
    location: [],
    experienceLevel: [],
    salary: "",
    remote: false,
  })
  const [showFilters, setShowFilters] = useState(false)
  const [userType, setUserType] = useState(null)

  useEffect(() => {
    // Check user type when component mounts
    const checkUserType = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/auth/status", {
          credentials: "include",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
          }
        })

        if (!response.ok) {
          throw new Error("Failed to check user status")
        }

        const data = await response.json()
        setUserType(data.user?.userType)
      } catch (err) {
        console.error("Error checking user type:", err)
        setError("Failed to verify user type")
      }
    }

    checkUserType()
  }, [])

  useEffect(() => {
    fetchJobs()
  }, [])

  const fetchJobs = async () => {
    try {
      setLoading(true)
      const response = await fetch("http://localhost:5000/api/jobs", {
        credentials: "include",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        }
      })

      if (!response.ok) {
        throw new Error("Failed to fetch jobs")
      }

      const data = await response.json()
      setJobs(data)
    } catch (err) {
      console.error("Error fetching jobs:", err)
      setError(err.message || "An error occurred while fetching jobs")
    } finally {
      setLoading(false)
    }
  }

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
      !job.company.toLowerCase().includes(searchTerm.toLowerCase()) &&
      !job.description.toLowerCase().includes(searchTerm.toLowerCase())
    ) {
      return false
    }

    // Job type filter
    if (filters.jobType.length > 0 && !filters.jobType.includes(job.type)) {
      return false
    }

    // Location filter
    if (
      filters.location.length > 0 &&
      !filters.location.some((loc) => job.location.toLowerCase().includes(loc.toLowerCase()))
    ) {
      return false
    }

    // Experience level filter
    if (filters.experienceLevel.length > 0 && !filters.experienceLevel.includes(job.experienceLevel)) {
      return false
    }

    // Remote filter
    if (filters.remote && !job.isRemote) {
      return false
    }

    return true
  })

  // Sort jobs by match score (highest first)
  const sortedJobs = [...filteredJobs].sort((a, b) => b.matchScore - a.matchScore)

  // If user is not a job seeker, show access denied message
  if (userType && userType !== 'jobSeeker') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-sm p-8 text-center max-w-md">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Access Denied</h2>
          <p className="text-gray-600 mb-4">
            Only job seekers can access the job search page. Please log in as a job seeker to continue.
          </p>
          <Link
            to="/signin"
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Sign In as Job Seeker
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Find Your Perfect Job</h1>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
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
                placeholder="Search jobs by title, company, or keywords"
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
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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

                {/* Experience Level Filter */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Experience Level</h3>
                  <div className="space-y-2">
                    {["Entry-level", "Mid-level", "Senior-level", "Executive"].map((level) => (
                      <label key={level} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={filters.experienceLevel.includes(level)}
                          onChange={() => toggleFilter("experienceLevel", level)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">{level}</span>
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

                  {/* Remote Only Toggle */}
                  <div className="mt-4">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={filters.remote}
                        onChange={(e) => setFilters({ ...filters, remote: e.target.checked })}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-700">Remote Only</span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="mt-4 flex justify-end">
                <button
                  onClick={() =>
                    setFilters({
                      jobType: [],
                      location: [],
                      experienceLevel: [],
                      salary: "",
                      remote: false,
                    })
                  }
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
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            {searchTerm || Object.values(filters).some((f) => (Array.isArray(f) ? f.length > 0 : f))
              ? "Search Results"
              : "Recommended Jobs"}
          </h2>

          {loading ? (
            <div className="text-center py-8">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
              <p className="mt-2 text-gray-600">Loading jobs...</p>
            </div>
          ) : error ? (
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <p className="text-red-500 mb-4">{error}</p>
              <button
                onClick={fetchJobs}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : sortedJobs.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <p className="text-gray-600">No jobs found matching your criteria.</p>
              <button
                onClick={() => {
                  setSearchTerm("")
                  setFilters({
                    jobType: [],
                    location: [],
                    experienceLevel: [],
                    salary: "",
                    remote: false,
                  })
                }}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Clear Filters
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {sortedJobs.map((job) => (
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
                        <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                          {job.experience_level}
                        </div>
                        <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          <Clock className="h-3 w-3 mr-1" />
                          {new Date(job.created_at).toLocaleDateString()}
                        </div>
                        {job.is_remote && (
                          <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                            Remote
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="mt-4 md:mt-0 flex flex-col items-end">
                      {job.matchScore && (
                        <div className="flex items-center mb-2">
                          <Star className="h-4 w-4 text-yellow-400 mr-1" />
                          <span className="text-sm font-medium">{job.matchScore}% Match</span>
                        </div>
                      )}
                      <Link
                        to={`/jobs/${job.id}`}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                      >
                        View Job
                      </Link>
                    </div>
                  </div>
                  <p className="mt-4 text-gray-600">{job.description}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default JobSearch

