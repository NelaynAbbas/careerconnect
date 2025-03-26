import { Briefcase, User } from "lucide-react"
import { Link } from "react-router-dom"

const UserTypes = () => {
  return (
    <section className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">For Job Seekers and Employers</h2>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto">
            CareerConnect offers tailored solutions for both sides of the hiring process.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* For Job Seekers */}
          <div id="for-job-seekers" className="bg-blue-50 p-8 rounded-lg">
            <div className="bg-blue-600 rounded-full p-3 w-fit mb-6">
              <User size={24} className="text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">For Job Seekers</h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">✓</span>
                <span>Personalized job recommendations based on your skills</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">✓</span>
                <span>AI-powered CV generation and improvement suggestions</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">✓</span>
                <span>Course recommendations to enhance your skills</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">✓</span>
                <span>Real-time application tracking and status updates</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">✓</span>
                <span>Advanced filtering to find the perfect job match</span>
              </li>
            </ul>
            <Link
              to="/signup"
              className="mt-8 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors inline-block"
            >
              Find Jobs Now
            </Link>
          </div>

          {/* For Employers */}
          <div id="for-employers" className="bg-indigo-50 p-8 rounded-lg">
            <div className="bg-indigo-600 rounded-full p-3 w-fit mb-6">
              <Briefcase size={24} className="text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">For Employers</h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">✓</span>
                <span>AI-powered candidate ranking and matching</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">✓</span>
                <span>Automated resume parsing and skill analysis</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">✓</span>
                <span>Comprehensive employer dashboard for job management</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">✓</span>
                <span>Streamlined interview scheduling and candidate communication</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">✓</span>
                <span>Data-driven insights on recruitment performance</span>
              </li>
            </ul>
            <Link
              to="/signup"
              className="mt-8 px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors inline-block"
            >
              Post a Job
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}

export default UserTypes

