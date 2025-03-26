import { ArrowRight } from "lucide-react"
import { Link } from "react-router-dom"
import matching from "../assets/matching.svg"

const Hero = () => {
  return (
    <section className="bg-gradient-to-r from-blue-50 to-indigo-50 py-16 md:py-24">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              AI-Powered Job Matching for Better Careers
            </h1>
            <p className="text-xl text-gray-700 mb-8">
              CareerConnect uses advanced AI algorithms to match job seekers with the perfect opportunities based on
              skills, experience, and preferences.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link
                to="/jobs"
                className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center justify-center"
              >
                Find Jobs <ArrowRight className="ml-2" size={18} />
              </Link>
              <Link
                to="/jobs/post"
                className="px-6 py-3 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-600 hover:text-white transition duration-300"
              >
                Post a Job
              </Link>
            </div>
          </div>
          <div className="md:w-1/2 flex justify-center">
            <img
              src={matching}
              alt="AI Job Matching"
              className="max-w-full h-auto rounded-lg shadow-lg"
            />
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero

