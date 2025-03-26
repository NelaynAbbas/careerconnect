import { ArrowRight } from "lucide-react"
import { Link } from "react-router-dom"

const CallToAction = () => {
  return (
    <section className="py-16 bg-blue-600">
      <div className="container mx-auto px-4 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">Ready to Transform Your Job Search?</h2>
        <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto">
          Join thousands of job seekers and employers already using CareerConnect to find their perfect match.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/signup"
            className="px-6 py-3 bg-white text-blue-600 rounded-md hover:bg-blue-50 transition-colors flex items-center justify-center"
          >
            Get Started <ArrowRight className="ml-2" size={18} />
          </Link>
          <Link
            to="/signin"
            className="px-6 py-3 border border-white text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Sign In
          </Link>
        </div>
      </div>
    </section>
  )
}

export default CallToAction

