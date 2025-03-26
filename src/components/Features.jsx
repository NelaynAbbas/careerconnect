import { Search, BrainCircuit, Clock, Filter, Bell, Shield } from "lucide-react"

const features = [
  {
    icon: <BrainCircuit size={24} className="text-blue-600" />,
    title: "AI-Powered Matching",
    description:
      "Our advanced NLP and ML algorithms analyze job descriptions and resumes to provide highly relevant matches.",
  },
  {
    icon: <Search size={24} className="text-blue-600" />,
    title: "Advanced Job Search",
    description:
      "Filter jobs by skills, location, salary range, experience level, and more to find your perfect match.",
  },
  {
    icon: <Clock size={24} className="text-blue-600" />,
    title: "Application Tracking",
    description: "Track all your job applications in one place and receive real-time updates on their status.",
  },
  {
    icon: <Filter size={24} className="text-blue-600" />,
    title: "Smart CV Generation",
    description:
      "Generate professional CVs and receive AI-powered suggestions to improve your skills for specific jobs.",
  },
  {
    icon: <Bell size={24} className="text-blue-600" />,
    title: "Real-time Notifications",
    description: "Get instant alerts for new job matches, application updates, and interview invitations.",
  },
  {
    icon: <Shield size={24} className="text-blue-600" />,
    title: "Data Security",
    description: "Your data is protected with encryption and complies with GDPR and data protection laws.",
  },
]

const Features = () => {
  return (
    <section id="features" className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Powerful Features</h2>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto">
            CareerConnect offers innovative tools to streamline the job search and hiring process.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
              <div className="mb-4 bg-blue-50 p-3 rounded-full w-fit">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-gray-700">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Features

