import { FileText, Search, CheckCircle, Calendar } from "lucide-react"

const steps = [
  {
    icon: <FileText size={24} className="text-white" />,
    title: "Create Your Profile",
    description:
      "Sign up and create your profile or upload your resume. Our AI will analyze your skills and experience.",
  },
  {
    icon: <Search size={24} className="text-white" />,
    title: "Get Personalized Matches",
    description: "Receive AI-powered job recommendations tailored to your skills, experience, and preferences.",
  },
  {
    icon: <CheckCircle size={24} className="text-white" />,
    title: "Apply with Ease",
    description: "Apply to jobs with a single click and track your applications in real-time.",
  },
  {
    icon: <Calendar size={24} className="text-white" />,
    title: "Interview and Connect",
    description: "Schedule interviews and connect with employers directly through the platform.",
  },
]

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-16 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto">
            Finding your dream job or the perfect candidate has never been easier.
          </p>
        </div>

        <div className="flex flex-col md:flex-row justify-between items-start gap-8">
          {steps.map((step, index) => (
            <div key={index} className="flex flex-col items-center text-center max-w-xs mx-auto">
              <div className="bg-blue-600 rounded-full p-4 mb-4">{step.icon}</div>
              <div className="h-0.5 w-full bg-blue-200 my-2 hidden md:block"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 mt-4">{step.title}</h3>
              <p className="text-gray-700">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default HowItWorks

