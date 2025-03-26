const testimonials = [
    {
      quote:
        "CareerConnect's AI matching technology helped me find a job that perfectly matches my skills and career goals.",
      name: "Sarah Johnson",
      title: "Software Developer",
      image: "/placeholder.svg?height=60&width=60",
    },
    {
      quote: "As a recruiter, I've cut my hiring time in half thanks to CareerConnect's AI-powered candidate ranking.",
      name: "Michael Chen",
      title: "HR Manager",
      image: "/placeholder.svg?height=60&width=60",
    },
    {
      quote: "The CV generation feature gave me professional templates and suggestions that improved my job prospects.",
      name: "Emily Rodriguez",
      title: "Marketing Specialist",
      image: "/placeholder.svg?height=60&width=60",
    },
  ]
  
  const Testimonials = () => {
    return (
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">What Our Users Say</h2>
            <p className="text-xl text-gray-700 max-w-3xl mx-auto">
              Hear from job seekers and employers who have found success with CareerConnect.
            </p>
          </div>
  
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md">
                <div className="mb-4">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className="text-yellow-400">
                      ★
                    </span>
                  ))}
                </div>
                <p className="text-gray-700 mb-6 italic">"{testimonial.quote}"</p>
                <div className="flex items-center">
                  <img
                    src={testimonial.image || "/placeholder.svg"}
                    alt={testimonial.name}
                    className="w-12 h-12 rounded-full mr-4"
                  />
                  <div>
                    <h4 className="font-semibold text-gray-900">{testimonial.name}</h4>
                    <p className="text-gray-600 text-sm">{testimonial.title}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    )
  }
  
  export default Testimonials
  
  