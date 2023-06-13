// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();

    document.querySelector(this.getAttribute('href')).scrollIntoView({
      behavior: 'smooth'
    });
  });
});

// Form submission
document.querySelector('form').addEventListener('submit', function (e) {
  e.preventDefault();

  // Perform form validation or submission logic here

  // Clear form inputs
  this.reset();
});
