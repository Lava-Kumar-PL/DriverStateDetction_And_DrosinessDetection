document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        threshold: 0.1 // Trigger when 10% of the element is visible
    };

    const observerCallback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            } else {
                entry.target.classList.remove('visible');
            }
        });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    const elements = document.querySelectorAll('.animate');
    elements.forEach(el => observer.observe(el));
});
