// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Animation on scroll
function animateOnScroll() {
    const elements = document.querySelectorAll('.glass-panel, h2, h3');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fadeIn');
            }
        });
    }, { threshold: 0.1 });
    
    elements.forEach(element => {
        observer.observe(element);
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    animateOnScroll();
    feather.replace();
});

// Add animation class to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fadeIn {
        animation: fadeIn 0.6s ease-out forwards;
    }
`;
document.head.appendChild(style);


// Typing effect script 
const words = ["Classifier", "Identifier", "Analyzer"];
let i = 0;
let isDeleting = false;
let txt = '';
const speed = 100;

function type() {
    const current = i % words.length;
    const fullText = words[current];

    if (isDeleting) {
        txt = fullText.substring(0, txt.length - 1);
    } else {
        txt = fullText.substring(0, txt.length + 1);
    }

    document.getElementById("typing-text").textContent = txt;

    if (!isDeleting && txt === fullText) {
        setTimeout(() => isDeleting = true, 1500); // pause at full word
    } else if (isDeleting && txt === '') {
        isDeleting = false;
        i++;
    }

    setTimeout(type, isDeleting ? speed / 2 : speed);
}

type();