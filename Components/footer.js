class CustomFooter extends HTMLElement {
    connectedCallback() {
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    background: rgba(15, 23, 42, 0.8);
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                footer {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 3rem 2rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 2rem;
                }
                
                .footer-section {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .footer-title {
                    font-size: 1.2rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(to right, #6366f1, #8b5cf6);
                    -webkit-background-clip: text;
                    background-clip: text;
                    color: transparent;
                }
                
                .footer-links {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .footer-links a {
                    color: rgba(255, 255, 255, 0.7);
                    text-decoration: none;
                    transition: color 0.3s;
                }
                
                .footer-links a:hover {
                    color: white;
                }
                
                .social-links {
                    display: flex;
                    gap: 1rem;
                }
                
                .social-links a {
                    color: white;
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s;
                }
                
                .social-links a:hover {
                    background: linear-gradient(to right, #6366f1, #8b5cf6);
                    transform: translateY(-3px);
                }
                
                .copyright {
                    grid-column: 1 / -1;
                    text-align: center;
                    padding-top: 2rem;
                    margin-top: 2rem;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 0.9rem;
                }
                
                @media (max-width: 768px) {
                    footer {
                        grid-template-columns: 1fr;
                        text-align: center;
                    }
                    
                    .social-links {
                        justify-content: center;
                    }
                }
            </style>
            <footer>
                <div class="footer-section">
                    <div class="footer-title">RoomVision AI</div>
                    <p>Advanced AI for room classification and recognition.</p>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Quick Links</div>
                    <div class="footer-links">
                        <a href="index.html">Home</a>
                        <a href="train.html">Train Model</a>
                        <a href="#features">Features</a>
                        <a href="#how-it-works">How It Works</a>
                    </div>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Project Info</div>
                    <div class="footer-links">
                        <a href="#">Team Members</a>
                        <a href="#">Documentation</a>
                        <a href="#">GitHub Repository</a>
                        <a href="#">API Reference</a>
                    </div>
                </div>
                
                <div class="footer-section">
                    <div class="footer-title">Contact</div>
                    <div class="footer-links">
                        <a href="mailto:info@roomvision.ai">alessandramingi@gmail.com</a>
                        <a href="mailto:info@roomvision.ai">oliveriadanica@gmail.com</a>
                        <a href="#">Feedback</a>
                    </div>
                </div>
                
                <div class="copyright">
                    <p>© 2025 RoomVision AI | Artificial Intelligence | College of Information and Computing Sciences</p>
                    <p>Project by: Alessandra A. Mingi and Danica A. Oliveria</p>
                </div>
            </footer>
        `;
    }
}

customElements.define('custom-footer', CustomFooter);