class CustomNavbar extends HTMLElement {
    connectedCallback() {
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    position: fixed;
                    top: 0;
                    left: 0;
                    z-index: 1000;
                    background: rgba(15, 23, 42, 0.8);
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                nav {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1.5rem 2rem;
                    max-width: 1400px;
                    margin: 0 auto;
                    width: 100%;
                }
                
                .logo {
                    font-size: 1.5rem;
                    font-weight: 700;
                    background: linear-gradient(to right, #6366f1, #8b5cf6);
                    -webkit-background-clip: text;
                    background-clip: text;
                    color: transparent;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .nav-links {
                    display: flex;
                    gap: 2rem;
                }
                
                .nav-links a {
                    color: white;
                    text-decoration: none;
                    font-weight: 500;
                    transition: color 0.3s;
                    position: relative;
                }
                
                .nav-links a:hover {
                    color: #8b5cf6;
                }
                
                .nav-links a::after {
                    content: '';
                    position: absolute;
                    bottom: -5px;
                    left: 0;
                    width: 0;
                    height: 2px;
                    background: linear-gradient(to right, #6366f1, #8b5cf6);
                    transition: width 0.3s;
                }
                
                .nav-links a:hover::after {
                    width: 100%;
                }
                
                @media (max-width: 768px) {
                    nav {
                        flex-direction: column;
                        padding: 1rem;
                    }
                    
                    .nav-links {
                        margin-top: 1rem;
                        gap: 1rem;
                    }
                }
            </style>
            <nav>
                <a href="index.html" class="logo">
                    <i data-feather="box"></i>
                    RoomClassify
                </a>
                <div class="nav-links">
                    <a href="index.html">Home</a>
                    <a href="train.html">Train Model</a>
                    <a href="#features">Features</a>
                    <a href="#how-it-works">How It Works</a>
                </div>
            </nav>
        `;
    }
}

customElements.define('custom-navbar', CustomNavbar);