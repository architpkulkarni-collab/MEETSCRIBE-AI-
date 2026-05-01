/* ═══════════════════════════════════════════════════
   AirScribe – Landing Page JavaScript
   Navigation · Animations · Interactions
   ═══════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── CONFIG ──
  const APP_URL = 'AirScribe_v3.html';

  // ── PAGE LOADER ──
  function hideLoader() {
    const loader = document.getElementById('pageLoader');
    if (loader) {
      setTimeout(() => loader.classList.add('hidden'), 800);
    }
  }

  // ── NAVBAR SCROLL EFFECT ──
  function setupNavbar() {
    const navbar = document.getElementById('navbar');
    let ticking = false;

    window.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          navbar.classList.toggle('scrolled', window.scrollY > 40);
          ticking = false;
        });
        ticking = true;
      }
    });
  }

  // ── MOBILE MENU ──
  function setupMobileMenu() {
    const toggle = document.getElementById('navToggle');
    const links = document.getElementById('navLinks');

    if (toggle && links) {
      toggle.addEventListener('click', () => {
        links.classList.toggle('open');
        const icon = toggle.querySelector('i');
        if (links.classList.contains('open')) {
          icon.className = 'fa fa-times';
        } else {
          icon.className = 'fa fa-bars';
        }
      });

      // Close menu when clicking a link
      links.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
          links.classList.remove('open');
          const icon = toggle.querySelector('i');
          icon.className = 'fa fa-bars';
        });
      });
    }
  }

  // ── SMOOTH SCROLL ──
  function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', (e) => {
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  // ── SCROLL REVEAL ──
  function setupReveal() {
    const elements = document.querySelectorAll('.reveal');

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.12,
      rootMargin: '0px 0px -40px 0px'
    });

    elements.forEach(el => observer.observe(el));
  }

  // ── NAVIGATION LOGIC ──
  function navigateTo(view) {
    window.location.href = APP_URL + '?view=' + encodeURIComponent(view);
  }

  function setupNavigation() {
    // Workspace card clicks
    document.querySelectorAll('[data-view]').forEach(el => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        const view = el.getAttribute('data-view');
        navigateTo(view);
      });
    });
  }

  // ── STAGGERED CARD ANIMATIONS ──
  function setupStaggeredAnimations() {
    const cards = document.querySelectorAll('.workspace-card, .feature-card');
    cards.forEach((card, i) => {
      card.style.transitionDelay = (i * 0.08) + 's';
    });
  }

  // ── PARALLAX EFFECT ON HERO PREVIEW ──
  function setupParallax() {
    const preview = document.querySelector('.hero-preview');
    if (!preview || window.innerWidth < 1024) return;

    document.addEventListener('mousemove', (e) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 8;
      const y = (e.clientY / window.innerHeight - 0.5) * 5;
      preview.style.transform = `perspective(1000px) rotateY(${-5 + x}deg) rotateX(${2 + y}deg)`;
    });
  }

  // ── TYPING EFFECT (OPTIONAL) ──
  function setupTypingEffect() {
    const el = document.getElementById('heroTagline');
    if (!el) return;

    const words = ['Notes', 'Ideas', 'Knowledge', 'Thoughts'];
    let wordIndex = 0;
    let charIndex = 0;
    let isDeleting = false;

    function type() {
      const current = words[wordIndex];

      if (isDeleting) {
        el.textContent = current.substring(0, charIndex - 1);
        charIndex--;
      } else {
        el.textContent = current.substring(0, charIndex + 1);
        charIndex++;
      }

      let delay = isDeleting ? 50 : 100;

      if (!isDeleting && charIndex === current.length) {
        delay = 2000;
        isDeleting = true;
      } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        wordIndex = (wordIndex + 1) % words.length;
        delay = 400;
      }

      setTimeout(type, delay);
    }

    setTimeout(type, 1500);
  }

  // ── INIT ──
  function init() {
    hideLoader();
    setupNavbar();
    setupMobileMenu();
    setupSmoothScroll();
    setupReveal();
    setupNavigation();
    setupStaggeredAnimations();
    setupParallax();
    setupTypingEffect();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
