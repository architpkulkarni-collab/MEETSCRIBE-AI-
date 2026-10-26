/* ══════════════════════════════════════════════════════════════════
   MEETSCRIBE AI — THE LIVING DIARY
   Landing Page Interactive Controller
   Scroll Interpolation · Travelling Pen · SVG Waveform Animation
   Vanilla JS · Zero Dependencies · High Performance
   ══════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // Check reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── 1. TRAVELLING FOUNTAIN PEN CONTROLLER ──
  function setupTravellingPen() {
    const pen = document.getElementById('travellingPen');
    if (!pen || prefersReducedMotion) {
      if (pen) pen.classList.add('pen-hidden');
      return;
    }

    const heroSection = document.getElementById('hero');
    const argumentSection = document.getElementById('argument');
    const demoSection = document.getElementById('demonstration');

    if (!heroSection || !argumentSection || !demoSection) return;

    let ticking = false;

    // Linear interpolation helper
    function lerp(start, end, t) {
      return start + (end - start) * t;
    }

    // Clamp helper
    function clamp(val, min, max) {
      return Math.min(Math.max(val, min), max);
    }

    function updatePenPosition() {
      // If mobile or viewport width <= 768px, hide pen
      if (window.innerWidth <= 768) {
        pen.classList.add('pen-hidden');
        ticking = false;
        return;
      }

      const scrollY = window.pageYOffset || document.documentElement.scrollTop;
      const heroRect = heroSection.getBoundingClientRect();
      const heroTop = scrollY + heroRect.top;
      const heroHeight = heroSection.offsetHeight;

      const argRect = argumentSection.getBoundingClientRect();
      const argTop = scrollY + argRect.top;
      const argHeight = argumentSection.offsetHeight;

      const demoRect = demoSection.getBoundingClientRect();
      const demoTop = scrollY + demoRect.top;
      const demoHeight = demoSection.offsetHeight;

      // Overall travel range: from top of page to end of demonstration
      const travelStart = 0;
      const travelEnd = demoTop + demoHeight * 0.75;
      const fadeStart = demoTop + demoHeight * 0.45;

      if (scrollY > travelEnd) {
        pen.classList.add('pen-hidden');
        ticking = false;
        return;
      }

      pen.classList.remove('pen-hidden');

      // Define 3 Waypoint Stages:
      // Stage 0: Hero (scrollY from 0 to argTop)
      // Stage 1: Argument (scrollY from argTop to demoTop)
      // Stage 2: Demonstration (scrollY from demoTop to travelEnd)

      let currentX, currentY, currentRot, currentScale, currentOpacity;

      if (scrollY <= argTop) {
        // Between Hero and Argument entry
        const progress = clamp(scrollY / Math.max(1, argTop), 0, 1);
        currentX = lerp(62, 46, progress);       // vw
        currentY = lerp(20, 36, progress);       // vh
        currentRot = lerp(-16, -28, progress);   // deg
        currentScale = lerp(1.0, 0.92, progress);
        currentOpacity = 1;
      } else if (scrollY <= demoTop) {
        // Between Argument and Demonstration entry
        const progress = clamp((scrollY - argTop) / Math.max(1, demoTop - argTop), 0, 1);
        currentX = lerp(46, 22, progress);       // moves toward left where SVG ink starts
        currentY = lerp(36, 48, progress);
        currentRot = lerp(-28, -38, progress);
        currentScale = lerp(0.92, 0.78, progress);
        currentOpacity = 1;
      } else {
        // Within Demonstration Section
        const progress = clamp((scrollY - demoTop) / Math.max(1, travelEnd - demoTop), 0, 1);
        currentX = lerp(22, 18, progress);
        currentY = lerp(48, 52, progress);
        currentRot = lerp(-38, -42, progress);
        currentScale = 0.76;

        // Fade out as it completes the journey past the demonstration panel
        if (scrollY >= fadeStart) {
          const fadeProgress = clamp((scrollY - fadeStart) / Math.max(1, travelEnd - fadeStart), 0, 1);
          currentOpacity = lerp(1, 0, fadeProgress);
        } else {
          currentOpacity = 1;
        }
      }

      // Apply transform using viewport units
      pen.style.transform = `translate3d(${currentX}vw, ${currentY}vh, 0) rotate(${currentRot}deg) scale(${currentScale})`;
      pen.style.opacity = currentOpacity.toFixed(3);

      if (currentOpacity <= 0.01) {
        pen.classList.add('pen-hidden');
      } else {
        pen.classList.remove('pen-hidden');
      }

      ticking = false;
    }

    function onScroll() {
      if (!ticking) {
        requestAnimationFrame(updatePenPosition);
        ticking = true;
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    updatePenPosition();
  }

  // ── 2. HERO SPREAD WORDMARK SCROLL EFFECT ──
  function setupWordmarkSpread() {
    if (prefersReducedMotion) return;

    const heroWordmark = document.getElementById('heroWordmark');
    const bottomWordmark = document.getElementById('bottomWordmark');
    const heroSection = document.getElementById('hero');

    if (!heroWordmark || !heroSection) return;

    const heroChars = heroWordmark.querySelectorAll('.char');
    const bottomChars = bottomWordmark ? bottomWordmark.querySelectorAll('.char') : [];
    const centerIdx = 4.5; // (10 characters: 0 to 9)

    let ticking = false;

    function updateWordmarks() {
      const scrollY = window.pageYOffset || document.documentElement.scrollTop;
      const heroHeight = heroSection.offsetHeight;

      // Hero wordmark spread and sink
      if (scrollY <= heroHeight * 1.2) {
        const p = Math.min(Math.max(scrollY / (heroHeight * 0.9), 0), 1);
        heroChars.forEach((char, i) => {
          const dist = i - centerIdx;
          const tx = dist * p * 32; // Spread outward from center
          const ty = p * 42;        // Sink downward
          char.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
        });
      }

      // Bottom wordmark spread and sink as it approaches view
      if (bottomWordmark && bottomChars.length > 0) {
        const bottomRect = bottomWordmark.getBoundingClientRect();
        const windowHeight = window.innerHeight;
        if (bottomRect.top < windowHeight && bottomRect.bottom > 0) {
          const visibleProgress = Math.max(0, (windowHeight - bottomRect.top) / windowHeight);
          const p = Math.min(visibleProgress, 1);
          bottomChars.forEach((char, i) => {
            const dist = i - centerIdx;
            const tx = dist * (1 - p) * 24;
            const ty = (1 - p) * 36;
            char.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
          });
        }
      }

      ticking = false;
    }

    function onScroll() {
      if (!ticking) {
        requestAnimationFrame(updateWordmarks);
        ticking = true;
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    updateWordmarks();
  }

  // ── 3. SIGNATURE DEMONSTRATION: SVG DRAW-ON-ENTRY ANIMATION ──
  function setupDemonstrationAnimation() {
    const demoPanel = document.getElementById('demoPanel');
    const strokePaths = document.querySelectorAll('.animate-stroke');

    if (!demoPanel || strokePaths.length === 0) return;

    // Initialize stroke lengths
    strokePaths.forEach(path => {
      if (path.getTotalLength) {
        const length = path.getTotalLength();
        path.style.strokeDasharray = `${length} ${length}`;
        if (!prefersReducedMotion) {
          path.style.strokeDashoffset = length;
        } else {
          path.style.strokeDashoffset = '0';
        }
      }
    });

    if (prefersReducedMotion) return;

    // Trigger draw once on entry
    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          strokePaths.forEach((path, idx) => {
            const delay = idx * 280;
            setTimeout(() => {
              path.style.transition = 'stroke-dashoffset 2.1s cubic-bezier(0.25, 1, 0.5, 1)';
              path.style.strokeDashoffset = '0';
            }, delay);
          });
          obs.unobserve(entry.target); // Runs exactly once per page load
        }
      });
    }, {
      root: null,
      threshold: 0.15,
      rootMargin: '0px 0px -50px 0px'
    });

    observer.observe(demoPanel);
  }

  // ── 4. DEMONSTRATION VARIANT PICKER ──
  function setupVariantPicker() {
    const buttons = document.querySelectorAll('.ms-variant-btn');
    const demoSvg = document.getElementById('demoSvg');
    const descEl = document.getElementById('demoVariantDesc');
    const factEl = document.getElementById('demoVariantFact');

    if (!buttons.length || !demoSvg || !descEl || !factEl) return;

    const variantData = {
      voice: {
        className: 'mode-voice',
        desc: 'A conversation begins as sound. Every spoken phrase carries cadence, tone, and acoustic rhythm before becoming text.',
        fact: 'INPUT / SPOKEN WORDS'
      },
      transcript: {
        className: 'mode-transcript',
        desc: 'Spoken words become a readable record. Acoustic sound patterns resolve into faithful text lines with word-level precision.',
        fact: 'OUTPUT / WRITTEN TEXT'
      },
      summary: {
        className: 'mode-summary',
        desc: 'Important information becomes easier to revisit. Key points, decisions, and follow-ups are synthesized for quick retrieval.',
        fact: 'OUTPUT / ORGANIZED KNOWLEDGE'
      }
    };

    buttons.forEach(btn => {
      btn.addEventListener('click', function () {
        const variant = this.getAttribute('data-variant');
        if (!variant || !variantData[variant]) return;

        // Update button states
        buttons.forEach(b => b.setAttribute('aria-pressed', 'false'));
        this.setAttribute('aria-pressed', 'true');

        // Update SVG mode class
        demoSvg.classList.remove('mode-voice', 'mode-transcript', 'mode-summary');
        demoSvg.classList.add(variantData[variant].className);

        // Update descriptions with fade
        descEl.style.opacity = '0';
        setTimeout(() => {
          descEl.textContent = variantData[variant].desc;
          factEl.textContent = variantData[variant].fact;
          descEl.style.opacity = '1';
        }, 150);
      });
    });
  }

  // ── 5. SCROLL ENTRY REVEALS ──
  function setupEntryReveals() {
    const revealEls = document.querySelectorAll('.reveal-entry');
    if (!revealEls.length) return;

    if (!('IntersectionObserver' in window) || prefersReducedMotion) {
      revealEls.forEach(el => el.classList.add('is-revealed'));
      return;
    }

    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-revealed');
          obs.unobserve(entry.target);
        }
      });
    }, {
      root: null,
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    revealEls.forEach(el => observer.observe(el));
  }

  // ── 6. SMOOTH ANCHOR NAVIGATION (58PX OFFSET) ──
  function setupAnchorScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (!href || href === '#') return;

        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          const navOffset = 58;
          const elementPosition = target.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - navOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: prefersReducedMotion ? 'auto' : 'smooth'
          });

          // Close mobile nav if open
          const navMenu = document.getElementById('navCenter');
          if (navMenu) navMenu.classList.remove('is-open');
        }
      });
    });
  }

  // ── 7. MOBILE NAVIGATION TOGGLE ──
  function setupMobileNav() {
    const toggleBtn = document.getElementById('mobileNavToggle');
    const navMenu = document.getElementById('navCenter');
    if (!toggleBtn || !navMenu) return;

    toggleBtn.addEventListener('click', () => {
      const isOpen = navMenu.classList.toggle('is-open');
      toggleBtn.textContent = isOpen ? 'CLOSE' : 'MENU';
      toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });
  }

  // ── INITIALIZATION ──
  function init() {
    setupTravellingPen();
    setupWordmarkSpread();
    setupDemonstrationAnimation();
    setupVariantPicker();
    setupEntryReveals();
    setupAnchorScroll();
    setupMobileNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
