/* ── Nav scroll effect ── */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 20);
}, { passive: true });

/* ── Intersection Observer: fade-in ── */
const fadeEls = document.querySelectorAll('.fade-in');
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      // Stagger cards in a group
      const siblings = entry.target.parentElement.querySelectorAll('.fade-in');
      let delay = 0;
      siblings.forEach((sib, idx) => { if (sib === entry.target) delay = idx * 80; });
      setTimeout(() => entry.target.classList.add('visible'), delay);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

fadeEls.forEach(el => observer.observe(el));

/* ── Lightbox ── */
function openLightbox(src) {
  const overlay = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  img.src = src;
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const overlay = document.getElementById('lightbox');
  overlay.classList.remove('active');
  document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});

// Prevent click on image from closing
document.getElementById('lightbox-img').addEventListener('click', (e) => e.stopPropagation());

/* ── Demo video: check if file exists, show video or placeholder ── */
function checkDemoVideo() {
  const video = document.getElementById('demo-video');
  const placeholder = document.getElementById('demo-placeholder');
  if (!video) return;

  // Try to detect if the video source is valid by loading metadata
  const testVideo = document.createElement('video');
  testVideo.src = 'demo.mp4';
  testVideo.addEventListener('loadedmetadata', () => {
    // Video exists — swap placeholder for real player
    placeholder.style.display = 'none';
    video.style.display = 'block';
  });
  testVideo.addEventListener('error', () => {
    // Video not found — keep placeholder
  });
}

checkDemoVideo();

/* ── Smooth active nav highlight ── */
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(link => {
        link.style.color = link.getAttribute('href') === '#' + entry.target.id
          ? 'var(--cyan)'
          : '';
      });
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => sectionObserver.observe(s));

/* ── Play button: scroll to or open video ── */
const playBtn = document.getElementById('play-btn');
if (playBtn) {
  playBtn.addEventListener('click', () => {
    const video = document.getElementById('demo-video');
    if (video.style.display !== 'none') {
      video.play();
    }
  });
}
