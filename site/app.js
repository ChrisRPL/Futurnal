const revealItems = document.querySelectorAll(".reveal");

const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    }
  },
  { threshold: 0.18 }
);

revealItems.forEach((item, i) => {
  item.style.transitionDelay = `${Math.min(i * 90, 260)}ms`;
  observer.observe(item);
});
