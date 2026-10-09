function initSectionNav() {
    const nav = document.getElementById('section-nav');
    const teaser = document.getElementById('teaser');
    if (!nav) return;

    const titles = Array.from(document.querySelectorAll('.x-section-title'));
    if (titles.length === 0) return;

    titles.forEach((title, i) => {
        if (!title.id) title.id = 'section-nav-target-' + i;
        const link = document.createElement('a');
        link.href = '#' + title.id;
        link.className = 'section-nav-link';
        link.innerHTML = '<span class="section-nav-dot"></span><span class="section-nav-label"></span>';
        link.querySelector('.section-nav-label').textContent = title.textContent.trim();
        nav.appendChild(link);
    });

    const links = Array.from(nav.querySelectorAll('.section-nav-link'));

    function update() {
        if (teaser) {
            nav.classList.toggle('visible', teaser.getBoundingClientRect().bottom <= 0);
        }

        let activeIndex = 0;
        for (let i = 0; i < titles.length; i++) {
            if (titles[i].getBoundingClientRect().top <= window.innerHeight * 0.35) {
                activeIndex = i;
            }
        }
        links.forEach((link, i) => link.classList.toggle('active', i === activeIndex));
    }

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
}

document.addEventListener('DOMContentLoaded', initSectionNav);
