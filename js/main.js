window.onload = function() {
    initWindow();
    make_carousel('results-pca', pca_carousel_item_template, pca_items, 1, 1);

    // Safari blocks script-triggered video.play() on first page load until
    // the user has interacted with the page at least once, even for muted
    // videos. Retry on the first scroll/touch/click so autoplaying videos
    // (e.g. the PCA carousel) still start for first-time visitors.
    function retryAutoplayVideos() {
        document.querySelectorAll('video[autoplay]').forEach(video => {
            if (video.paused) video.play().catch(() => {});
        });
    }
    ['scroll', 'touchstart', 'pointerdown', 'keydown'].forEach(evt => {
        document.addEventListener(evt, retryAutoplayVideos, { once: true, passive: true });
    });
};
