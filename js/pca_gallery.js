var pca_items = [
    {
        video: "nibio2_plot34_uls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["ULS"],
        alt: "NIBIO2 forest plot 34, ULS point cloud"
    },
    {
        video: "nibio2_plot32_uls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["ULS"],
        alt: "NIBIO2 forest plot 32, ULS point cloud"
    },
    {
        video: "culs_plot2_uls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["ULS"],
        alt: "CULS forest plot 2, ULS point cloud"
    },
    {
        video: "mls_burum_plot2_mls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["MLS"],
        alt: "NIBIO Burum forest plot 2, MLS point cloud"
    },
    {
        video: "rn_test4_tls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["TLS"],
        alt: "RN merged trees plot 4, TLS point cloud"
    },
    {
        video: "rn_test8_tls.mp4",
        video_scale: 1.25, video_offset_y: 0,
        tags: ["TLS"],
        alt: "RN merged trees plot 8, TLS point cloud"
    },
];


function pca_carousel_item_template(item) {
    let button = '';
    if (item.model) {
        button = `<div class="x-button" style="position: absolute; top: 10px; right: 10px" onclick=\'openWindow(pca_window_template(${JSON.stringify(item)}))\'>View GLB</div>`;
    }

    return `<div class="x-card" style="height: 70vh;">
                ${button}
                <div style="width: auto; height: 100%; max-width: 100%; aspect-ratio: 1; pointer-events: none;">
                    <video autoplay playsinline loop muted style="position: relative; width: 100%; height: 100%; z-index: -1; transform: scale(${item.video_scale}) translateY(${item.video_offset_y}px)" src="assets/pca_videos/${item.video}"></video>
                </div>
            </div>`;
}


function pca_window_template(item) {
    let panel = asset_panel_template('');
    item = JSON.parse(JSON.stringify(item));
    item.model = 'assets/pca_videos/' + item.model
    return modelviewer_window_template(item, panel, {viewer_size: 800, panel_size: 300});
}
