# TaoMate-HD project page

Project page for **TaoMate-HD: Progressive-Resolution Autoregressive Audio-Video Generation**, by the Alibaba TaoLive AIGC Team.

- [Inference code](https://github.com/TaoLiveAIGC/TaoMate-HD/tree/main)
- [Preview weights](https://huggingface.co/TaoLiveAIGC/TaoMate-HD-preview)
- [Demonstration videos](https://huggingface.co/datasets/TaoLiveAIGC/TaoMate-HD-demos)

## Preview

```bash
git clone --branch gh-pages --single-branch \
  https://github.com/TaoLiveAIGC/TaoMate-HD.git TaoMate-HD-page
cd TaoMate-HD-page
python3 -m http.server 8086
```

Open `http://localhost:8086`. No build step or dependencies are required. The files can also be served by GitHub Pages or another static host.

## Contents

- `index.html`: introduction, news, method, results, and release links.
- `styles.css`: the paper's white, charcoal, and orange palette, with blue/orange/purple method stages.
- `data.js`: 29 selected clips and four hour-long scenes.
- `app.js`: inline playback, fullscreen controls, category filters, long-video navigation, and the scroll-driven method walkthrough.
- `assets/`: bundled video posters and the project icon.

Original videos are served from the public Hugging Face demonstration dataset. Videos autoplay muted; hovering enables the selected video's sound, and leaving mutes it. The city establishing shot is intentionally silent.

The layout is inspired by [MaineCoon](https://mainecoon.tech/), with an original implementation and TaoMate-HD's own media and text. Powered by MiniMax H3. Model use follows the MiniMax H3 Community License Agreement provided with the code and model release.
