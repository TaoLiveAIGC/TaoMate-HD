# <img src="assets/taomate-hd-logo.gif" width="40" height="40" alt="" /> TaoMate-HD project page

A standalone research project page for TaoMate-HD. Plain HTML, CSS, and JavaScript; no build step or dependencies.

## Preview

```bash
git clone --branch gh-pages --single-branch https://github.com/TaoLiveAIGC/TaoMate-HD.git TaoMate-HD-page
cd TaoMate-HD-page
python3 -m http.server 8080
```

Open http://localhost:8080. This branch can also be served directly by GitHub Pages or any static host.

## Edit

- `index.html`: project introduction, method, results, and release resources.
- `styles.css`: paper-aligned light palette, responsive to desktop and mobile.
- `data.js`: demo titles and original media URLs.
- `app.js`: inline video previews, category filters, long-video time navigation, and the scroll-driven method walkthrough.
- `assets/`: video posters and logo files.

The page links to the [paper](https://arxiv.org/abs/2611.03394), [inference code](https://github.com/TaoLiveAIGC/TaoMate-HD), and [preview weights](https://huggingface.co/TaoLiveAIGC/TaoMate-HD-preview). The News section records the initial release on November 3, 2026. The citation block contains the paper's arXiv BibTeX.

Technical descriptions and comparison values follow the paper. The page reports the 1.60× matched spatial speedup, 1.92× optimized deployment speedup, and 11.18-point CER improvement, without showing specific FPS values. Accelerated inference decodes video in ordered windows and audio as a complete sequence.

Colors follow the public reading edition: white, charcoal `#252525`, orange `#CF4E08`, and warm panels `#FFF7F1`. The method walkthrough uses blue for LOW, orange for HIGH, and purple for audiovisual history, matching the paper's figures. Videos retain neutral dark backgrounds and high-contrast fullscreen controls.

The navigation, hero title, and footer use an animated SVG mark: a steady T with a short segment moving forward and returning as the next chunk. The favicon stays static, and reduced-motion preferences disable the mark's animation.

## Media and attribution

The gallery contains 29 clips: 23 selected additions and six earlier demos. Landscape videos appear together, followed by portrait videos; six category filters apply to both sections. Four hour-long videos remain in the long-form section. Descriptive clip names are used only for accessibility, not displayed below the videos.

Video cards show only the picture and a fullscreen button, without names, durations, or pop-up players. Visible videos autoplay muted and loop; hovering enables that video's audio, and leaving mutes it. Off-screen and filtered-out videos pause. Browsers may require one normal page interaction (such as selecting a category) before hover audio is allowed. Fullscreen uses the browser's video player; touch users can use its sound controls.

Videos are served from the public [TaoMate-HD demonstration dataset](https://huggingface.co/datasets/TaoLiveAIGC/TaoMate-HD-demos). The urban establishing shot is a video-only copy with its audio track removed; other previews retain original audio. The opening excerpt uses seconds 3–13 of the origami-dragon scene, including its matching original sound. The small landscape hero card shows rally racing.

The first-screen background uses the one-hour sunset dance video, muted and softly blurred beneath a light gradient that keeps the introduction readable. Like the other previews, it pauses off-screen and respects reduced-motion preferences.

The method diagram stays in view as scrolling reveals the two low-resolution steps, coupled-noise lift, high-resolution refinement, and audiovisual output/history. Scrolling back reverses the walkthrough. Reduced-motion preferences and short viewports show the complete static diagram instead.

The project-page structure is inspired by [MaineCoon](https://mainecoon.tech/), with an original implementation and TaoMate-HD's own media and text. Model use and redistribution follow the MiniMax H3 Community License; this page does not introduce different model-license terms.
