const gallery = document.querySelector('#demo-gallery');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const visibleVideos = new Set();
const previewVideos = [];
let selectedLong = 0;
let requestedTime = 0;

function playPreview(video) {
  video.play().catch(error => {
    // A browser may require a click before an autoplaying video can become audible.
    if (error.name === 'NotAllowedError' && !video.muted) {
      video.muted = true;
      syncPreview(video);
    }
  });
}

function syncPreview(video) {
  if (document.hidden) {
    video.muted = true;
    video.pause();
    return;
  }
  if (document.fullscreenElement === video || video.webkitDisplayingFullscreen) return;
  if (!visibleVideos.has(video) || reducedMotion.matches) {
    video.muted = true;
    video.pause();
    return;
  }
  if (!video.hasAttribute('src')) video.src = video.dataset.src;
  playPreview(video);
}

function enableSound(video) {
  previewVideos.forEach(other => { other.muted = other !== video || other.dataset.silent === 'true'; });
  playPreview(video);
}

const previewObserver = new IntersectionObserver(entries => {
  entries.forEach(({target, intersectionRatio}) => {
    if (intersectionRatio >= .15) visibleVideos.add(target);
    else visibleVideos.delete(target);
    syncPreview(target);
  });
}, {threshold: [0, .15]});

function mountPreview(player, {title, poster, src, silent = false}) {
  player.classList.add('inline-player');
  const video = document.createElement('video');
  video.muted = video.defaultMuted = true;
  video.autoplay = video.loop = video.playsInline = true;
  video.preload = 'none';
  video.poster = poster;
  video.dataset.src = src;
  video.dataset.silent = silent;
  video.setAttribute('aria-label', title);
  const expand = document.createElement('button');
  expand.type = 'button';
  expand.className = 'media-expand';
  expand.setAttribute('aria-label', `View ${title} fullscreen`);
  expand.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/></svg>';
  player.append(video, expand);
  player.addEventListener('pointerenter', event => {
    if (event.pointerType === 'touch') return;
    if (visibleVideos.has(video) && !reducedMotion.matches) enableSound(video);
  });
  player.addEventListener('pointerleave', () => {
    if (document.fullscreenElement !== video && !video.webkitDisplayingFullscreen) {
      video.muted = true;
      syncPreview(video);
    }
  });
  player.addEventListener('pointerdown', () => {
    if (video.controls) return;
    if (!video.hasAttribute('src')) video.src = video.dataset.src;
    enableSound(video);
  });
  expand.addEventListener('click', () => {
    if (!video.hasAttribute('src')) video.src = video.dataset.src;
    video.controls = true;
    enableSound(video);
    if (video.requestFullscreen) video.requestFullscreen().catch(() => { video.controls = false; });
    else video.webkitEnterFullscreen();
  });
  video.addEventListener('webkitendfullscreen', () => {
    video.controls = false;
    video.muted = true;
    syncPreview(video);
  });
  previewVideos.push(video);
  previewObserver.observe(video);
  return video;
}

function createDemoCard(demo) {
  const card = document.createElement('div');
  card.className = 'demo-card';
  card.dataset.demo = demo.id;
  card.dataset.category = demo.category;
  const player = document.createElement('div');
  player.className = 'demo-image';
  mountPreview(player, {...demo, src: demo.url});
  card.append(player);
  return card;
}

const filters = document.querySelector('#demo-filters');
for (const category of [{id: 'all', title: 'All'}, ...CATEGORIES]) {
  const button = document.createElement('button');
  button.dataset.filter = category.id;
  button.textContent = category.title;
  button.classList.toggle('active', category.id === 'all');
  button.setAttribute('aria-pressed', category.id === 'all');
  filters.append(button);
}

for (const portrait of [false, true]) {
  const orientation = portrait ? 'portrait' : 'landscape';
  const demos = CATEGORIES.flatMap(category => DEMOS.filter(demo => demo.category === category.id && Boolean(demo.portrait) === portrait));
  const section = document.createElement('section');
  section.className = 'demo-group';
  section.dataset.orientation = orientation;
  section.setAttribute('aria-labelledby', `gallery-${orientation}`);
  section.innerHTML = `<div class="demo-group-heading"><h3 id="gallery-${orientation}">${portrait ? 'Portrait' : 'Landscape'}</h3><span>${demos.length} CLIPS</span></div>`;
  const grid = document.createElement('div');
  grid.className = `demo-grid ${orientation}-grid`;
  demos.forEach(demo => grid.append(createDemoCard(demo)));
  section.append(grid);
  gallery.append(section);
}
document.querySelector('#demo-count').textContent = `${DEMOS.length} CLIPS`;

document.querySelectorAll('[data-filter]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-filter]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', active);
    });
    gallery.querySelectorAll('.demo-group').forEach(group => {
      let count = 0;
      group.querySelectorAll('.demo-card').forEach(card => {
        card.hidden = button.dataset.filter !== 'all' && card.dataset.category !== button.dataset.filter;
        if (card.hidden) {
          const video = card.querySelector('video');
          visibleVideos.delete(video);
          syncPreview(video);
        }
        if (!card.hidden) count++;
      });
      group.hidden = count === 0;
      group.querySelector('.demo-group-heading > span').textContent = `${count} CLIPS`;
    });
    const count = DEMOS.filter(demo => button.dataset.filter === 'all' || demo.category === button.dataset.filter).length;
    document.querySelector('#demo-count').textContent = `${count} CLIPS`;
  });
});

const heroBackground = document.querySelector('#hero-background-video');
heroBackground.dataset.src = MEDIA_ROOT + 'hour-dance.mp4';
heroBackground.dataset.silent = 'true';
previewVideos.push(heroBackground);
previewObserver.observe(heroBackground);

mountPreview(document.querySelector('.hero-main'), {
  title: 'Origami animation', poster: 'assets/dragon.jpg', src: MEDIA_ROOT + 'hero-origami.mp4'
});
document.querySelectorAll('.hero-card[data-demo]').forEach(player => {
  const demo = DEMOS.find(item => item.id === player.dataset.demo);
  mountPreview(player, {...demo, src: demo.url});
});

const longVideo = mountPreview(document.querySelector('.long-video-wrap'), {
  title: `One-hour video: ${LONG_SCENES[0].title}`,
  poster: LONG_SCENES[0].poster,
  src: LONG_SCENES[0].url
});
longVideo.id = 'long-video';
LONG_SCENES.forEach((scene, index) => {
  const button = document.createElement('button');
  button.innerHTML = `<small>0${index + 1}</small>${scene.title}`;
  button.classList.toggle('active', index === 0);
  button.setAttribute('aria-pressed', index === 0);
  button.addEventListener('click', () => {
    selectedLong = index;
    requestedTime = 0;
    longVideo.pause();
    longVideo.muted = true;
    longVideo.removeAttribute('src');
    longVideo.load();
    longVideo.dataset.src = scene.url;
    longVideo.poster = scene.poster;
    longVideo.setAttribute('aria-label', `One-hour video: ${scene.title}`);
    longVideo.parentElement.querySelector('.media-expand').setAttribute('aria-label', `View ${scene.title} fullscreen`);
    syncPreview(longVideo);
    document.querySelectorAll('#long-scenes button').forEach(item => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-pressed', item === button);
    });
    document.querySelectorAll('[data-time]').forEach(item => item.classList.remove('active'));
  });
  document.querySelector('#long-scenes').append(button);
});

function playLong(time = 0) {
  requestedTime = time;
  if (!longVideo.hasAttribute('src')) {
    longVideo.src = LONG_SCENES[selectedLong].url;
  } else if (longVideo.readyState >= 1) {
    longVideo.currentTime = time;
  }
  playPreview(longVideo);
}
longVideo.addEventListener('loadedmetadata', () => { longVideo.currentTime = requestedTime; });
document.querySelectorAll('[data-time]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-time]').forEach(item => item.classList.toggle('active', item === button));
    playLong(Number(button.dataset.time));
  });
});

document.addEventListener('fullscreenchange', () => {
  previewVideos.forEach(video => {
    video.controls = document.fullscreenElement === video;
    if (!video.controls) {
      video.muted = true;
      syncPreview(video);
    }
  });
});
document.addEventListener('visibilitychange', () => {
  previewVideos.forEach(syncPreview);
});
reducedMotion.addEventListener('change', () => previewVideos.forEach(syncPreview));

const methodStory = document.querySelector('#method-story');
const methodDiagram = methodStory.querySelector('.method-diagram');
const methodPhases = methodStory.querySelectorAll('[data-phase]');
const methodProgress = methodStory.querySelectorAll('.method-progress i');
const methodStages = [
  ['Start with the big picture.', 'The first 480p step establishes coarse visual structure while generating audio.'],
  ['Give the scene its shape.', 'A second 480p step develops motion and composition along the same audiovisual trajectory.'],
  ['A finer canvas. The same trajectory.', 'The coupled-noise lift carries video onto a finer spatial grid. Audio keeps its latent geometry.'],
  ['Bring the details into focus.', 'The final denoising evaluation refines the picture at 768p, with audio still generated jointly.'],
  ['Keep the stream moving.', 'Decode video as chunks arrive; decode accumulated audio as a complete sequence. Each stage carries its clean AV history forward.']
];
let activeMethodStage = -1;
let methodFramePending = false;

function updateMethodStory() {
  methodFramePending = false;
  if (!methodStory.classList.contains('is-animated')) return;
  const travel = methodStory.offsetHeight - methodDiagram.offsetHeight;
  const offset = parseFloat(getComputedStyle(methodDiagram).top) - methodStory.getBoundingClientRect().top;
  const progress = Math.max(0, Math.min(1, offset / travel));
  const stage = Math.min(methodStages.length - 1, Math.floor(progress * methodStages.length));
  methodProgress.forEach((bar, index) => {
    bar.style.setProperty('--fill', Math.max(0, Math.min(1, progress * methodStages.length - index)));
  });
  if (stage === activeMethodStage) return;
  activeMethodStage = stage;
  methodPhases.forEach(element => {
    element.classList.toggle('is-reached', Number(element.dataset.phase) <= stage);
    element.classList.toggle('is-current', Number(element.dataset.phase) === stage);
  });
  methodStory.querySelector('.method-story-number').textContent = String(stage + 1).padStart(2, '0');
  document.querySelector('#method-stage-title').textContent = methodStages[stage][0];
  document.querySelector('#method-stage-detail').textContent = methodStages[stage][1];
}

function sizeMethodStory() {
  methodStory.classList.toggle('is-animated', !reducedMotion.matches);
  methodStory.style.setProperty('--panel-height', `${methodDiagram.offsetHeight}px`);
  if (innerHeight < methodDiagram.offsetHeight + 48) methodStory.classList.remove('is-animated');
  activeMethodStage = -1;
  updateMethodStory();
}

addEventListener('scroll', () => {
  if (!methodFramePending) {
    methodFramePending = true;
    requestAnimationFrame(updateMethodStory);
  }
}, {passive: true});
addEventListener('resize', sizeMethodStory);
reducedMotion.addEventListener('change', sizeMethodStory);
document.fonts.ready.then(sizeMethodStory);
sizeMethodStory();
