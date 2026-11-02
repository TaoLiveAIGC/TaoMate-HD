// Original video demonstrations hosted on Hugging Face.
const MEDIA_ROOT = 'https://huggingface.co/datasets/TaoLiveAIGC/TaoMate-HD-demos/resolve/main/videos/';
const CATEGORIES = [
  {id: 'commerce', title: 'Products & interiors'},
  {id: 'speech', title: 'People & dialogue'},
  {id: 'cinema', title: 'Cinematic scenes'},
  {id: 'environment', title: 'Environment & detail'},
  {id: 'animation', title: 'Imagined worlds'},
  {id: 'action', title: 'Action'}
];
const DEMOS = [
  { id: 'selected-002', title: 'Studio explainer', category: 'speech', label: 'Mandarin dialogue', duration: 10, portrait: true, poster: 'assets/selected/clip-002.jpg', url: MEDIA_ROOT + 'selected-002.mp4' },
  { id: 'selected-001', title: 'Drinkware presentation', category: 'commerce', label: 'Product presentation', duration: 10, portrait: true, poster: 'assets/selected/clip-001.jpg', url: MEDIA_ROOT + 'selected-001.mp4' },
  { id: 'selected-003', title: 'Indoor conversation', category: 'speech', label: 'Mandarin dialogue', duration: 10, portrait: true, poster: 'assets/selected/clip-003.jpg', url: MEDIA_ROOT + 'selected-003.mp4' },
  { id: 'selected-007', title: 'Travel commentary', category: 'speech', label: 'Mandarin dialogue', duration: 10, portrait: true, poster: 'assets/selected/clip-007.jpg', url: MEDIA_ROOT + 'selected-007.mp4' },
  { id: 'selected-045', title: 'Home presentation', category: 'commerce', label: 'Interior presentation', duration: 60, poster: 'assets/selected/clip-045.jpg', url: MEDIA_ROOT + 'selected-045.mp4' },
  { id: 'selected-004', title: 'Backpack presentation', category: 'commerce', label: 'Product presentation', duration: 5, portrait: true, poster: 'assets/selected/clip-004.jpg', url: MEDIA_ROOT + 'selected-004.mp4' },
  { id: 'selected-005', title: 'Stationery presentation', category: 'commerce', label: 'Product presentation', duration: 5, portrait: true, poster: 'assets/selected/clip-005.jpg', url: MEDIA_ROOT + 'selected-005.mp4' },
  { id: 'selected-006', title: 'Science explainer', category: 'speech', label: 'Mandarin dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-006.jpg', url: MEDIA_ROOT + 'selected-006.mp4' },
  { id: 'selected-008', title: 'Outdoor commentary', category: 'speech', label: 'Mandarin dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-008.jpg', url: MEDIA_ROOT + 'selected-008.mp4' },
  { id: 'selected-009', title: 'Seated conversation', category: 'speech', label: 'Mandarin dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-009.jpg', url: MEDIA_ROOT + 'selected-009.mp4' },
  { id: 'selected-011', title: 'Natural-light dialogue', category: 'speech', label: 'English dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-011.jpg', url: MEDIA_ROOT + 'selected-011.mp4' },
  { id: 'selected-012', title: 'Studio conversation', category: 'speech', label: 'English dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-012.jpg', url: MEDIA_ROOT + 'selected-012.mp4' },
  { id: 'selected-014', title: 'Urban establishing shot', category: 'environment', label: 'Cityscape', duration: 5, silent: true, poster: 'assets/selected/clip-014.jpg', url: MEDIA_ROOT + 'selected-014.mp4' },
  { id: 'selected-015', title: 'Platform scene', category: 'cinema', label: 'Cinematic scene', duration: 5, poster: 'assets/selected/clip-015.jpg', url: MEDIA_ROOT + 'selected-015.mp4' },
  { id: 'selected-016', title: 'Street close-up', category: 'cinema', label: 'Cinematic dialogue', duration: 5, poster: 'assets/selected/clip-016.jpg', url: MEDIA_ROOT + 'selected-016.mp4' },
  { id: 'selected-022', title: 'Spacecraft interior', category: 'cinema', label: 'Science fiction', duration: 5, poster: 'assets/selected/clip-022.jpg', url: MEDIA_ROOT + 'selected-022.mp4' },
  { id: 'selected-023', title: 'Ceramics presentation', category: 'commerce', label: 'Product presentation', duration: 5, portrait: true, poster: 'assets/selected/clip-023.jpg', url: MEDIA_ROOT + 'selected-023.mp4' },
  { id: 'selected-026', title: 'Bag presentation', category: 'commerce', label: 'Product presentation', duration: 5, portrait: true, poster: 'assets/selected/clip-026.jpg', url: MEDIA_ROOT + 'selected-026.mp4' },
  { id: 'selected-027', title: 'Casual conversation', category: 'speech', label: 'Mandarin dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-027.jpg', url: MEDIA_ROOT + 'selected-027.mp4' },
  { id: 'selected-028', title: 'Window-lit portrait', category: 'speech', label: 'Mandarin dialogue', duration: 5, portrait: true, poster: 'assets/selected/clip-028.jpg', url: MEDIA_ROOT + 'selected-028.mp4' },
  { id: 'selected-036', title: 'Backstage dialogue', category: 'cinema', label: 'Cinematic dialogue', duration: 5, poster: 'assets/selected/clip-036.jpg', url: MEDIA_ROOT + 'selected-036.mp4' },
  { id: 'selected-041', title: 'Domestic interior', category: 'environment', label: 'Everyday environment', duration: 5, poster: 'assets/selected/clip-041.jpg', url: MEDIA_ROOT + 'selected-041.mp4' },
  { id: 'selected-044', title: 'Botanical macro', category: 'environment', label: 'Nature detail', duration: 5, poster: 'assets/selected/clip-044.jpg', url: MEDIA_ROOT + 'selected-044.mp4' },
  { id: 'dragon', title: 'Origami animation', category: 'animation', label: 'Paper world', duration: 75, poster: 'assets/dragon.jpg', url: MEDIA_ROOT + 'dragon.mp4' },
  { id: 'commerce-serum', title: 'Skincare presentation', category: 'commerce', label: 'Product presentation', duration: 60, poster: 'assets/serum.jpg', portrait: true, url: MEDIA_ROOT + 'commerce-serum.mp4' },
  { id: 'rally', title: 'Rally racing', category: 'action', label: 'Rally racing', duration: 75, poster: 'assets/rally.jpg', url: MEDIA_ROOT + 'rally.mp4' },
  { id: 'skeleton', title: 'Character dance', category: 'animation', label: 'Character animation', duration: 75, poster: 'assets/skeleton.jpg', url: MEDIA_ROOT + 'skeleton.mp4' },
  { id: 'commerce-backpack', title: 'Backpack overview', category: 'commerce', label: 'Product presentation', duration: 60, poster: 'assets/backpack.jpg', portrait: true, url: MEDIA_ROOT + 'commerce-backpack.mp4' },
  { id: 'bear', title: 'City flight', category: 'action', label: 'Cinematic flight', duration: 75, poster: 'assets/bear.jpg', url: MEDIA_ROOT + 'bear.mp4' }
];
const LONG_SCENES = [
  {title:'The origami voyage', url:MEDIA_ROOT + 'hour-origami.mp4', poster:'assets/hour-dragon.jpg'},
  {title:'A natural expression', url:MEDIA_ROOT + 'hour-expression.mp4', poster:'assets/hour-expression.jpg'},
  {title:'Dancing into the sunset', url:MEDIA_ROOT + 'hour-dance.mp4', poster:'assets/hour-skeleton.jpg'},
  {title:'Around the next bend', url:MEDIA_ROOT + 'hour-rally.mp4', poster:'assets/hour-rally.jpg'}
];
