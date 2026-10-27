import {chart} from './charts.js';
const $ = id => document.getElementById(id);
const state = {config: null, generation: null, forward: null, frame: 0, timer: null, tab: 'generation', inspectionRequest: 0};
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
async function request(path, params = {}) {
  const response = await fetch(path + (Object.keys(params).length ? '?' + new URLSearchParams(params) : ''));
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Check the input values.`);
  }
  return response.json();
}
function uint(id, max = 4294967295) {
  const input = $(id), value = input.valueAsNumber;
  if (!Number.isInteger(value) || value < 0 || value > max) { input.focus(); throw new Error(`Enter a whole number between 0 and ${max}.`); }
  return value;
}
function showImage(id, src) { $(id).src = src; $(id).hidden = false; }
function pause() { clearInterval(state.timer); state.timer = null; $('play').textContent = '▶'; $('play').setAttribute('aria-label', 'Play denoising animation'); }
function showFrame(index) {
  if (!state.generation) return;
  const frames = state.generation.frames;
  state.frame = Math.max(0, Math.min(index, frames.length - 1));
  const frame = frames[state.frame], steps = state.generation.steps;
  showImage('generation-image', frame.image);
  showImage('clean-preview', frame.clean);
  $('generation-placeholder').hidden = true;
  $('timeline').value = state.frame;
  $('depth-readout').textContent = `k = ${frame.t} / ${steps}`;
  $('stat-depth').textContent = `${frame.t} / ${steps}`;
  const alpha = frame.t === 0 ? 1 : state.config?.schedules[state.generation.schedule].alpha_bar[frame.t - 1];
  $('stat-alpha').textContent = alpha == null ? '—' : alpha.toFixed(5);
  $('signal-fill').style.width = `${(alpha || 0) * 100}%`;
  $('generation-progress').textContent = `${Math.round((steps - frame.t) / steps * 100)}% DENOISED`;
  $('stage-phase').textContent = frame.t === 0 ? 'A sample from the learned distribution' : frame.t === steps ? 'Starting from Gaussian noise' : 'Predict noise. Take one reverse step.';
  if (state.forward) {
    const matching = state.forward.frames.reduce((best, f) => Math.abs(f.t - frame.t) < Math.abs(best.t - frame.t) ? f : best);
    showImage('forward-current', matching.image);
    $('forward-caption').textContent = `Forward: k = ${matching.t}`;
  }
}
function play() {
  if (!state.generation) return;
  if (state.timer) {pause(); return;}
  if (state.frame === state.generation.frames.length - 1) showFrame(0);
  $('play').textContent = 'Ⅱ'; $('play').setAttribute('aria-label', 'Pause denoising animation');
  state.timer = setInterval(() => {
    showFrame(state.frame + 1);
    if (state.frame === state.generation.frames.length - 1) pause();
  }, Number($('speed').value));
}
async function generate() {
  const button = $('generate');
  try {
    const seed = uint('seed'), schedule = $('schedule').value;
    button.disabled = true; button.textContent = 'Running diffusion…'; notice(); pause();
    const generation = await request('/api/generate', {seed, schedule});
    state.generation = generation;
    state.forward = null;
    $('forward-original').removeAttribute('src'); $('forward-current').removeAttribute('src');
    $('timeline').max = generation.frames.length - 1;
    for (const id of ['timeline', 'play', 'restart']) $(id).disabled = false;
    $('generation-source').textContent = `${generation.cached ? 'EXPORTED' : 'LIVE'} · ${generation.checkpoint_step.toLocaleString()} UPDATES`;
    showFrame(0);
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) play();
    try {
      const forward = await request('/api/forward', {seed, schedule, index: 0});
      if (state.generation !== generation) return;
      state.forward = forward;
      showImage('forward-original', forward.frames[0].image);
      $('forward-label').textContent = `${forward.label.toUpperCase()} · TEST #0`;
      showFrame(state.frame);
    } catch (error) {notice(`Generation ready. Forward comparison unavailable: ${error.message}`);}
  } catch (error) { notice(error.message); }
  finally { button.disabled = false; button.innerHTML = 'Generate a garment <span>↗</span>'; }
}
async function inspect() {
  const button = $('inspect'), requestId = ++state.inspectionRequest;
  try {
    const index = uint('inspect-index', 9999), seed = uint('inspect-seed');
    const schedule = $('inspect-schedule').value, depth = Number($('inspect-depth').value);
    button.disabled = true; button.textContent = 'Inspecting…'; notice();
    const result = await request('/api/inspect', {index, seed, schedule, depth});
    if (requestId !== state.inspectionRequest) return;
    for (const key of ['original', 'noisy', 'actual_noise', 'predicted_noise', 'estimated_clean']) showImage('micro-' + key, result[key]);
    $('noise-mse').textContent = result.noise_mse.toFixed(5);
    $('clean-mse').textContent = result.clean_mse.toFixed(5);
    $('inspect-alpha').textContent = result.alpha_bar.toFixed(5);
    $('raw-clean-mse').textContent = `Unclipped clean MSE: ${result.clean_mse_raw.toFixed(5)}`;
    $('signal-coefficient').textContent = Math.sqrt(result.alpha_bar).toFixed(4);
    $('noise-coefficient').textContent = Math.sqrt(1 - result.alpha_bar).toFixed(4);
    $('inspection-label').textContent = `${result.label} · test #${result.index} · noise seed ${seed} · depth ${depth}. Noise: blue = negative, orange = positive; shared fixed scale −3 to +3.`;
  } catch (error) {notice(error.message);}
  finally {button.disabled = false; button.textContent = 'Inspect prediction ↗';}
}
function switchTab(name) {
  state.tab = name; notice(); pause();
  for (const button of document.querySelectorAll('[data-tab]')) {
    const selected = button.dataset.tab === name;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(button.dataset.tab + '-panel').hidden = !selected;
  }
}
for (const button of document.querySelectorAll('[data-tab]')) {
  button.addEventListener('click', () => switchTab(button.dataset.tab));
  button.addEventListener('keydown', e => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const tabs = [...document.querySelectorAll('[data-tab]')], i = tabs.indexOf(button);
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    switchTab(tabs[next].dataset.tab); tabs[next].focus();
  });
}
for (const button of document.querySelectorAll('[data-seed]')) button.addEventListener('click', () => {
  $('seed').value = button.dataset.seed;
  document.querySelectorAll('[data-seed]').forEach(b => b.classList.toggle('selected', b === button));
});
$('seed').addEventListener('input', () => document.querySelectorAll('[data-seed]').forEach(b => b.classList.toggle('selected', b.dataset.seed === $('seed').value)));
$('random-seed').addEventListener('click', () => { $('seed').value = crypto.getRandomValues(new Uint32Array(1))[0]; $('seed').dispatchEvent(new Event('input')); });
$('generate').addEventListener('click', generate);
$('seed').addEventListener('keydown', e => {if (e.key === 'Enter' && !$('generate').disabled) generate();});
$('play').addEventListener('click', play);
$('restart').addEventListener('click', () => {pause(); showFrame(0);});
$('timeline').addEventListener('input', () => {pause(); showFrame(Number($('timeline').value));});
$('speed').addEventListener('change', () => {if (state.timer) {pause(); play();}});
$('inspect').addEventListener('click', inspect);
function updateDepth() {
  const max = state.config?.schedules[$('inspect-schedule').value].steps || 200;
  $('inspect-depth').max = max;
  $('inspect-depth-value').textContent = `${$('inspect-depth').value} / ${max}`;
}
$('inspect-depth').addEventListener('input', updateDepth);
$('inspect-schedule').addEventListener('change', updateDepth);
document.addEventListener('visibilitychange', () => {if (document.hidden) pause();});
async function init() {
  for (const name of ['training-chart', 'timestep-chart', 'matched-chart']) chart(name, []);
  const [health, config, metrics] = await Promise.allSettled([request('/api/health'), request('/api/config'), request('/api/results')]);
  if (health.status === 'fulfilled') {
    const h = health.value, ready = h.checkpoints.linear || h.checkpoints.cosine || h.results;
    $('model-status').textContent = ready ? h.checkpoints.linear || h.checkpoints.cosine ? '● Trained model connected' : '● Recorded model results' : '○ Awaiting trained checkpoints';
    $('model-status').classList.toggle('ready', ready);
    if (!ready) notice('No trained checkpoints or exported results are available yet. Train the two schedules and run the evaluation to populate this lab.');
  } else {$('model-status').textContent = '○ API unavailable'; notice('Cannot reach the model API. Start the app with scripts/serve.sh.');}
  if (config.status === 'fulfilled') {
    state.config = config.value; updateDepth();
    chart('schedule-chart', ['linear', 'cosine'].map(name => ({name, values: [{x: 0, y: 1}, ...config.value.schedules[name].alpha_bar.map((y, i) => ({x: i + 1, y}))]})), {xLabel: 'noise depth k', yMax: 1});
  }
  if (metrics.status === 'fulfilled') {
    const m = metrics.value, names = ['linear', 'cosine'];
    const verified = Object.values(m.controls_verified).every(Boolean);
    $('controls-status').textContent = verified ? 'PAIRED CONTROLS VERIFIED' : 'CONTROL MISMATCH';
    $('controls-status').style.color = verified ? 'var(--lime)' : 'var(--orange)';
    $('experiment-summary').replaceChildren();
    const summary = [[m.schedules.linear.step.toLocaleString(), 'Updates per schedule'], [m.schedules.linear.parameters.toLocaleString(), 'U-Net parameters'], [m.protocol.test_samples.toLocaleString(), 'Held-out examples'], [String(m.protocol.replicates), 'Paired training seed']];
    for (const [value, label] of summary) {
      const node = document.createElement('div'); node.className = 'summary-stat';
      const strong = document.createElement('strong'); strong.textContent = value;
      const span = document.createElement('span'); span.textContent = label;
      node.append(strong, span); $('experiment-summary').append(node);
    }
    chart('training-chart', names.map(name => ({name, values: m.schedules[name].training.map(p => ({x: p.step, y: p.mse}))})), {xLabel: 'optimizer updates', yFormat: y => y.toFixed(3)});
    chart('timestep-chart', names.map(name => ({name, values: m.schedules[name].by_timestep.map(p => ({x: p.depth, y: p.noise_mse, error: p.noise_mse_se}))})), {xLabel: 'noise depth k', yFormat: y => y.toFixed(3)});
    chart('matched-chart', names.map(name => ({name, values: m.schedules[name].signal_matched.map(p => ({x: -Math.log10(p.target_alpha_bar), y: p.noise_mse, error: p.noise_mse_se, label: `ᾱ target ${p.target_alpha_bar}, achieved ${p.alpha_bar.toFixed(4)}, k=${p.depth}`}))})), {xLabel: 'target signal retained ᾱ (log scale)', xMax: 2, xTicks: [0, 1, 2], xFormat: x => (10 ** -x).toFixed(x === 0 ? 0 : x), yFormat: y => y.toFixed(3)});
    for (const name of names) $(name + '-grid').src = m.schedules[name].sample_grid;
    $('sample-comparison').hidden = false; $('metrics-download').hidden = false;
  } else {$('controls-status').textContent = 'RESULTS PENDING';}
}
init().catch(error => notice(error.message));
