/* Small SVG charts with no build tooling or external runtime dependency. */
const NS = 'http://www.w3.org/2000/svg';
const COLORS = {linear: '#f69b79', cosine: '#beed87'};
function el(tag, attributes = {}, text = '') {
  const node = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([k, v]) => node.setAttribute(k, String(v)));
  if (text) node.textContent = text;
  return node;
}
export function chart(target, series, {xLabel = '', yMax, xMax, xTicks, xFormat = x => String(Math.round(x)), yFormat = y => y.toFixed(2), yMin = 0} = {}) {
  const host = document.getElementById(target);
  host.replaceChildren();
  const points = series.flatMap(s => s.values);
  if (!points.length) {host.textContent = 'Measurements available after training.'; return;}
  const W = 500, H = 225, L = 47, R = 16, T = 12, B = 38;
  xMax = xMax ?? Math.max(...points.map(p => p.x), 1);
  yMax = yMax ?? Math.max(...points.map(p => p.y + (p.error || 0)), 0.01) * 1.08;
  const X = x => L + x / xMax * (W - L - R);
  const Y = y => H - B - (y - yMin) / (yMax - yMin) * (H - B - T);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, 'aria-hidden': 'true'});
  for (let i = 0; i < 5; i++) {
    const y = yMin + (yMax - yMin) * i / 4;
    svg.append(el('line', {x1: L, x2: W - R, y1: Y(y), y2: Y(y), class: 'gridline'}));
    svg.append(el('text', {x: L - 9, y: Y(y) + 3, 'text-anchor': 'end'}, yFormat(y)));
  }
  for (const x of xTicks ?? [0, xMax / 4, xMax / 2, xMax * .75, xMax]) {
    svg.append(el('text', {x: X(x), y: H - B + 17, 'text-anchor': 'middle'}, xFormat(x)));
  }
  svg.append(el('text', {x: (L + W - R) / 2, y: H - 2, 'text-anchor': 'middle'}, xLabel));
  for (const s of series) {
    const color = COLORS[s.name] || '#83a9ef';
    const data = [...s.values].sort((a, b) => a.x - b.x);
    svg.append(el('path', {d: data.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(2)},${Y(p.y).toFixed(2)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2}));
    for (const p of data.filter((p, i) => p.error || data.length < 20 || i === data.length - 1)) {
      if (p.error) {
        svg.append(el('line', {x1: X(p.x), x2: X(p.x), y1: Y(p.y - p.error), y2: Y(p.y + p.error), stroke: color}));
        for (const sign of [-1, 1]) svg.append(el('line', {x1: X(p.x) - 3, x2: X(p.x) + 3, y1: Y(p.y + sign * p.error), y2: Y(p.y + sign * p.error), stroke: color}));
      }
      const circle = el('circle', {cx: X(p.x), cy: Y(p.y), r: 2.5, fill: color});
      circle.append(el('title', {}, `${s.name}: ${p.label || xFormat(p.x)} → ${p.y.toFixed(6)}${p.error ? ` ± ${p.error.toFixed(6)} SE` : ''}`));
      svg.append(circle);
    }
  }
  host.append(svg);
}
