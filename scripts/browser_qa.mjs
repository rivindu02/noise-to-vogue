// Browser QA using Chrome DevTools Protocol and Node's built-in WebSocket.
// Start a separate browser with --remote-debugging-port=19222 first.
import fs from 'node:fs/promises';
const base = process.env.NTV_URL || 'http://127.0.0.1:17860';
const out = process.env.NTV_QA_OUTPUT || '/tmp/noise-to-vogue-qa';
await fs.mkdir(out, {recursive: true});
const tabs = await (await fetch('http://127.0.0.1:19222/json')).json();
const ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, {once: true}));
let serial = 0;
const pending = new Map(), errors = [], checks = [];
ws.addEventListener('message', event => {
  const data = JSON.parse(event.data);
  if (data.id) {
    const pair = pending.get(data.id);
    pending.delete(data.id);
    if (data.error) pair?.reject(new Error(JSON.stringify(data.error)));
    else pair?.resolve(data.result);
  }
  if (data.method === 'Runtime.exceptionThrown') errors.push(data.params.exceptionDetails);
  if (data.method === 'Log.entryAdded' && data.params.entry.level === 'error') errors.push(data.params.entry);
});
function cdp(method, params = {}) {
  const id = ++serial;
  return new Promise((resolve, reject) => { pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params})); });
}
async function js(expression) {
  const result = await cdp('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function wait(expression, timeout = 120000) {
  const start = Date.now();
  while (!await js(expression)) {
    if (Date.now() - start > timeout) throw new Error(`Timed out: ${expression}`);
    await new Promise(resolve => setTimeout(resolve, 150));
  }
}
async function check(name, expression) {
  const value = await js(expression);
  if (!value) throw new Error(`Failed: ${name}`);
  checks.push({name, passed: true});
  console.log(`PASS ${name}`);
}
async function screenshot(name) {
  const shot = await cdp('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await fs.writeFile(`${out}/${name}.png`, Buffer.from(shot.data, 'base64'));
}
async function generate(schedule, seed) {
  await js(`document.querySelector('#schedule').value=${JSON.stringify(schedule)}; document.querySelector('#seed').value=${seed}; document.querySelector('#seed').dispatchEvent(new Event('input')); document.querySelector('#generate').click()`);
  await wait(`!document.querySelector('#generate').disabled && !document.querySelector('#generation-image').hidden && !document.querySelector('#notice').textContent`);
  await js(`document.querySelector('#timeline').value=document.querySelector('#timeline').max; document.querySelector('#timeline').dispatchEvent(new Event('input'))`);
  return js(`document.querySelector('#generation-image').src`);
}
try {
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Log.enable');
  await cdp('Log.clear'); errors.length=0;
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await cdp('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  await cdp('Page.navigate', {url: base});
  await wait(`document.querySelector('#model-status')?.textContent !== 'Checking model…'`);
  await screenshot('initial-view');
  // Results can be created while this process waits; reload to fetch them.
  for (let attempt=0; attempt<120; attempt++) {
    if (await js(`fetch('/api/health').then(r=>r.json()).then(h=>h.results)`)) break;
    await new Promise(resolve=>setTimeout(resolve,10000));
  }
  await cdp('Page.reload');
  await wait(`document.querySelector('#controls-status')?.textContent === 'PAIRED CONTROLS VERIFIED'`);
  await check('trained models and measured results connected', `document.querySelector('#model-status').textContent.includes('Trained model')`);
  await check('only active tab is in initial tab order', `[...document.querySelectorAll('[role=tab]')].filter(t=>t.tabIndex===0).length===1`);
  const finals = {};
  for (const schedule of ['cosine','linear']) {
    for (const seed of [7,42,2026]) finals[`${schedule}-${seed}`] = await generate(schedule,seed);
  }
  await check('reduced motion suppresses autoplay', `document.querySelector('#play').getAttribute('aria-label').startsWith('Play')`);
  if (finals['cosine-42'] !== await generate('cosine',42)) throw new Error('Preset seed not reproducible');
  if (finals['cosine-42'] === finals['cosine-7'] || finals['cosine-42'] === finals['linear-42']) throw new Error('Seed or model selection did not change image');
  checks.push({name:'both trained schedules, all presets and fixed-seed reproducibility',passed:true});
  const custom = await generate('cosine',123);
  if (custom !== await generate('cosine',123)) throw new Error('Live seed not reproducible');
  checks.push({name:'custom live seed reproducibility',passed:true});
  await check('custom generation uses live model', `document.querySelector('#generation-source').textContent.startsWith('LIVE')`);
  await js(`document.querySelector('.forward-details').open=true`);
  await check('forward images, depth and independence caveat', `document.querySelector('#forward-original').naturalWidth>0 && document.querySelector('#forward-current').naturalWidth>0 && document.querySelector('#forward-caption').textContent.includes('k = 0') && document.querySelector('.forward-details').textContent.includes('not a reconstruction pair')`);
  await js(`document.querySelector('#restart').click(); document.querySelector('#speed').value='45'; document.querySelector('#speed').dispatchEvent(new Event('change')); document.querySelector('#play').click()`);
  await wait(`Number(document.querySelector('#timeline').value)>1`);
  await js(`document.querySelector('#play').click()`);
  await check('restart, speed, play and pause', `document.querySelector('#play').getAttribute('aria-label').startsWith('Play')`);
  await js(`document.querySelector('#timeline').value='20'; document.querySelector('#timeline').dispatchEvent(new Event('input'))`);
  await check('timeline scrub', `document.querySelector('#timeline').value==='20'`);
  await js(`document.querySelector('#timeline').value=Number(document.querySelector('#timeline').max)-1;document.querySelector('#timeline').dispatchEvent(new Event('input'));document.querySelector('#play').click()`);
  await wait(`document.querySelector('#play').getAttribute('aria-label').startsWith('Play')`);
  await check('playback stops at final clean frame', `document.querySelector('#depth-readout').textContent==='k = 0 / 200'`);
  await screenshot('desktop-generation');
  await js(`document.querySelector('#tab-generation').focus(); document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))`);
  await check('keyboard tab selection and focus', `document.activeElement.id==='tab-microscope' && !document.querySelector('#microscope-panel').hidden && document.querySelector('#generation-panel').hidden`);
  for (const [schedule,index,seed,depth] of [['cosine',0,42,100],['linear',17,123,50],['cosine',9999,7,200],['linear',1,2026,1]]) {
    await js(`document.querySelector('#inspect-schedule').value='${schedule}'; document.querySelector('#inspect-index').value=${index}; document.querySelector('#inspect-seed').value=${seed}; document.querySelector('#inspect-depth').value=${depth}; document.querySelector('#inspect-depth').dispatchEvent(new Event('input')); document.querySelector('#inspect').click()`);
    await wait(`!document.querySelector('#inspect').disabled`);
    await check(`microscope ${schedule} index ${index} depth ${depth}`, `!document.querySelector('#notice').textContent && [...document.querySelectorAll('.micro-image img')].every(i=>!i.hidden&&i.naturalWidth>0) && Number.isFinite(Number(document.querySelector('#noise-mse').textContent)) && document.querySelector('#inspection-label').textContent.includes('depth ${depth}') && document.querySelector('#signal-coefficient').textContent!=='√ᾱₖ'`);
    await check(`raw and clipped MSE and equation coefficients at depth ${depth}`, `Number(document.querySelector('#raw-clean-mse').textContent.split(': ')[1])>=Number(document.querySelector('#clean-mse').textContent) && Math.abs(Number(document.querySelector('#signal-coefficient').textContent)**2+Number(document.querySelector('#noise-coefficient').textContent)**2-1)<0.0002`);
  }
  await screenshot('desktop-microscope');
  for (const invalid of [-1,10000]) {
    await js(`document.querySelector('#inspect-index').value=${invalid};document.querySelector('#inspect').click()`);
    await check(`invalid microscope index ${invalid} explained`, `document.querySelector('#notice').textContent.includes('whole number between 0 and 9999')`);
  }
  await js(`document.querySelector('#inspect-index').value=1`);
  await js(`document.querySelector('#tab-experiment').click()`);
  await check('measured charts and samples loaded', `[...document.querySelectorAll('.chart')].every(c=>c.querySelector('svg')) && [...document.querySelectorAll('.sample-comparison img')].every(i=>i.naturalWidth>0) && !document.querySelector('#metrics-download').hidden`);
  await check('measurement download works', `(async()=>{const r=await fetch(document.querySelector('#metrics-download').href);return r.ok&&(await r.json()).protocol.test_samples===1024})()`);
  await screenshot('desktop-experiment');
  for (const width of [390,320]) {
    await cdp('Emulation.setDeviceMetricsOverride', {width,height:844,deviceScaleFactor:1,mobile:true});
    await js(`document.querySelector('#tab-generation').click()`);
    await check(`seed and schedule labels visible at ${width}px`, `['seed','schedule'].every(id=>getComputedStyle(document.querySelector('label[for='+id+']')).display!=='none')`);
    for (const tab of ['generation','microscope','experiment']) {
      await js(`document.querySelector('#tab-${tab}').click()`);
      await check(`no page overflow at ${width}px in ${tab}`, `document.documentElement.scrollWidth<=innerWidth`);
      if (width===390) await screenshot(`mobile-${tab}`);
    }
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await cdp('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion',value:'no-preference'}]});
  await js(`document.querySelector('#tab-generation').click()`);
  await generate('cosine',42);
  // generate() scrubs and pauses; trigger a new run to test automatic playback.
  await js(`document.querySelector('#generate').click()`);
  await wait(`!document.querySelector('#generate').disabled`);
  await check('normal motion automatically plays', `document.querySelector('#play').getAttribute('aria-label').startsWith('Pause')`);
  await js(`document.querySelector('#play').click()`);
  for (const seed of [-1,4294967296]) {
    await js(`document.querySelector('#seed').value=${seed};document.querySelector('#generate').click()`);
    await check(`invalid generation seed ${seed} explained`, `document.querySelector('#notice').textContent.includes('whole number between 0 and 4294967295')`);
  }
  await js(`document.querySelector('#seed').value=42`);
  await cdp('Fetch.enable', {patterns:[{urlPattern:'*/api/generate*',requestStage:'Request'}]});
  const intercept = event => {
    const d=JSON.parse(event.data);
    if(d.method==='Fetch.requestPaused') cdp('Fetch.fulfillRequest',{requestId:d.params.requestId,responseCode:503,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({detail:'No trained checkpoint available.'})).toString('base64')});
  };
  ws.addEventListener('message',intercept);
  await js(`document.querySelector('#generate').click()`);
  await wait(`!document.querySelector('#generate').disabled`);
  await check('API failure is explained', `document.querySelector('#notice').textContent==='No trained checkpoint available.'`);
  await cdp('Fetch.disable'); ws.removeEventListener('message',intercept);
  const unexpectedErrors = errors.filter(e=>!(e.source==='network' && e.text.includes('503') && /\/api\/(results|generate)/.test(e.url || '')));
  if(unexpectedErrors.length) throw new Error(`Browser errors: ${JSON.stringify(unexpectedErrors)}`);
  checks.push({name:'no browser console errors or uncaught exceptions',passed:true});
  const version=await cdp('Browser.getVersion');
  await fs.writeFile(`${out}/browser-qa.json`, JSON.stringify({status:'passed',created_utc:new Date().toISOString(),browser:version,url:base,checks,expectedNetworkErrors:errors.filter(e=>!unexpectedErrors.includes(e)),screenshots:['desktop-generation','desktop-microscope','desktop-experiment','mobile-generation','mobile-microscope','mobile-experiment'],limitations:'Missing-checkpoint error presentation uses an intercepted 503; real missing-model behavior is covered by pytest. No claim of assistive technology testing.'},null,2));
  console.log(`${checks.length} browser checks passed`);
} catch(error) {
  await fs.writeFile(`${out}/browser-qa-failure.json`,JSON.stringify({message:error.message,checks,errors},null,2));
  await screenshot('failure').catch(()=>{});
  throw error;
} finally {ws.close();}
