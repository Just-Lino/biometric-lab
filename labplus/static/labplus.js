/* LAB+ общие функции: меню, запросы, графики, запись голоса, голосовой помощник.
   Подключается на любой странице: <script src="/labplus/static/labplus.js" data-assistant></script> */
(function () {
  'use strict';
  const LAB = window.LAB = window.LAB || {};
  const SELF = document.currentScript;

  /* ───────── НАСТРОЙКА: поправьте под свои вкладки ───────── */
  LAB.config = Object.assign({
    homeUrl: '/',
    // куда переходить по голосовой команде. Для своих вкладок (отпечатки, лицо, радужка) можно задать
    // функцию window.LABPLUS_OPEN(target, slots) на странице — тогда переход не нужен (см. README).
    routes: {
      fingerprint: '/#fingerprint', face: '/#face', iris: '/#iris',
      voice: '/labplus/voice', metrics: '/labplus/metrics', robustness: '/labplus/robustness', standards: '/labplus/standards'
    },
    // необязательно: window.LABPLUS_EXPLAIN = async (parsed, lastResult) => 'текст от вашего ИИ-наставника'
  }, LAB.config || {});

  const NAV = [['voice', 'Голос'], ['metrics', 'Метрики и ROC/DET'], ['robustness', 'Устойчивость'], ['standards', 'Стандарты']];
  LAB.nav = function (active) {
    const n = document.createElement('div');
    n.className = 'nav';
    n.innerHTML = '<b>LAB+</b>' + NAV.map(([k, t]) => `<a href="/labplus/${k}" class="${k === active ? 'on' : ''}">${t}</a>`).join('') +
      `<a href="${LAB.config.homeUrl}" style="margin-left:auto">← в лабораторию</a>`;
    document.body.prepend(n);
  };

  LAB.store = {
    get(k, d) { try { const v = localStorage.getItem('labplus.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('labplus.' + k, JSON.stringify(v)); } catch (e) { /* ничего */ } },
  };
  LAB.setLast = function (kind, payload) { try { sessionStorage.setItem('labplus.last', JSON.stringify({ kind, payload })); } catch (e) { /* ничего */ } };
  LAB.getLast = function () { try { return JSON.parse(sessionStorage.getItem('labplus.last')); } catch (e) { return null; } };

  LAB.api = async function (path, opts) {
    let r;
    try { r = await fetch(path, opts); } catch (e) { throw new Error('Нет связи с сервером. Он запущен?'); }
    let data = null;
    try { data = await r.json(); } catch (e) { /* не JSON */ }
    if (!r.ok) throw new Error((data && data.detail && (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))) || ('Ошибка ' + r.status));
    return data;
  };
  LAB.post = (p, obj) => LAB.api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });
  LAB.$ = (s, r) => (r || document).querySelector(s);
  LAB.show = function (el, text, err) { el.textContent = text; el.className = 'msg' + (err ? ' err' : ''); el.style.display = text ? 'block' : 'none'; };
  LAB.pct = (v, d) => (v * 100).toFixed(d === undefined ? 2 : d) + ' %';
  LAB.params = new URLSearchParams(location.search);

  /* ───────── цвета и графики ───────── */
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  LAB.col = () => ({ ink: css('--ink'), mute: css('--mute'), line: css('--line'), acc: css('--acc'), ok: css('--ok'), bad: css('--bad'), warn: css('--warn'), b: css('--b') });

  function setup(cv, h) {
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth || 400, H = h || cv.clientHeight || 260;
    cv.style.height = H + 'px';
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    return { c, W, H };
  }
  function ticks(a, b, n) {
    const step0 = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  /* o: {xmin,xmax,ymin,ymax,xlabel,ylabel,xticks:[[v,label]],yticks,xfmt,yfmt,
         bars:[{edges,vals,color}], series:[{x,y,color,dash,w}], vlines:[{x,color,label}], hlines:[{y,color,label}], points:[{x,y,color,label}], h} */
  LAB.plot = function (cv, o) {
    const { c, W, H } = setup(cv, o.h);
    const C = LAB.col(), m = { l: 48, r: 12, t: 12, b: 36 };
    const X = x => m.l + (x - o.xmin) / (o.xmax - o.xmin) * (W - m.l - m.r);
    const Y = y => H - m.b - (y - o.ymin) / (o.ymax - o.ymin) * (H - m.t - m.b);
    c.font = '12px system-ui'; c.fillStyle = C.mute; c.strokeStyle = C.line; c.lineWidth = 1;
    const xt = o.xticks || ticks(o.xmin, o.xmax, 6).map(v => [v, (o.xfmt || (t => +t.toFixed(2)))(v)]);
    const yt = o.yticks || ticks(o.ymin, o.ymax, 5).map(v => [v, (o.yfmt || (t => +t.toFixed(2)))(v)]);
    c.textAlign = 'center';
    xt.forEach(([v, l]) => { c.beginPath(); c.moveTo(X(v), m.t); c.lineTo(X(v), H - m.b); c.stroke(); c.fillText(l, X(v), H - m.b + 14); });
    c.textAlign = 'right';
    yt.forEach(([v, l]) => { c.beginPath(); c.moveTo(m.l, Y(v)); c.lineTo(W - m.r, Y(v)); c.stroke(); c.fillText(l, m.l - 5, Y(v) + 4); });
    c.save(); c.beginPath(); c.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); c.clip();
    (o.bars || []).forEach(b => {
      c.fillStyle = b.color; c.globalAlpha = 0.45;
      b.vals.forEach((v, i) => { const x0 = X(b.edges[i]), x1 = X(b.edges[i + 1]); c.fillRect(x0, Y(v), Math.max(1, x1 - x0 - 1), Y(0) - Y(v)); });
      c.globalAlpha = 1;
    });
    (o.series || []).forEach(s => {
      c.strokeStyle = s.color; c.lineWidth = s.w || 2; c.setLineDash(s.dash || []);
      c.beginPath(); s.x.forEach((x, i) => { i ? c.lineTo(X(x), Y(s.y[i])) : c.moveTo(X(x), Y(s.y[i])); }); c.stroke(); c.setLineDash([]);
    });
    (o.vlines || []).forEach(v => { c.strokeStyle = v.color || C.ink; c.setLineDash([5, 4]); c.lineWidth = 1.5; c.beginPath(); c.moveTo(X(v.x), m.t); c.lineTo(X(v.x), H - m.b); c.stroke(); c.setLineDash([]); });
    (o.hlines || []).forEach(v => { c.strokeStyle = v.color || C.ink; c.setLineDash([5, 4]); c.lineWidth = 1.5; c.beginPath(); c.moveTo(m.l, Y(v.y)); c.lineTo(W - m.r, Y(v.y)); c.stroke(); c.setLineDash([]); });
    (o.points || []).forEach(p => { c.fillStyle = p.color || C.ink; c.beginPath(); c.arc(X(p.x), Y(p.y), 5, 0, 7); c.fill(); });
    c.restore();
    c.fillStyle = C.mute; c.textAlign = 'center';
    if (o.xlabel) c.fillText(o.xlabel, (m.l + W - m.r) / 2, H - 4);
    if (o.ylabel) { c.save(); c.translate(12, (m.t + H - m.b) / 2); c.rotate(-Math.PI / 2); c.fillText(o.ylabel, 0, 0); c.restore(); }
    (o.vlines || []).concat(o.hlines || []).forEach(v => { if (v.label) { c.fillStyle = v.color || C.ink; c.textAlign = 'left'; c.fillText(v.label, v.x !== undefined ? X(v.x) + 4 : m.l + 4, v.x !== undefined ? m.t + 12 : Y(v.y) - 4); } });
    (o.points || []).forEach(p => { if (p.label) { c.fillStyle = p.color || C.ink; c.textAlign = 'left'; c.fillText(p.label, X(p.x) + 8, Y(p.y) - 6); } });
  };

  LAB.waveform = function (cv, env) {
    const { c, W, H } = setup(cv, 70), C = LAB.col(), mid = H / 2;
    c.strokeStyle = C.acc; c.lineWidth = Math.max(1, W / env.length - 0.5);
    env.forEach((v, i) => { const x = (i + 0.5) * W / env.length; c.beginPath(); c.moveTo(x, mid - v * mid); c.lineTo(x, mid + v * mid); c.stroke(); });
  };
  LAB.heat = function (cv, mat, h) {
    const { c, W, H } = setup(cv, h || 120);
    const rows = mat.length, cols = mat[0].length;
    let lo = Infinity, hi = -Infinity;
    mat.forEach(r => r.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    const cw = W / cols, ch = H / rows;
    for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
      const t = (mat[r][k] - lo) / (hi - lo || 1);
      c.fillStyle = `hsl(${250 - 230 * t},75%,${20 + 45 * t}%)`;
      c.fillRect(k * cw, H - (r + 1) * ch, cw + 0.6, ch + 0.6);
    }
  };

  /* ───────── вероятностная шкала (для DET) ───────── */
  LAB.probit = function (p) {
    p = Math.min(1 - 1e-12, Math.max(1e-12, p));
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    let q, r;
    if (p < 0.02425) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  };

  /* ───────── запись голоса → WAV 16 кГц ───────── */
  LAB.Recorder = function () {
    let ctx, stream, node, src, chunks = [], rate = 16000, on = false, onLevel = null;
    this.start = async function (levelCb) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('Микрофон недоступен: откройте страницу через localhost или HTTPS.');
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } }); }
      catch (e) { throw new Error('Нет доступа к микрофону. Разрешите его в браузере.'); }
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      rate = ctx.sampleRate; chunks = []; onLevel = levelCb; on = true;
      src = ctx.createMediaStreamSource(stream);
      node = ctx.createScriptProcessor(4096, 1, 1);   // устарел, но работает во всех браузерах без доп. файлов
      node.onaudioprocess = e => {
        if (!on) return;
        const d = new Float32Array(e.inputBuffer.getChannelData(0)); chunks.push(d);
        if (onLevel) { let s = 0; for (let i = 0; i < d.length; i++) s += d[i] * d[i]; onLevel(Math.sqrt(s / d.length)); }
      };
      src.connect(node); node.connect(ctx.destination);
    };
    this.stop = async function () {
      on = false;
      try { node.disconnect(); src.disconnect(); } catch (e) { /* ничего */ }
      stream.getTracks().forEach(t => t.stop());
      try { await ctx.close(); } catch (e) { /* ничего */ }
      let n = 0; chunks.forEach(c => n += c.length);
      const all = new Float32Array(n); let o = 0; chunks.forEach(c => { all.set(c, o); o += c.length; });
      return LAB.wav(resample(all, rate, 16000), 16000);
    };
  };
  function resample(x, from, to) {
    if (from === to) return x;
    const n = Math.floor(x.length * to / from), out = new Float32Array(n), r = from / to;
    for (let i = 0; i < n; i++) {
      const a = i * r, b = Math.min(x.length, (i + 1) * r), lo = Math.floor(a), hi = Math.max(lo + 1, Math.ceil(b));
      let s = 0, k = 0; for (let j = lo; j < hi && j < x.length; j++) { s += x[j]; k++; }
      out[i] = k ? s / k : 0;
    }
    return out;
  }
  LAB.wav = function (f32, rate) {
    const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + f32.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, f32.length * 2, true);
    for (let i = 0; i < f32.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f32[i])) * 32767, true);
    return new Blob([buf], { type: 'audio/wav' });
  };

  /* ───────── синтез речи ───────── */
  /* Голос лаборатории (Piper на сервере) общий с главной страницей: выбор хранится в localStorage под ключом lab_tts */
  LAB.tts = { server: null, audio: null };
  LAB.ttsPrefs = function () { try { return JSON.parse(localStorage.getItem('lab_tts') || '{}'); } catch (e) { return {}; } };
  LAB.ttsSave = function (patch) { try { localStorage.setItem('lab_tts', JSON.stringify(Object.assign({ on: true, rate: 1.05, voice: '' }, LAB.ttsPrefs(), patch))); } catch (e) { /* ничего */ } };
  LAB.ttsServer = async function () {
    if (LAB.tts.server) return LAB.tts.server;
    try { const r = await fetch('/api/v1/assistant/voices'); LAB.tts.server = r.ok ? ((await r.json()).voices || []) : []; } catch (e) { LAB.tts.server = []; }
    return LAB.tts.server;
  };
  LAB.speak = async function (text) {
    const pref = LAB.ttsPrefs();
    if (!text || pref.on === false) return;
    const vs = await LAB.ttsServer(), v = pref.voice || '';
    const id = v.startsWith('srv:') && vs.some(x => 'srv:' + x.id === v) ? v.slice(4) : (!v && vs.length ? vs[0].id : null);
    if (id) {
      try {
        const parts = (String(text).match(/[^.!?;:]+[.!?;:]?/g) || [String(text)]).map(x => x.trim()).filter(Boolean).flatMap(x => x.match(/.{1,500}/g));
        const get = t => fetch('/api/v1/assistant/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: t, voice: id }) })
          .then(r => { if (!r.ok) throw new Error('tts'); return r.blob(); });
        let nxt = get(parts[0]); nxt.catch(() => { });
        for (let i = 0; i < parts.length; i++) {
          const blob = await nxt;
          if (i + 1 < parts.length) { nxt = get(parts[i + 1]); nxt.catch(() => { }); }
          await new Promise(res => { const a = new Audio(URL.createObjectURL(blob)); LAB.tts.audio = a; a.playbackRate = pref.rate || 1.05; a.onended = a.onerror = () => { URL.revokeObjectURL(a.src); res(); }; a.play().catch(() => res()); });
        }
        return;
      } catch (e) { LAB.tts.server = []; /* сервер не ответил: дальше голос браузера */ }
    }
    return LAB.speakBrowser(text);
  };
  LAB.speakBrowser = function (text) {
    return new Promise(res => {
      if (!text || !('speechSynthesis' in window)) return res();
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text); u.lang = 'ru-RU'; u.rate = 1.0;
        const v = speechSynthesis.getVoices().find(x => /^ru/i.test(x.lang)); if (v) u.voice = v;
        let done = false; const fin = () => { if (!done) { done = true; res(); } };
        u.onend = fin; u.onerror = fin; setTimeout(fin, Math.max(3000, text.length * 90));
        speechSynthesis.speak(u);
      } catch (e) { res(); }
    });
  };

  /* ───────── голосовой помощник (плавающая кнопка) ───────── */
  LAB.assistant = function () {
    if (document.getElementById('lp-fab')) return;
    const fab = document.createElement('button'); fab.id = 'lp-fab'; fab.title = 'Голосовая команда'; fab.textContent = '🎤';
    const panel = document.createElement('div'); panel.id = 'lp-panel';
    const name = LAB.store.get('voiceName', '');
    panel.innerHTML = `<div class="row"><b>Голосовой помощник</b><button id="lp-x" style="margin-left:auto">×</button></div>
      <label>Режим</label><select id="lp-mode"><option value="auto">Автозапуск: сам запускает</option><option value="hint">Подсказка: объясняет, я нажимаю</option><option value="confirm">Спрашивать подтверждение</option></select>
      <label><input type="checkbox" id="lp-spk"> Проверять мой голос (профиль: <span id="lp-name">${name || 'нет'}</span>)</label>
      <label>Голос озвучки</label><select id="lp-voice"></select>
      <div class="row" style="margin-top:8px"><button class="pri" id="lp-go">🎤 Сказать команду</button></div>
      <div class="row" style="margin-top:8px"><input type="text" id="lp-txt" placeholder="или напишите команду" style="flex:1"><button id="lp-send">▶</button></div>
      <div id="lp-log" style="margin-top:8px"></div>
      <div class="note">Примеры: «покажи распознавание отпечатков пальцев по датасету FVC2004», «покажи кривые ROC», «проверь устойчивость к шуму», «объясни результат», «помощь».</div>`;
    document.body.append(fab, panel);
    const $ = s => panel.querySelector(s), log = $('#lp-log');
    const fillVoices = async () => {  // список голосов: «Голос лаборатории» (сервер) и голоса браузера; выбор общий с главной страницей
      const sel = $('#lp-voice'), vs = await LAB.ttsServer();
      const br = ('speechSynthesis' in window ? speechSynthesis.getVoices() : []).filter(x => /^ru/i.test(x.lang));
      sel.innerHTML = '<option value="">Автоматически' + (vs.length ? ' (голос лаборатории: ' + vs[0].title + ')' : '') + '</option>' +
        vs.map(x => `<option value="srv:${x.id}">Голос лаборатории: ${x.title}</option>`).join('') +
        br.map(x => `<option value="${x.name}">Браузер: ${x.name}</option>`).join('') +
        (vs.length ? '' : '<option value="" disabled>Голос лаборатории не установлен (см. README)</option>');
      const cur = LAB.ttsPrefs().voice || ''; sel.value = [...sel.options].some(o => o.value === cur) ? cur : '';
      sel.onchange = e => { LAB.ttsSave({ voice: e.target.value }); };
    };
    fillVoices();
    if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = fillVoices;  // голоса браузера подгружаются позже
    $('#lp-mode').value = LAB.store.get('mode', 'auto'); $('#lp-spk').checked = !!name && LAB.store.get('requireSpeaker', false); $('#lp-spk').disabled = !name;
    $('#lp-mode').onchange = e => LAB.store.set('mode', e.target.value);
    $('#lp-spk').onchange = e => LAB.store.set('requireSpeaker', e.target.checked);
    const say = (cls, t) => { const d = document.createElement('div'); d.className = 'line ' + cls; d.textContent = t; log.prepend(d); };
    fab.onclick = () => panel.classList.toggle('open'); $('#lp-x').onclick = () => panel.classList.remove('open');
    let pending = null, busy = false;

    async function handle(text, wavBlob) {
      say('you', '🗣 ' + text);
      const fd = new FormData();
      fd.append('text', text); fd.append('mode', $('#lp-mode').value);
      const nm = LAB.store.get('voiceName', ''), req = $('#lp-spk').checked && !!nm;
      fd.append('name', nm || ''); fd.append('require_speaker', req ? 'true' : 'false'); fd.append('threshold', LAB.store.get('voiceThreshold', 0.5));
      if (wavBlob && req) fd.append('file', wavBlob, 'cmd.wav');
      const r = await LAB.api('/api/labplus/voice/command', { method: 'POST', body: fd });
      const p = r.parsed;
      if (r.speaker && r.speaker.score !== undefined) say('', `Голос: сходство ${r.speaker.score}, ${r.speaker.decision === 'MATCH' ? 'узнан' : 'не узнан'}`);
      if (!r.allowed) { say('', '🔒 ' + r.reply); await LAB.speak(r.reply); return; }
      if (pending && (p.intent === 'yes' || p.intent === 'no')) {
        const act = pending; pending = null;
        if (p.intent === 'yes') { say('', '✅ Выполняю'); await LAB.speak('Хорошо, запускаю.'); return LAB.runAction(act); }
        say('', 'Отменено'); return LAB.speak('Хорошо, не буду.');
      }
      if (p.intent === 'explain') {
        const t = p.target ? r.reply : await LAB.explainLast(p); say('', '🤖 ' + t); return LAB.speak(t);
      }
      say('', '🤖 ' + r.reply);
      if (p.intent === 'run') {
        const mode = $('#lp-mode').value;
        if (mode === 'confirm') { pending = p; return LAB.speak(r.reply); }
        await LAB.speak(r.reply);
        if (mode === 'auto') return LAB.runAction(p);
        return;
      }
      return LAB.speak(r.reply);
    }
    async function guarded(fn) { if (busy) return; busy = true; try { await fn(); } catch (e) { say('', '⚠️ ' + e.message); } finally { busy = false; fab.classList.remove('live'); } }
    $('#lp-send').onclick = () => guarded(async () => { const t = $('#lp-txt').value.trim(); if (t) { $('#lp-txt').value = ''; await handle(t, null); } });
    $('#lp-txt').onkeydown = e => { if (e.key === 'Enter') $('#lp-send').click(); };
    $('#lp-go').onclick = () => guarded(async () => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) throw new Error('Распознавание речи этим браузером не поддерживается. Откройте Chrome или введите команду текстом.');
      const rec = new LAB.Recorder(); await rec.start(); fab.classList.add('live'); panel.classList.add('open'); say('', '🎤 Слушаю…');
      const text = await new Promise((res) => {
        const sr = new SR(); sr.lang = 'ru-RU'; sr.interimResults = false; sr.maxAlternatives = 1; let t = '';
        sr.onresult = e => { t = e.results[0][0].transcript; }; sr.onerror = () => res(''); sr.onend = () => res(t);
        setTimeout(() => { try { sr.stop(); } catch (e) { /* ничего */ } }, 9000); sr.start();
      });
      const blob = await rec.stop();
      if (!text) { say('', 'Не расслышала.'); return LAB.speak('Не расслышала. Повторите, пожалуйста.'); }
      await handle(text, blob);
    });
    LAB._assistantSay = say;
  };

  LAB.explainLast = async function (parsed) {
    const last = LAB.getLast();
    if (typeof window.LABPLUS_EXPLAIN === 'function') { try { return await window.LABPLUS_EXPLAIN(parsed, last); } catch (e) { /* падаем на встроенного наставника */ } }
    if (!last) return 'Пока нет результата для объяснения. Запустите эксперимент и повторите.';
    return (await LAB.post('/api/labplus/explain', { kind: last.kind, payload: last.payload, level: 2 })).text;
  };

  LAB.runAction = function (p) {
    const q = new URLSearchParams({ autorun: '1' });
    Object.entries(p.slots || {}).forEach(([k, v]) => q.set(k, v));
    if (typeof window.LABPLUS_OPEN === 'function' && window.LABPLUS_OPEN(p.target, p.slots || {}) !== false) return;
    const base = LAB.config.routes[p.target];
    if (!base) { if (LAB._assistantSay) LAB._assistantSay('', 'Для этого раздела не настроен переход (см. LAB.config.routes).'); return; }
    const [path, hash] = base.split('#');
    location.href = path + '?' + q.toString() + (hash ? '#' + hash : '');
  };

  document.addEventListener('DOMContentLoaded', () => {
    if ((SELF && SELF.hasAttribute('data-assistant')) || document.body.dataset.assistant !== undefined) LAB.assistant();
  });
})();
