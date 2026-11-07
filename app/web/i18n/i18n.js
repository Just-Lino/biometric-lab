/* Три языка интерфейса (русский, казахский, английский): клиентская часть. Правила те же, что в app/core/i18n.py.
   Страница перед загрузкой этого файла получает от сервера window.LAB_LANG ('ru' | 'kk' | 'en') и window.LAB_CAT
   (словарь «русский текст -> перевод» для своего языка; для русского пуст).
   LT('Русский текст {0}', [значение]) - перевод строки из исходного кода: точное совпадение с каталогом.
   LTX(текст) - перевод уже готового динамического русского текста (с числами внутри): точное совпадение, шаблоны {n}, предложения.
   LABI18N.translateDOM(корень) - перевод статичной разметки (текст, title, placeholder, alt, aria-label).
   Перевод на казахский выполнен автоматически: его должен проверить носитель языка. */
(function () {
  'use strict';
  var LANG = window.LAB_LANG || 'ru', CAT = window.LAB_CAT || {};
  var CYR = /[А-Яа-яЁё]/, PH = /\{(\d+)\}/g;
  var norm = function (s) { return s.replace(/\s+/g, ' ').trim(); };
  var esc = function (s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); };
  var tpls = null;

  function sub(t, args) {
    if (!args) return t;
    return t.replace(PH, function (m, i) { return Object.prototype.hasOwnProperty.call(args, i) ? args[i] : m; });
  }
  function buildTemplates() {
    tpls = [];
    for (var k in CAT) {
      if (!/\{\d+\}/.test(k)) continue;
      var lits = k.split(/\{\d+\}/), nums = (k.match(/\{\d+\}/g) || []).map(function (x) { return +x.slice(1, -1); });
      var rx = new RegExp('^' + lits.map(esc).join('(.+?)') + '$', 's');
      tpls.push({ rx: rx, nums: nums, tr: CAT[k], w: lits.join('').length });
    }
    tpls.sort(function (a, b) { return b.w - a.w; });
  }
  function lookup(core) {
    var hit = CAT[core];
    if (hit === undefined) hit = CAT[norm(core)];
    if (hit !== undefined) return hit;
    if (tpls === null) buildTemplates();
    for (var i = 0; i < tpls.length; i++) {
      var t = tpls[i], m = t.rx.exec(core);
      if (m) {
        var vals = {};
        t.nums.forEach(function (n, j) { vals[n] = LTX(m[j + 1]); });
        return t.tr.replace(PH, function (mm, n) { return vals[n] !== undefined ? vals[n] : mm; });
      }
    }
    return undefined;
  }
  function LT(s, args) {
    if (LANG === 'ru' || typeof s !== 'string') return sub(s, args);
    var r = CAT[s];
    if (r === undefined) {
      var rr = CAT[norm(s)];                                   // записи каталога хранятся без лишних пробелов: края возвращаем
      if (rr !== undefined) r = s.slice(0, s.length - s.replace(/^\s+/, '').length) + rr + s.slice(s.replace(/\s+$/, '').length);
    }
    return sub(r === undefined ? s : r, args);
  }
  function LTX(text) {
    if (LANG === 'ru' || typeof text !== 'string' || !CYR.test(text)) return text;
    var lead = text.slice(0, text.length - text.replace(/^\s+/, '').length);
    var trail = text.slice(text.replace(/\s+$/, '').length);
    var core = text.trim(), hit = lookup(core);
    if (hit !== undefined) return lead + hit + trail;
    var parts = core.split(/((?<=[.!?…])\s+|\n+)/), out = [], any = false;
    if (parts.length < 2) return text;
    for (var i = 0; i < parts.length; i += 2) {
      var h = (parts[i] && CYR.test(parts[i])) ? lookup(parts[i]) : undefined;
      if (h !== undefined) any = true;
      out.push(h !== undefined ? h : parts[i]);
      if (i + 1 < parts.length) out.push(parts[i + 1]);
    }
    return any ? lead + out.join('') + trail : text;
  }

  var ATTRS = ['title', 'placeholder', 'alt', 'aria-label'];
  function translateDOM(root) {
    if (LANG === 'ru') return;
    root = root || document.documentElement;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        var p = n.parentNode && n.parentNode.nodeName;
        return (p === 'SCRIPT' || p === 'STYLE' || p === 'TEXTAREA' || !CYR.test(n.nodeValue)) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [], n;
    while ((n = walker.nextNode())) nodes.push(n);
    nodes.forEach(function (t) { t.nodeValue = LTX(t.nodeValue); });
    var els = (root.querySelectorAll ? root.querySelectorAll('*') : []);
    for (var i = 0; i < els.length; i++) {
      for (var j = 0; j < ATTRS.length; j++) {
        var v = els[i].getAttribute(ATTRS[j]);
        if (v && CYR.test(v)) els[i].setAttribute(ATTRS[j], LTX(v));
      }
    }
  }

  var NAMES = { ru: 'RU', kk: 'ҚАЗ', en: 'EN' };
  var TITLES = { ru: 'Русский язык', kk: 'Қазақ тілі', en: 'English' };
  function setLang(code) {
    document.cookie = 'lab_lang=' + code + '; path=/; max-age=31536000; SameSite=Lax';
    try { localStorage.setItem('lab_lang', code); } catch (e) { /* без localStorage язык хранится в cookie */ }
    location.reload();
  }
  function mount(el) {
    if (typeof el === 'string') el = document.getElementById(el);
    if (!el) return;
    el.className = 'langsw';
    el.setAttribute('role', 'group');
    ['ru', 'kk', 'en'].forEach(function (c) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = NAMES[c]; b.title = TITLES[c]; b.setAttribute('aria-label', TITLES[c]);
      if (c === LANG) b.className = 'on';
      b.onclick = function () { if (c !== LANG) setLang(c); };
      el.appendChild(b);
    });
  }
  var css = document.createElement('style');
  css.textContent = '.langsw{display:inline-flex;gap:2px;margin-left:8px;vertical-align:middle}.langsw button{border:1px solid rgba(255,255,255,.35);' +
    'background:transparent;color:#fff;border-radius:6px;padding:4px 8px;font-size:12px;cursor:pointer;min-width:34px}.langsw button.on{background:#fff;color:#14213d;font-weight:700}' +
    '.langsw.light button{color:#14213d;border-color:#b8c0d4}.langsw.light button.on{background:#14213d;color:#fff}';
  (document.head || document.documentElement).appendChild(css);
  document.documentElement.lang = LANG;
  window.LT = LT;
  window.LTX = LTX;
  window.LABI18N = { lang: LANG, translateDOM: translateDOM, setLang: setLang, mount: mount };
})();
