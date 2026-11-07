/* Web Worker практикума (ES-модуль: Pyodide 314 не загружается в классическом воркере): Python в браузере (Pyodide).
   Загружает Pyodide с нашего сервера (/vendor/pyodide/, см. python -m scripts.download_pyodide), запускает пример задания
   через lab_runner.py и присылает результат странице. Время выполнения ограничивает страница: если код работает дольше 3 с,
   она вызывает worker.terminate() и создаёт воркер заново. Сообщения:
     страница -> воркер: {type:'run', id, payload}
     воркер -> страница: {type:'ready'} | {type:'failed', reason} | {type:'phase'|'started'|'out'|'result'|'envfail', id, ...} */
import { loadPyodide } from '/vendor/pyodide/pyodide.mjs';

const BASE = '/vendor/pyodide/';
let py = null;

function post(m) { self.postMessage(m); }

async function init() {
  try {
    py = await loadPyodide({ indexURL: BASE });
    await py.loadPackage(['numpy']);
    const [runner, safety] = await Promise.all([
      fetch('/codelab/lab_runner.py').then(r => { if (!r.ok) throw new Error('lab_runner.py: ' + r.status); return r.text(); }),
      fetch('/codelab/safety.py').then(r => { if (!r.ok) throw new Error('safety.py: ' + r.status); return r.text(); }),
    ]);
    py.FS.mkdirTree('/home/pyodide/lab');
    py.FS.writeFile('/home/pyodide/lab/lab_runner.py', runner);
    py.FS.writeFile('/home/pyodide/lab/safety.py', safety);
    py.runPython("import sys\nsys.path.insert(0, '/home/pyodide/lab')\nimport safety, lab_runner");
    post({ type: 'ready', python: py.runPython('import sys; sys.version.split()[0]') });
  } catch (e) {
    post({ type: 'failed', reason: String(e && e.message || e) });
  }
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type !== 'run' || !py) return;
  try {
    // подгружаем только то, что код действительно импортирует (scipy, scikit-image ...); если пакета нет на сервере - ошибка среды
    await py.loadPackagesFromImports(JSON.parse(m.payload).code);
  } catch (e) {
    post({ type: 'envfail', id: m.id, reason: String(e && e.message || e) });
    return;
  }
  post({ type: 'started', id: m.id });          // с этого момента страница отсчитывает 3 секунды
  try {
    py.globals.set('_lab_payload', m.payload);
    py.globals.set('_lab_emit', (s) => post({ type: 'out', id: m.id, text: String(s) }));
    py.globals.set('_lab_phase', (s) => post({ type: 'phase', id: m.id, name: String(s) }));
    const res = py.runPython('import lab_runner\nlab_runner.run_example(_lab_payload, _lab_emit, _lab_phase)');
    post({ type: 'result', id: m.id, result: JSON.parse(res) });
  } catch (e) {
    post({ type: 'envfail', id: m.id, reason: String(e && e.message || e) });
  }
};

init();
