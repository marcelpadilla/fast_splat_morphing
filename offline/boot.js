// boot.js: lets demo/app.js run straight off disk, with no web server.
//
// A file:// page may not fetch() local assets, and may not start a Worker from
// a file:// script, so a plain double-click on the online index.html comes up
// empty. This shim removes both limits:
//
//   * fetch()      — the manifest and each .splat are inlined as scripts
//                    (offline/manifest.js, offline/data/*.splat.js), loaded on
//                    demand so the page opens at once and only the two objects
//                    actually picked get read. The base64 is decoded in a
//                    worker, never on the page's main thread, so choosing an
//                    object cannot freeze the view.
//   * Worker       — app.js starts "fsm-worker.js" by name; that is replaced by
//                    a blob whose source already has the paper's four files
//                    (cloud.js, pairings.js, trajectories.js, renderer.js)
//                    spliced in, produced by offline/build_offline.py.
//
// Nothing here touches the method. The online demo keeps loading its assets
// over HTTP; this file no-ops unless offline/manifest.js was loaded (which only
// OPEN_ME.html does).
'use strict';
(function () {
  const manifest = window.FSM_MANIFEST;
  if (!manifest) return;                      // online page: leave everything alone

  const B64 = window.__FSM_B64 || (window.__FSM_B64 = {});
  const src = {};                             // 'assets/x.splat' -> data script path
  for (const o of manifest.objects) src['assets/' + o.file] = '../offline/data/' + o.file + '.js';

  const loading = {};
  function loadScript(path) {
    return loading[path] || (loading[path] = new Promise((ok, no) => {
      const s = document.createElement('script');
      s.src = path;
      s.onload = () => ok();
      s.onerror = () => no(new Error('cannot read ' + path));
      document.head.appendChild(s);
    }));
  }

  // -------------------------------------------------------------------------
  // Base64 -> ArrayBuffer, in a worker so a multi-megabyte payload never runs
  // its decode loop on the page's thread.
  // -------------------------------------------------------------------------
  const RealWorker = window.Worker;
  const DECODER = 'self.onmessage=function(e){var d=e.data;try{' +
    'var u=(self.Uint8Array.fromBase64?Uint8Array.fromBase64(d.b64):null);' +
    'if(!u){var b=atob(d.b64),n=b.length;u=new Uint8Array(n);for(var i=0;i<n;i++)u[i]=b.charCodeAt(i);}' +
    'self.postMessage({id:d.id,buf:u.buffer},[u.buffer]);' +
    '}catch(x){self.postMessage({id:d.id,error:String(x&&x.message||x)});}};';
  let decoder = null, seq = 0;
  const pending = new Map();
  function decode(b64) {
    if (!decoder) {
      decoder = new RealWorker(URL.createObjectURL(new Blob([DECODER], { type: 'application/javascript' })));
      decoder.onmessage = (e) => {
        const p = pending.get(e.data.id);
        if (!p) return;
        pending.delete(e.data.id);
        if (e.data.error) p.no(new Error(e.data.error)); else p.ok(e.data.buf);
      };
    }
    return new Promise((ok, no) => {
      const id = ++seq;
      pending.set(id, { ok, no });
      decoder.postMessage({ id, b64 });
    });
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url);
    if (url === 'assets/objects.json')
      return Promise.resolve(new Response(JSON.stringify(manifest),
        { headers: { 'content-type': 'application/json' } }));
    if (url && url in src)
      return loadScript(src[url]).then(() => decode(B64[url])).then((buf) => {
        try {
          return new Response(buf, { headers: {
            'content-type': 'application/octet-stream', 'content-length': String(buf.byteLength) } });
        } catch (e) {
          return new Response(buf, { headers: { 'content-type': 'application/octet-stream' } });
        }
      });
    return realFetch(input, init);
  };

  const workerURL = URL.createObjectURL(new Blob([window.__FSM_WORKER_SRC],
    { type: 'application/javascript' }));
  function OfflineWorker(url, opts) {
    return new RealWorker(url === 'fsm-worker.js' ? workerURL : url, opts);
  }
  OfflineWorker.prototype = RealWorker.prototype;
  window.Worker = OfflineWorker;
})();
