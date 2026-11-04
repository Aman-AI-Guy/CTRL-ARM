// Trova voice: Grok Voice realtime (speech-to-speech) in the browser, English or Italian.
// Flow: our server mints a 5-minute token -> the browser opens wss://api.x.ai/v1/realtime with it ->
// mic audio streams as 24 kHz PCM16 -> Grok speaks back and calls our tools (find, point, check, stop).
// Safety: the live transcript is watched here; a stop word halts the arm at once, without waiting for
// the model's turn. The STOP button and the space bar always work too.
// Needs a secure page: run it on the laptop at http://localhost:8000 (phones over plain http get no mic).

const TrovaVoice = (() => {
  const RATE = 24000;
  let ws = null, ctx = null, mic = null, node = null, sink = null;
  let playTime = 0, sources = [], toolRuns = [], cfg = null, h = {};
  let lastStop = 0, userLine = '', botLine = '', buf = [], bufLen = 0;

  const WORKLET = `class Cap extends AudioWorkletProcessor {
    process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
  } registerProcessor('trova-cap', Cap);`;

  const b64 = (i16) => { const u8 = new Uint8Array(i16.buffer); let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s); };
  const unb64 = (s) => { const bin = atob(s), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Int16Array(u8.buffer); };

  function resample(f32, from) {           // only if the browser ignored our 24 kHz request
    if (from === RATE) return f32;
    const n = Math.round(f32.length * RATE / from), out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const x = i * from / RATE, j = Math.floor(x), k = x - j;
      out[i] = (f32[j] || 0) * (1 - k) + (f32[j + 1] || 0) * k; }
    return out;
  }

  function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
  const status = (s) => h.onStatus && h.onStatus(s);
  const line = (who, text, final) => h.onLine && h.onLine(who, text, final);
  const log = (who, text) => fetch('/api/voice/log', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({who, text})}).catch(() => {});

  async function api(url, body, isJson) {
    const opts = {method: 'POST'};
    if (body && isJson) { opts.headers = {'Content-Type': 'application/json'}; opts.body = JSON.stringify(body); }
    else if (body) { const f = new FormData(); for (const k in body) f.append(k, body[k]); opts.body = f; }
    const r = await fetch(url, opts); let d = {}; try { d = await r.json(); } catch (e) {}
    if (!r.ok && !d.error) d.error = d.detail || r.statusText;
    return d;
  }

  // ---- tools Grok can call: each returns a small JSON for the model to talk about
  const tools = {
    async find_object({description}) {
      const d = await api('/api/find', {text: description, move: cfg.confirm ? 'false' : 'true'});
      if (d.error) return {error: d.error};
      h.onFind && h.onFind(d);
      const r = d.result;
      return {status: r.status, label: r.label, reason: r.reason, clarifying_question: r.clarifying_question,
              candidate_ids: r.candidate_ids, arm: d.arm_blocked ? 'blocked: ' + d.arm_blocked : (d.moved ? 'pointing' : 'waiting for yes')};
    },
    async answer_clarification({answer}) {
      const d = await api('/api/clarify', {answer, move: 'false'});
      if (d.error) return {error: d.error};
      h.onFind && h.onFind(d);
      const r = d.result;
      return {status: r.status, label: r.label, reason: r.reason, clarifying_question: r.clarifying_question};
    },
    async point_at_found() { const d = await api('/api/arm/point_last'); return d.error ? {error: d.error} : {ok: true, label: d.label}; },
    async find_all({description}) {
      const d = await api('/api/find_all', {text: description, move: cfg.confirm ? 'false' : 'true'});
      if (d.error) return {error: d.error};
      h.onFindAll && h.onFindAll(d);
      return {count: d.result.count, label: d.result.label, arm: d.arm_blocked ? 'blocked: ' + d.arm_blocked : (d.moved ? 'visiting them' : 'waiting for yes')};
    },
    async describe_table() {
      const d = await api('/api/inventory'); if (d.error) return {error: d.error};
      h.onInventory && h.onInventory(d);
      return {summary: d.summary, objects: d.objects.map(o => o.name)};
    },
    async save_reference() { const d = await api('/api/reference'); h.onRefresh && h.onRefresh(); return d.error ? {error: d.error} : {saved: true, objects: d.n_objects}; },
    async check_changes() {
      const d = await api('/api/changes'); if (d.error) return {error: d.error};
      h.onCheck && h.onCheck(d);
      const r = d.result;
      return {line_clear: r.line_clear, summary: r.summary,
              changes: r.changes.map(c => ({kind: c.kind, object: c.label, risk: c.risk, why: c.explanation}))};
    },
    async read_labels() { const d = await api('/api/labels'); if (d.error) return {error: d.error};
      h.onLabels && h.onLabels(d);
      return {batch: d.lot, ok: d.ok, problems: d.problems.map(p => p.text), labels: d.items.map(i => ({object: i.box_id, lot: i.lot, expiry: i.expiry}))}; },
    async watch_line({on}) { const d = await api('/api/watch', {on: !!on, point: false}, true); h.onRefresh && h.onRefresh();
      return d.error ? {error: d.error} : {watching: d.on, status: d.status}; },
    async stop_arm() { await api('/api/arm/stop'); return {stopped: true}; },
    async arm_home() { await api('/api/arm/home'); return {parking: true}; },
    async search_other_rooms({description}) {
      const d = await api('/api/dog/search', {description}); h.onRefresh && h.onRefresh();
      return d.error ? {error: d.error} : {status: d.status, spot: d.spot, where: d.where, label: d.label, visited: d.visited};
    },
  };

  function heardStop(text) {
    const t = ' ' + text.toLowerCase().replace(/[^a-zà-ù ]/g, ' ') + ' ';
    if (cfg.stop_words.some(w => t.includes(' ' + w + ' ')) && Date.now() - lastStop > 1500) {
      lastStop = Date.now();
      fetch('/api/arm/stop', {method: 'POST'});          // no model in the loop
      status('STOP heard: arm halted');
      h.onRefresh && h.onRefresh();
    }
  }

  function stopPlayback() { sources.forEach(s => { try { s.stop(); } catch (e) {} }); sources = []; playTime = ctx ? ctx.currentTime : 0; }

  function play(i16) {
    const f = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
    const ab = ctx.createBuffer(1, f.length, RATE); ab.copyToChannel(f, 0);
    const s = ctx.createBufferSource(); s.buffer = ab; s.connect(ctx.destination);
    const at = Math.max(ctx.currentTime + 0.02, playTime); s.start(at); playTime = at + ab.duration;
    sources.push(s); s.onended = () => { sources = sources.filter(x => x !== s); };
  }

  const playbackDone = () => new Promise(res => { const tick = () => (!ctx || ctx.currentTime >= playTime - 0.05) ? res() : setTimeout(tick, 60); tick(); });

  async function onEvent(e) {
    switch (e.type) {
      case 'session.updated': status('listening'); break;
      case 'input_audio_buffer.speech_started': stopPlayback(); status('listening…'); userLine = ''; break;
      case 'conversation.item.input_audio_transcription.updated':
        userLine = e.transcript || e.text || userLine; heardStop(userLine); line('you', userLine, false); break;
      case 'conversation.item.input_audio_transcription.completed':
        userLine = e.transcript || e.text || userLine; heardStop(userLine); line('you', userLine, true); log('you', userLine); break;
      case 'response.output_audio_transcript.delta': botLine += e.delta || ''; line('trova', botLine, false); status('speaking'); break;
      case 'response.output_audio_transcript.done': line('trova', botLine, true); log('trova', botLine); botLine = ''; break;
      case 'response.output_audio.delta': if (e.delta) play(unb64(e.delta)); break;
      case 'response.function_call_arguments.done': {
        let args = {}; try { args = JSON.parse(e.arguments || '{}'); } catch (err) {}
        status('working: ' + e.name);
        const run = (tools[e.name] ? tools[e.name](args) : Promise.resolve({error: 'unknown tool ' + e.name}))
          .catch(err => ({error: String(err)}))
          .then(out => { h.onTool && h.onTool(e.name, args, out);
            send({type: 'conversation.item.create', item: {type: 'function_call_output', call_id: e.call_id, output: JSON.stringify(out)}}); });
        toolRuns.push(run); break;
      }
      case 'response.done':
        if (toolRuns.length) {               // all tool outputs first, then one response.create, after playback
          const runs = toolRuns; toolRuns = [];
          await Promise.all(runs); await playbackDone(); send({type: 'response.create'});
        } else status('listening');
        break;
      case 'error': status('error: ' + ((e.error && e.error.message) || JSON.stringify(e).slice(0, 120))); break;
    }
  }

  async function start(handlers, confirm = true) {
    h = handlers || {};
    status('connecting…');
    const c = await (await fetch('/api/voice/config?confirm=' + (confirm ? 1 : 0))).json();
    cfg = {...c, confirm};
    if (!c.available) { status('voice needs XAI_API_KEY (mock mode)'); return false; }
    const tok = await api('/api/voice/session');
    if (tok.error || !tok.token) { status('token error: ' + (tok.error || 'none')); return false; }
    try {
      ctx = new AudioContext({sampleRate: RATE});
      await ctx.resume();                    // Safari starts contexts suspended
      mic = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, channelCount: 1}});
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type: 'application/javascript'})));
      const src = ctx.createMediaStreamSource(mic);
      node = new AudioWorkletNode(ctx, 'trova-cap');
      sink = ctx.createGain(); sink.gain.value = 0;
      src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    } catch (err) { stop(); throw err; }    // no mic (denied, or plain http on a phone): release what we opened
    const sock = new WebSocket('wss://api.x.ai/v1/realtime?model=' + encodeURIComponent(c.model), ['xai-client-secret.' + tok.token]);
    ws = sock;
    sock.onopen = () => { send({type: 'session.update', session: c.session}); status('connected'); };
    sock.onmessage = (m) => { if (typeof m.data === 'string') onEvent(JSON.parse(m.data)); };
    sock.onclose = () => { if (ws === sock) { stop(); h.onClosed && h.onClosed(); } };   // dropped: free the mic too
    sock.onerror = () => status('connection error');
    node.port.onmessage = (ev) => {          // ~100 ms chunks of PCM16
      buf.push(resample(ev.data, ctx.sampleRate)); bufLen += buf[buf.length - 1].length;
      if (bufLen >= RATE / 10) {
        const f = new Float32Array(bufLen); let o = 0; buf.forEach(b => { f.set(b, o); o += b.length; }); buf = []; bufLen = 0;
        const i16 = new Int16Array(f.length);
        for (let i = 0; i < f.length; i++) i16[i] = Math.max(-1, Math.min(1, f[i])) * 0x7fff;
        send({type: 'input_audio_buffer.append', audio: b64(i16)});
      }
    };
    return true;
  }

  function stop() {
    try { ws && ws.close(); } catch (e) {}
    try { mic && mic.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { ctx && ctx.close(); } catch (e) {}
    ws = ctx = mic = node = sink = null; sources = []; toolRuns = []; status('voice off');
  }

  return {start, stop, running: () => !!ws};
})();
