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
  let proposal = null, confirmationHeard = false, serialTools = Promise.resolve();
  let tableProposal = null;                 // table scan: camera-above move awaiting an explicit spoken yes
  function holdTableProposal(d) {
    tableProposal = d && d.proposal_id && d.approach === 'awaiting_confirmation'
      ? {id: d.proposal_id, expires_at: d.expires_at || Date.now() / 1000 + 55} : null;
  }
  let selectionQuestion = false;
  let tourPending = false;

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
      tourPending = false;
      // After a table scan the arm camera's object map is the source: "trovami X" goes over X or points at it.
      try {
        const ts = await (await fetch('/api/table_scan')).json();
        if (ts && (ts.objects || []).length) return tools.find_on_table({description});
      } catch (e) {}
      const d = await api('/api/find', {text: description, move: 'false'});
      if (d.error) return {error: d.error};
      h.onFind && h.onFind(d);
      const r = d.result;
      return {status: r.status, label: r.label, reason: r.reason, clarifying_question: r.clarifying_question,
              candidate_ids: r.candidate_ids, arm: d.arm_blocked ? 'blocked: ' + d.arm_blocked : 'not moved'};
    },
    async answer_clarification({answer}) {
      if (selectionQuestion) return tools.select_object({description: answer});
      const d = await api('/api/clarify', {answer, move: 'false'});
      if (d.error) return {error: d.error};
      h.onFind && h.onFind(d);
      const r = d.result;
      return {status: r.status, label: r.label, reason: r.reason, clarifying_question: r.clarifying_question};
    },
    async point_at_found() {
      if (tourPending) return {status: 'refused', spoken_message: 'Select one object, then request its approach. An automatic voice tour has no validated proposal.'};
      return tools.run_vision_task({text: 'move above the selected object'});
    },
    async find_all({description}) {
      tourPending = true;
      const d = await api('/api/find_all', {text: description, move: 'false'});
      if (d.error) return {error: d.error};
      h.onFindAll && h.onFindAll(d);
      return {count: d.result.count, label: d.result.label, arm: d.arm_blocked ? 'blocked: ' + d.arm_blocked : 'not moved'};
    },
    async describe_table() {
      const d = await api('/api/inventory'); if (d.error) return {error: d.error};
      h.onInventory && h.onInventory(d);
      const rows = (d.snapshot && d.snapshot.objects) || d.objects;
      return {summary: d.summary, grok_mode: d.snapshot && d.snapshot.grok_mode,
              authoritative: !!(d.snapshot && d.snapshot.authoritative), motion_allowed: false,
              objects: rows.map(o => ({track_id: o.track_id || o.box_id, name: o.name, ambiguous: !!o.ambiguous,
                                       confidence: o.confidence}))};
    },
    async select_object({description, track_id}) {
      tourPending = false;
      const body = {};
      if (description) body.text = description;
      if (track_id != null && track_id !== '') body.track_id = String(track_id);
      const d = await api('/api/select', body);
      proposal = null;
      selectionQuestion = d.status === 'clarify';
      if (d.error) return {error: d.error};
      h.onSelect && h.onSelect(d);
      return {spoken_message: d.spoken_message || d.question || (d.object ? 'Selected: ' + d.object.name : d.reason),
              status: d.status, name: d.object && d.object.name, track_id: d.object && d.object.track_id,
              question: d.question, candidates: (d.candidates || []).map(o => o.track_id),
              authoritative: d.authoritative, motion_allowed: d.motion_allowed, reason: d.reason};
    },
    async save_reference() { const d = await api('/api/reference'); h.onRefresh && h.onRefresh(); return d.error ? {error: d.error} : {saved: true, objects: d.n_objects}; },
    async check_changes() {
      const d = await api('/api/changes'); if (d.error) return {error: d.error};
      h.onCheck && h.onCheck(d);
      const r = d.result;
      return {line_clear: r.line_clear, summary: r.summary,
              changes: r.changes.map(c => ({kind: c.kind, object: c.label, risk: c.risk, why: c.explanation}))};
    },
    async look_around({description}) {
      const s = await api('/api/room/find', {text: description}); if (s.error) return {error: s.error};
      for (let i = 0; i < 120; i++) {          // the scan takes a few seconds per photo: wait for it
        await new Promise(r => setTimeout(r, 1000));
        const st = await (await fetch('/api/state')).json(), d = st.room;
        if (d && d.status !== 'scanning') { h.onRefresh && h.onRefresh();
          return {status: d.status, label: d.label, where: d.reason, photo: d.view, facing: !!d.facing, error: d.error}; }
      }
      return {error: 'the scan took too long'};
    },
    async read_labels() { const d = await api('/api/labels'); if (d.error) return {error: d.error};
      h.onLabels && h.onLabels(d);
      return {batch: d.lot, ok: d.ok, problems: d.problems.map(p => p.text), labels: d.items.map(i => ({object: i.box_id, lot: i.lot, expiry: i.expiry}))}; },
    async watch_line({on}) { const d = await api('/api/watch', {on: !!on, point: false}, true); h.onRefresh && h.onRefresh();
      return d.error ? {error: d.error} : {watching: d.on, status: d.status}; },
    async run_vision_task({text}) {
      const d = await api('/api/agent/task', {text}, true);
      proposal = d.confirmation || null;
      selectionQuestion = d.status === 'clarify';
      confirmationHeard = false;
      h.onAgent && h.onAgent(d);
      return d;
    },
    async confirm_motion({proposal_id}) {
      if (!confirmationHeard || !proposal || proposal.id !== proposal_id || Date.now() / 1000 >= proposal.expires_at)
        return {status: 'refused', spoken_message: 'No current validated proposal and explicit confirmation. Movement not executed.'};
      confirmationHeard = false;
      proposal = null;
      const d = await api('/api/agent/confirm', {proposal_id}, true);
      proposal = d.confirmation || null;  // A new scan view still requires a new spoken yes.
      h.onAgent && h.onAgent(d);
      return d;
    },
    async scan_requests() {
      let d = {};
      try { d = await (await fetch('/api/scan/review')).json(); } catch (e) { return {error: 'scan review unavailable'}; }
      h.onRefresh && h.onRefresh();
      const pending = (d.requests || []).filter(q => q.status === 'pending');
      return {job_id: d.job_id, complete: !!d.complete, summary: d.summary || d.reason || '',
              pending: pending.map(q => ({request_id: q.id, instruction: q.instruction, reason: q.reason})),
              surface_coverage_measured: false, moves_arm: false};
    },
    async scan_request_done({request_id, status}) {
      const d = await api('/api/scan/review/request', {request_id, status}, true);
      h.onRefresh && h.onRefresh();
      return d.error ? {error: d.error} : {status: d.request.status, requires_reselection: d.requires_reselection,
        spoken_message: d.spoken_message || (status === 'done' ? 'Segnato come fatto.' : 'Segnato come non possibile.')};
    },
    async cancel_task() {
      proposal = null; confirmationHeard = false;
      const d = await api('/api/agent/cancel');
      return {...d, spoken_message: 'Task cancelled. Previous motion confirmations are invalid.'};
    },
    async task_status() {
      const state = await (await fetch('/api/state')).json();
      const d = state.agent || {spoken_message: 'No autonomous task has run.'};
      proposal = d.confirmation || null;
      return d;
    },
    async jog_relative({direction, distance_mm}) {
      const heard = typeof TrovaSpeech !== 'undefined' && TrovaSpeech.parse(userLine, cfg.stop_words || []);
      if (!heard || heard.cmd !== 'jog' || heard.direction !== direction || heard.distance_mm !== Number(distance_mm))
        return {ok: false, status: 'refused', reason: 'A jog needs an explicit direction and distance in the operator transcript.'};
      const d = await api('/api/arm/jog', {direction, distance_mm: Number(distance_mm)}, true);
      h.onRefresh && h.onRefresh();
      return {ok: !!d.ok, status: d.status, reason: d.reason || d.error, mode: d.mode,
              requested_mm: d.requested && d.requested.distance_mm,
              software_stop_is_not_safety_rated: true, collision_avoidance: false};
    },
    // ---- table scan with the V2 camera model (docs/TABLE_SCAN_V2.md)
    async scan_table({move} = {}) {
      tableProposal = null;
      const s = await api('/api/table_scan/start', {move: move !== false}, true);
      if (s.error) return {status: 'refused', spoken_message: s.error};
      h.onRefresh && h.onRefresh();
      for (let i = 0; i < 300; i++) {          // moving + Grok labelling: up to about five minutes
        await new Promise(r => setTimeout(r, 1500));
        let d = {}; try { d = await (await fetch('/api/table_scan')).json(); } catch (e) { continue; }
        if (!['planned', 'moving', 'capturing', 'labelling'].includes(d.status)) {
          h.onRefresh && h.onRefresh();
          return {status: d.status, spoken_message: d.summary || d.reason || d.status,
                  objects: (d.objects || []).map(o => ({id: o.id, name: o.name, place: o.place})),
                  physical_positions: !!d.physical_positions};
        }
      }
      return {status: 'error', spoken_message: 'The table scan took too long.'};
    },
    async find_on_table({description}) {
      tableProposal = null; confirmationHeard = false;
      const d = await api('/api/table_scan/find', {text: description}, true); h.onRefresh && h.onRefresh();
      if (d.error) return {status: 'error', spoken_message: d.error};
      holdTableProposal(d);       // the move is planned at once: the spoken "sì" must be able to confirm it
      return {status: d.status, spoken_message: d.spoken_message, question: d.question, candidates: d.candidates,
              object_id: d.object_id, approach: d.approach, proposal_id: d.proposal_id};
    },
    async choose_table_object({object_id, answer}) {
      tableProposal = null; confirmationHeard = false;
      const body = {}; if (object_id != null && object_id !== '') body.object_id = Number(object_id); if (answer) body.answer = answer;
      const d = await api('/api/table_scan/choose', body, true); h.onRefresh && h.onRefresh();
      if (d.error) return {status: 'error', spoken_message: d.error};
      holdTableProposal(d);
      return {status: d.status, object_id: d.object_id, spoken_message: d.spoken_message, approach: d.approach,
              proposal_id: d.proposal_id};
    },
    async move_camera_above({object_id} = {}) {
      confirmationHeard = false;
      const body = {}; if (object_id != null && object_id !== '') body.object_id = Number(object_id);
      const d = await api('/api/table_scan/approach', body, true); h.onRefresh && h.onRefresh();
      if (d.error) { tableProposal = null; return {status: 'refused', spoken_message: d.error}; }
      tableProposal = d.proposal_id ? {id: d.proposal_id, expires_at: d.expires_at} : null;
      return {status: d.status, proposal_id: d.proposal_id, spoken_message: d.spoken_message};
    },
    async confirm_camera_move({proposal_id}) {
      if (!confirmationHeard || !tableProposal || tableProposal.id !== proposal_id || Date.now() / 1000 >= tableProposal.expires_at)
        return {status: 'refused', spoken_message: 'Nessuna proposta valida confermata a voce: movimento non eseguito.'};
      confirmationHeard = false; tableProposal = null;
      const d = await api('/api/table_scan/confirm', {proposal_id}, true); h.onRefresh && h.onRefresh();
      return d.error ? {status: 'refused', spoken_message: d.error} : {status: d.status, spoken_message: d.spoken_message};
    },
    async stop_arm() { proposal = null; tableProposal = null; confirmationHeard = false; const d = await api('/api/arm/stop');
      return {...d, spoken_message: d.error || 'STOP requested.', software_stop_is_not_safety_rated: true}; },
    async arm_home() { const d = await api('/api/arm/home'); return {...d, spoken_message: d.error || 'Home command submitted; completion is not verified.'}; },
    async search_other_rooms({description}) {
      const d = await api('/api/dog/search', {description}); h.onRefresh && h.onRefresh();
      return d.error ? {error: d.error} : {status: d.status, spot: d.spot, where: d.where, label: d.label, visited: d.visited};
    },
  };

  function heardStop(text) {
    const t = ' ' + text.toLowerCase().replace(/[^a-zà-ù ]/g, ' ') + ' ';
    if (cfg.stop_words.some(w => t.includes(' ' + w + ' ')) && Date.now() - lastStop > 1500) {
      lastStop = Date.now();
      proposal = null; tableProposal = null; confirmationHeard = false;
      fetch('/api/arm/stop', {method: 'POST'});          // no model in the loop
      status('STOP heard: arm halted');
      h.onRefresh && h.onRefresh();
    }
  }

  // Half duplex. Browser echo cancellation does not cover speechSynthesis output, so the microphone heard Trova's
  // spoken result, Grok transcribed it as the operator and answered itself. While Trova speaks (and a short tail for
  // room echo) the microphone sends silence. Exception: while the arm moves the microphone always stays open, so a
  // spoken stop word is never muted. The on-screen STOP and the physical e-stop are unaffected.
  const ECHO_TAIL_MS = 700;
  let quietUntil = 0;
  function echoGuard() {
    const tts = typeof speechSynthesis !== 'undefined' && (speechSynthesis.speaking || speechSynthesis.pending);
    if (tts) quietUntil = Date.now() + ECHO_TAIL_MS;
    if (h && h.armMoving && h.armMoving()) return false;
    return Date.now() < quietUntil;
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
        userLine = e.transcript || e.text || userLine; heardStop(userLine); line('you', userLine, true); log('you', userLine);
        confirmationHeard = !!(proposal || tableProposal) && /^(s[iì]|yes|ok|procedi|proceed|go ahead)([,! .]+(procedi|proceed))?[.! ]*$/i.test(userLine.trim());
        break;
      // Spoken status comes from backend tool results. Discard speculative model audio/text.
      case 'response.output_audio_transcript.delta':
      case 'response.output_audio_transcript.done':
      case 'response.output_audio.delta': break;
      case 'response.function_call_arguments.done': {
        let args = {}; try { args = JSON.parse(e.arguments || '{}'); } catch (err) {}
        status('working: ' + e.name);
        const run = (['stop_arm', 'cancel_task'].includes(e.name) ? tools[e.name]() :
          serialTools.then(() => tools[e.name] ? tools[e.name](args) : {error: 'unknown tool ' + e.name}))
          .catch(err => ({error: String(err)}))
          .then(out => { h.onTool && h.onTool(e.name, args, out);
            const message = out.spoken_message || out.error || out.question || out.clarifying_question || out.summary || out.reason || out.status || 'Result received.';
            line('trova', message, true); log('trova', message);
            if (typeof TrovaSpeech !== 'undefined') TrovaSpeech.say(message, TrovaSpeech.italian(userLine) ? 'it-IT' : 'en-US');
            send({type: 'conversation.item.create', item: {type: 'function_call_output', call_id: e.call_id, output: JSON.stringify(out)}}); });
        serialTools = run;
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
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Open http://localhost:8000 on this laptop, or use HTTPS, to enable the microphone.');
    }
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
        if (echoGuard()) f.fill(0);          // Trova is talking: send silence so its own voice is not heard as "you"
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
    ws = ctx = mic = node = sink = null; sources = []; toolRuns = []; proposal = null; tableProposal = null; confirmationHeard = false; status('voice off');
  }

  return {start, stop, running: () => !!ws,
    ...(typeof module !== 'undefined' ? {tools, onEvent, configure: (config, handlers = {}) => {cfg = config; h = handlers;}} : {})};
})();

if (typeof module !== 'undefined') module.exports = TrovaVoice;
