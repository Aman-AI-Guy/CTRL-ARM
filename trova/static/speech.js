// Feature 12: browser speech, the fallback when Grok Voice is not available (mock mode, token error, bad Wi-Fi).
// Push to talk: the browser's own speech recognition (Chrome, Edge, Safari) turns speech into text, a small
// parser maps it to a Trova command, and the result is spoken back with speechSynthesis. English and Italian.
// A stop word stops the arm on the INTERIM transcript already, before the sentence is finished.

const TrovaSpeech = (() => {
  const RULES = [   // [command, regex on the lower-case transcript]; first match wins, find is the default.
    // Specific before general: "check the lot" is labels, "find my watch" is a find, not watch mode.
    ['labels', /\b(read (the )?labels?|check (the )?(lots?|batch|labels?)|leggi (le )?etichette|controlla (il )?lott[oi])\b/],
    ['watch_off', /\b(stop watching|smetti di sorvegliare)\b/],
    ['watch_on', /\b(watch the line|start watching|guard the line|sorveglia( la linea)?)\b/],
    ['inventory', /\b(what('s| is) on the table|describe the table|cosa c'?(è|e) sul tavolo|descrivi il tavolo)\b/],
    ['check', /^\s*(check|check now|controlla|controlla ora)\s*$|\b(check (the line|for changes)|what changed|line clearance|cosa (è|e) cambiato|is the line clear|controlla la linea|(la )?linea (è |e )?libera)\b/],
    ['reference', /\b(save (the )?reference|salva (il )?riferimento)\b/],
    ['home', /\b(go home|park( the arm)?|vai a casa|torna a casa|parcheggia)\b/],
  ];
  const ALL = /\b(all the|every|tutti i|tutte le|tutti gli)\b/;
  const ROOM = /\b(look around( the room)?( for)?|in the room|guarda (in giro|intorno)( per)?|nella stanza)\b/g;

  function parse(text, stopWords) {
    const t = ' ' + String(text || '').toLowerCase().replace(/[^a-zà-ù0-9' ]/g, ' ').replace(/\s+/g, ' ') + ' ';
    if ((stopWords || []).some(w => t.includes(' ' + w + ' ')))      // safety first: any stop word stops the arm
      return RULES[1][1].test(t) ? {cmd: 'stop', also: 'watch_off'} : {cmd: 'stop'};
    const q = t.trim().replace(/^(trova|find|where is|where's|dov'?(è|e)|cerca|show me|mostrami)\s+/, '');
    if (new RegExp(ROOM.source).test(t)) return {cmd: 'room', text: q.replace(ROOM, ' ').replace(/\s+/g, ' ').trim()};
    for (const [cmd, re] of RULES) if (re.test(t)) return {cmd};
    if (!q) return {cmd: 'none'};
    return {cmd: ALL.test(t) ? 'find_all' : 'find', text: q};
  }

  const italian = s => /\b(il|la|lo|gli|le|un|una|trova|dove|cerca|tutti|tutte|rosso|rossa|penna|tazza|forbici)\b/.test(String(s).toLowerCase());

  function say(text, lang) {
    if (typeof speechSynthesis === 'undefined' || !text) return;
    setTimeout(() => {                       // after the result is drawn: speech must never hold up the page
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text); u.lang = lang || 'en-US'; u.rate = 1.05;
      speechSynthesis.speak(u);
    }, 50);
  }

  // short sentences for the operator, in the language of the request
  function phraseFind(d, it) {
    const r = d.result;
    if (r.status === 'found') return (it ? 'Trovato: ' : 'Found: ') + r.label + (d.moved ? (it ? '. Lo indico.' : '. Pointing at it.') : '.');
    if (r.status === 'clarify') return r.clarifying_question;
    return it ? 'Non lo vedo sul tavolo.' : "I can't see it on the table.";
  }
  function phraseCheck(d, it) {
    const r = d.result, n = r.changes.length;
    if (r.line_clear) return it ? 'Linea libera.' : 'Line clear.';
    return (it ? `Linea non libera: ${n} differenze. ` : `Line not clear: ${n} change${n === 1 ? '' : 's'}. `) + (r.summary || '');
  }

  function supported() { return typeof window !== 'undefined' && !!(window.SpeechRecognition || window.webkitSpeechRecognition); }

  let rec = null;
  function listen({lang, stopWords, onInterim, onFinal, onStop, onEnd}) {
    const R = window.SpeechRecognition || window.webkitSpeechRecognition;
    rec = new R(); rec.lang = lang; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    let stopped = false;
    rec.onresult = e => {
      const res = e.results[e.results.length - 1], text = res[0].transcript;
      if (!stopped && parse(text, stopWords).cmd === 'stop') { stopped = true; onStop(text); }   // do not wait for the end
      // after a stop, the final sentence still runs if it is a stop too: "stop watching" also turns the watch off
      if (res.isFinal) { if (!stopped || parse(text, stopWords).cmd === 'stop') onFinal(text); } else onInterim(text);
    };
    rec.onerror = e => onEnd('mic: ' + e.error);
    rec.onend = () => { rec = null; onEnd(''); };
    rec.start();
  }
  function cancel() { try { rec && rec.stop(); } catch (e) {} }

  return {parse, italian, say, phraseFind, phraseCheck, supported, listen, cancel};
})();

if (typeof module !== 'undefined') module.exports = TrovaSpeech;
