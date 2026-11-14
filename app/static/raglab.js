/* RAGLab UI behaviour, powered by anime.js v4 (global `anime`). Motion is kept short and functional. */
(function () {
  "use strict";
  if (!window.anime) { console.warn("[raglab] anime.js missing; UI stays static."); return; }
  var A = anime, animate = A.animate, stagger = A.stagger;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function fresh(el) { if (el.dataset.rlDone) return false; el.dataset.rlDone = "1"; return true; }
  function freshList(sel, root) { return $$(sel, root).filter(fresh); }

  /* ---------------- header ---------------- */
  function animateHero(root) {
    freshList(".rl-title", root).forEach(function (t) {
      var text = t.textContent.trim();
      t.setAttribute("aria-label", text);
      t.innerHTML = text.split("").map(function (c) { return '<span class="ch" aria-hidden="true">' + c + "</span>"; }).join("");
      if (reduce) return;
      animate($$(".ch", t), { translateY: [18, 0], opacity: [0, 1], delay: stagger(45), duration: 650, ease: "outCubic" });
    });
    freshList(".rl-eyebrow, .rl-sub", root).forEach(function (el, i) {
      if (!reduce) animate(el, { opacity: [0, 1], translateY: [8, 0], delay: 150 + i * 150, duration: 500, ease: "outCubic" });
    });
    var stats = freshList(".rl-stat", root);
    if (stats.length && !reduce) {
      animate(stats, { opacity: [0, 1], translateY: [10, 0], delay: stagger(80, { start: 300 }), duration: 500, ease: "outCubic" });
    }
    counters(root, 450);
  }

  function counters(root, baseDelay) {
    freshList("[data-count]", root).forEach(function (el) {
      var target = parseFloat(el.getAttribute("data-count"));
      if (isNaN(target)) return;
      var dec = parseInt(el.getAttribute("data-dec") || "0", 10);
      var unit = el.getAttribute("data-unit") || "";
      var fmt = function (v) { return (dec ? v.toFixed(dec) : Math.round(v).toLocaleString()) + unit; };
      if (reduce) { el.textContent = fmt(target); return; }
      var o = { v: 0 };
      animate(o, { v: target, duration: 1100, delay: baseDelay || 0, ease: "outCubic", onUpdate: function () { el.textContent = fmt(o.v); } });
    });
  }

  /* ---------------- reusable entrances ---------------- */
  function bars(root, baseDelay) {
    var list = freshList(".rl-bar i[data-w]", root);
    if (!list.length) return;
    if (reduce) { list.forEach(function (b) { b.style.width = b.getAttribute("data-w") + "%"; }); return; }
    animate(list, {
      width: function (el) { return ["0%", el.getAttribute("data-w") + "%"]; },
      duration: 800, delay: stagger(30, { start: baseDelay || 100 }), ease: "outCubic"
    });
  }

  function reveal(sel, root, opts) {
    var list = freshList(sel, root);
    if (!list.length) return list;
    if (reduce) { list.forEach(function (e) { e.style.opacity = 1; }); return list; }
    animate(list, {
      opacity: [0, 1], translateY: [(opts && opts.y) || 12, 0],
      delay: stagger((opts && opts.step) || 50), duration: 450, ease: "outCubic"
    });
    return list;
  }

  /* ---------------- playground: pipeline strip ---------------- */
  var RETR = {
    colbert: ["Dense", "ColBERT"], distilbert: ["Dense", "DistilBERT"],
    colbert_bm25: ["BM25 + RRF", "ColBERT"], distilbert_bm25: ["BM25 + RRF", "DistilBERT"]
  };
  function ddv(id) { var i = $("#" + id + " input"); return i ? i.value : ""; }
  function setNode(name, text) {
    var n = $('.rl-node[data-node="' + name + '"] em'); if (n && n.textContent !== text) n.textContent = text;
  }
  function syncPipe() {
    var c = ddv("rl-dd-chunker"), e = ddv("rl-dd-embedder"), r = ddv("rl-dd-retrieval"), p = ddv("rl-dd-style");
    if (c) setNode("chunker", c);
    if (e) setNode("embedder", e);
    if (r && RETR[r]) { setNode("retrieval", RETR[r][0]); setNode("rescore", RETR[r][1]); }
    if (p) setNode("generator", "Ollama · " + p);
  }

  /* while a query runs, stages light up in sequence; when results land they settle to "done" */
  var pulse = null, pulseTimer = 0;
  function startPulse() {
    stopPulse(true);
    var nodes = $$(".rl-node");
    nodes.forEach(function (n) { n.classList.remove("done", "on"); });
    if (reduce || !nodes.length) return;
    var i = 0;
    function tick() {
      nodes.forEach(function (n, k) { n.classList.toggle("on", k === i % nodes.length); });
      animate(nodes[i % nodes.length], { translateY: [0, -3, 0], duration: 360, ease: "outQuad" });
      i++;
    }
    tick();
    pulse = setInterval(tick, 420);
    pulseTimer = setTimeout(function () { stopPulse(false); }, 180000);
  }
  function stopPulse(silent) {
    clearTimeout(pulseTimer);
    if (pulse) { clearInterval(pulse); pulse = null; }
    var nodes = $$(".rl-node");
    nodes.forEach(function (n) { n.classList.remove("on"); });
    if (silent) return;
    nodes.forEach(function (n) { n.classList.add("done"); });
    if (!reduce) animate(nodes, { translateY: [0, -3, 0], delay: stagger(50), duration: 320, ease: "outQuad" });
  }

  /* ---------------- sortable tables ---------------- */
  function wireTable(tbl) {
    if (!fresh(tbl)) return;
    $$("th", tbl).forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var asc = !th.classList.contains("sorted-asc");
        $$("th", tbl).forEach(function (h) { h.classList.remove("sorted-asc", "sorted-desc"); });
        th.classList.add(asc ? "sorted-asc" : "sorted-desc");
        var body = $("tbody", tbl), rows = $$("tr", body);
        rows.sort(function (a, b) {
          var x = a.children[idx].getAttribute("data-v"), y = b.children[idx].getAttribute("data-v");
          var nx = parseFloat(x), ny = parseFloat(y), r;
          r = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : String(x).localeCompare(String(y));
          return asc ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
        if (!reduce) animate(rows, { opacity: [0.3, 1], delay: stagger(12), duration: 300, ease: "outCubic" });
      });
    });
  }

  /* ---------------- chart lightbox ---------------- */
  function wireGallery(root) {
    $$(".rl-fig", root).forEach(function (fig) {
      if (fig.dataset.rlWired) return;
      fig.dataset.rlWired = "1";
      fig.addEventListener("click", function () {
        var box = $("#rl-lightbox");
        if (!box) {
          box = document.createElement("div"); box.id = "rl-lightbox"; box.innerHTML = "<img alt=''>";
          box.addEventListener("click", function () {
            if (reduce) { box.style.display = "none"; return; }
            animate(box, { opacity: 0, duration: 180, ease: "outQuad", onComplete: function () { box.style.display = "none"; } });
          });
          document.body.appendChild(box);
        }
        $("img", box).src = $("img", fig).src;
        box.style.display = "grid";
        if (!reduce) animate(box, { opacity: [0, 1], duration: 200, ease: "outQuad" });
      });
    });
  }

  /* ---------------- answer reveal ---------------- */
  var lastAnswer = "";
  function watchAnswer() {
    var a = $("#rl-answer");
    if (!a) return;
    var t = a.textContent.trim();
    if (t === lastAnswer) return;
    var first = lastAnswer === "";
    lastAnswer = t;
    if (first || reduce) return;
    animate(a, { opacity: [0.2, 1], translateY: [6, 0], duration: 500, ease: "outCubic" });
  }

  /* ---------------- main scan ---------------- */
  function scan(root) {
    root = root || document;
    animateHero(root);
    reveal(".rl-node", root, { y: 8, step: 60 });
    var hits = reveal(".rl-hit", root, { step: 70 });
    var empties = reveal(".rl-empty", root, { y: 8 });
    var chips = freshList(".rl-chip", root);
    if (chips.length) {
      if (!reduce) animate(chips, { opacity: [0, 1], translateY: [8, 0], delay: stagger(60), duration: 400, ease: "outCubic" });
      else chips.forEach(function (c) { c.style.opacity = 1; });
    }
    counters(root, 100);
    bars(root, hits && hits.length ? 200 : 100);
    reveal(".rl-sec", root, { y: 8, step: 50 });
    reveal(".rl-pending", root, { y: 8 });
    reveal(".rl-row", root, { y: 8, step: 20 });
    $$("table.rl-table", root).forEach(wireTable);
    reveal(".rl-fig", root, { step: 60 });
    wireGallery(root);
    if (pulse && (hits.length || empties.length || chips.length)) stopPulse(false);
    syncPipe();
    watchAnswer();
  }

  /* re-run entrance animations for the tab the user just opened */
  function replayVisiblePanel() {
    var panels = $$('[role="tabpanel"]').filter(function (p) { return p.offsetParent !== null; });
    panels.forEach(function (p) {
      $$("[data-rl-done]", p).forEach(function (e) {
        if (e.matches(".rl-hit, .rl-row, .rl-fig, .rl-sec, .rl-pending, .rl-bar i, [data-count], .rl-node")) delete e.dataset.rlDone;
      });
      scan(p);
    });
  }

  var queued = 0;
  function schedule() {
    if (queued) return;
    queued = setTimeout(function () { queued = 0; scan(); }, 60);
  }

  function boot() {
    scan();
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });

    document.addEventListener("click", function (e) {
      var t = e.target;
      if (t.closest && t.closest('button[role="tab"]')) setTimeout(replayVisiblePanel, 60);
      var ask = t.closest && t.closest("#rl-ask");
      if (ask) {
        if (!reduce) animate(ask, { scale: [0.97, 1], duration: 220, ease: "outQuad" });
        syncPipe(); startPulse();
      }
    }, true);
    ["input", "change", "click", "keyup"].forEach(function (ev) {
      document.addEventListener(ev, function () { setTimeout(syncPipe, 80); }, true);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && e.target.closest && e.target.closest("#rl-question")) startPulse();
    }, true);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
