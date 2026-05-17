const $ = (id) => document.getElementById(id);
const fmt = (n, d=1) => (n == null ? '—' : Number(n).toFixed(d));

let lastOkTs = null;

async function fetchStatus() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    render(s);
    lastOkTs = Date.now();
    $('footer').textContent = 'updated ' + s.ts;
    $('footer').classList.remove('err');
  } catch (e) {
    $('footer').textContent = 'fetch error: ' + e.message;
    $('footer').classList.add('err');
  }
  updateAgo();
}

function setBar(barId, pct, threshGood=70, threshBad=90) {
  const el = $(barId);
  if (!el) return;
  pct = Math.max(0, Math.min(100, pct || 0));
  el.style.width = pct + '%';
  el.parentElement.classList.remove('warn','bad');
  if (pct >= threshBad) el.parentElement.classList.add('bad');
  else if (pct >= threshGood) el.parentElement.classList.add('warn');
}

function render(s) {
  $('run').textContent = s.run || '—';
  $('now-ep').textContent = s.now?.ep ?? '—';
  $('now-batch').textContent = s.now?.batch ?? '—';

  const last = s.epochs?.length ? s.epochs[s.epochs.length - 1] : null;
  $('last-p1').textContent = last ? fmt(last.te_p1, 2) + '%' : '—';
  $('last-p5').textContent = last ? fmt(last.te_p5, 2) + '%' : '—';
  $('best').textContent = s.best ? fmt(s.best.p1, 2) + '% (ep ' + s.best.ep + ')' : '—';

  if (s.gpu) {
    const pct = (s.gpu.used_gb / s.gpu.total_gb) * 100;
    $('gpu-txt').textContent = `${fmt(s.gpu.used_gb,1)} / ${fmt(s.gpu.total_gb,1)} GB`;
    setBar('gpu-bar', pct, 80, 95);
    $('gpu-util').textContent = s.gpu.util_pct + '%';
    setBar('util-bar', s.gpu.util_pct, 50, 0);  // util high is good, don't color
  }
  if (s.ram) {
    const pct = (s.ram.used_gb / s.ram.total_gb) * 100;
    $('ram-txt').textContent = `${s.ram.used_gb} / ${s.ram.total_gb} GB`;
    setBar('ram-bar', pct, 70, 90);
  }
  if (s.disk) {
    const pct = (s.disk.used_gb / s.disk.total_gb) * 100;
    $('disk-txt').textContent = `${fmt(s.disk.used_gb,1)} / ${s.disk.total_gb} GB`;
    setBar('disk-bar', pct, 70, 90);
  }

  const tbody = $('epochs');
  tbody.innerHTML = '';
  const bestEp = s.best?.ep;
  for (const e of (s.epochs || []).slice().reverse()) {
    const tr = document.createElement('tr');
    if (e.ep === bestEp) tr.classList.add('best');
    tr.innerHTML = `<td>${e.ep}</td><td>${fmt(e.tr_acc,1)}</td><td>${fmt(e.tr_loss,3)}</td><td>${fmt(e.te_p1,2)}</td><td>${fmt(e.te_p5,2)}</td>`;
    tbody.appendChild(tr);
  }
}

function updateAgo() {
  const el = $('ago');
  if (!lastOkTs) { el.textContent = '…'; return; }
  const s = Math.round((Date.now() - lastOkTs) / 1000);
  el.textContent = s < 60 ? `${s}s ago` : `${Math.floor(s/60)}m ${s%60}s ago`;
  el.classList.toggle('stale', s > 30);
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(()=>{});
}

fetchStatus();
setInterval(fetchStatus, 10000);
setInterval(updateAgo, 1000);
