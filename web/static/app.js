'use strict';

// --- State ---

const State = {
  screen: 'landing',
  levels: [],
  currentLevelId: null,
  currentLevelData: null,
  completed: JSON.parse(localStorage.getItem('prm_completed') || '{}'),
  walkthroughDone: localStorage.getItem('prm_wt_done') === '1',
  wtStep: 0,
  evalRunning: false,
  sandboxRunning: false,
};

function saveCompleted() {
  localStorage.setItem('prm_completed', JSON.stringify(State.completed));
}

// --- Walkthrough steps ---

const WT_STEPS = [
  {
    icon: '🧪',
    title: 'Welcome to PRM Research Lab',
    body: `You are an AI researcher. Your mission: build a <strong>Process Reward Model (PRM)</strong> for mathematical reasoning and submit it to an automated 7-gate judge — without triggering any reward hacks.`,
  },
  {
    icon: '🎯',
    title: 'What is a Process Reward Model?',
    body: `Unlike Outcome Reward Models that only check if the <em>final answer</em> is correct, a PRM scores <strong>each individual reasoning step</strong>. A good PRM catches errors exactly where they occur — not just at the end.`,
    visual: `
      <div class="wt-steps-demo">
        <div class="wt-step-demo ok"><span>Step 1: 2+3 = 5</span><span>0.94</span></div>
        <div class="wt-step-demo ok"><span>Step 2: 5×4 = 20</span><span>0.91</span></div>
        <div class="wt-step-demo err"><span>Step 3: 20÷5 = 3 ❌</span><span>0.08</span></div>
        <div class="wt-step-demo err"><span>Step 4: 3+1 = 4</span><span>0.05</span></div>
      </div>`,
  },
  {
    icon: '⚡',
    title: 'The 7-Gate Judge',
    body: `Your submission is evaluated through 7 sequential gates. <strong>Fail any gate → score = 0.</strong> The gates are mutually adversarial — you can't game one without breaking another.`,
    visual: `
      <div class="wt-gates-grid">
        <div class="wt-gate-item">📋 Existence &amp; Signature</div>
        <div class="wt-gate-item">🚫 No Banned Checkpoints</div>
        <div class="wt-gate-item">⚡ FLOP Budget</div>
        <div class="wt-gate-item">📊 Capability Floor</div>
        <div class="wt-gate-item">🚀 BoN-32 Lift</div>
        <div class="wt-gate-item">🎯 Step Localization F1</div>
        <div class="wt-gate-item" style="grid-column:span 2">⚖️ Calibration (ECE)</div>
      </div>`,
  },
  {
    icon: '💀',
    title: 'Three Ways to Fail',
    body: `Common hacks — and why they don't work:`,
    visual: `
      <div class="wt-hacks">
        <div class="wt-hack"><span class="wt-hack-icon">🎲</span><div><strong>Naive:</strong> Return 0.5 for everything → fails Capability Floor (Gate 4)</div></div>
        <div class="wt-hack"><span class="wt-hack-icon">🎭</span><div><strong>Outcome Shortcut:</strong> Only check the final answer → fails Step Localization (Gate 6)</div></div>
        <div class="wt-hack"><span class="wt-hack-icon">🎮</span><div><strong>Threshold Gaming:</strong> Fake the score shape → fails BoN-32 Lift (Gate 5)</div></div>
      </div>`,
  },
  {
    icon: '🏆',
    title: 'Your Mission',
    body: `Work through <strong>4 levels</strong>, each demonstrating a different submission strategy with real judge output. Understand the failure modes — then win with Level 4 by passing all 7 gates. The lab is open.`,
    cta: true,
  },
];

// --- Syntax highlighter ---

function highlight(src) {
  let out = src
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const store = [];
  function protect(html) {
    const idx = store.length;
    store.push(html);
    return `\x00${idx}\x00`;
  }

  out = out.replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g,
    m => protect(`<span class="h-string">${m}</span>`));
  out = out.replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g,
    m => protect(`<span class="h-string">${m}</span>`));
  out = out.replace(/(#[^\n]*)/g,
    m => protect(`<span class="h-comment">${m}</span>`));
  out = out.replace(/(^|\n)(@[A-Za-z_]\w*)/g,
    (_, pre, dec) => pre + protect(`<span class="h-dec">${dec}</span>`));

  const KW = /\b(def|class|return|import|from|if|else|elif|for|while|in|not|and|or|True|False|None|as|with|try|except|finally|raise|assert|lambda|pass|break|continue|yield|async|await|global|nonlocal|del|is)\b/g;
  out = out.replace(KW, m => protect(`<span class="h-kw">${m}</span>`));

  const BUILTINS = /\b(print|len|range|list|dict|set|tuple|str|int|float|bool|type|isinstance|getattr|setattr|hasattr|max|min|abs|sum|zip|map|filter|enumerate|super|staticmethod|classmethod|property)\b/g;
  out = out.replace(BUILTINS, m => protect(`<span class="h-builtin">${m}</span>`));

  out = out.replace(/\bdef\s+([A-Za-z_]\w*)/g,
    (_, n) => `def ` + protect(`<span class="h-func">${n}</span>`));
  out = out.replace(/\bclass\s+([A-Za-z_]\w*)/g,
    (_, n) => `class ` + protect(`<span class="h-class">${n}</span>`));
  out = out.replace(/\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g,
    m => protect(`<span class="h-num">${m}</span>`));

  out = out.replace(/\x00(\d+)\x00/g, (_, i) => store[+i]);
  return out;
}

// --- DOM helpers ---

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function showScreen(name) {
  document.querySelectorAll('.screen').forEach(el => {
    el.classList.toggle('active', false);
    el.classList.toggle('hidden', true);
  });
  const el = $(`screen-${name}`);
  if (el) {
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('active'), 10);
  }
  State.screen = name;
}

function animateCount(el, target, duration = 1000) {
  const start = Date.now();
  function tick() {
    const t = Math.min((Date.now() - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * ease).toFixed(3);
    if (t < 1) requestAnimationFrame(tick);
    else el.textContent = target.toFixed(3);
  }
  requestAnimationFrame(tick);
}

// --- API ---

async function fetchLevels() {
  const r = await fetch('/api/levels');
  return r.json();
}

async function fetchLevel(id) {
  const r = await fetch(`/api/level/${id}`);
  return r.json();
}

async function fetchPrompt() {
  const r = await fetch('/api/prompt');
  return r.json();
}

async function evaluateLevel(id) {
  const r = await fetch(`/api/evaluate/${id}`, { method: 'POST' });
  return r.json();
}

async function evaluateSandbox(code) {
  const r = await fetch('/api/sandbox', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  return r.json();
}

// --- Map screen ---

function renderMap() {
  const grid = $('map-grid');
  grid.innerHTML = '';

  const cleared = Object.values(State.completed).filter(c => c.passed).length;
  $('map-progress').textContent = `${cleared} / 4 cleared`;

  State.levels.forEach(lv => {
    const done = State.completed[lv.id];
    const card = document.createElement('div');
    card.className = `level-card${done ? ' completed' : ''}`;
    card.onclick = () => App.goLevel(lv.id);

    const scoreStr = done
      ? `<span class="card-score">✓ ${done.score.toFixed(3)}</span>`
      : '';

    card.innerHTML = `
      <div class="card-top">
        <div class="card-icon">${lv.icon}</div>
        <div class="card-meta">
          <div class="card-num">Level ${lv.number}</div>
          <div class="card-title">${lv.title}</div>
          <span class="diff-badge" style="color:${lv.difficulty_color};background:${lv.difficulty_color}18;border:1px solid ${lv.difficulty_color}40">${lv.difficulty}</span>
        </div>
        <div class="card-status">${scoreStr}</div>
      </div>
      <p class="card-teaser">${lv.teaser}</p>
      <div class="card-footer">
        <span class="card-subtitle">${lv.subtitle}</span>
        <button class="btn btn-primary btn-sm" onclick="event.stopPropagation();App.goLevel('${lv.id}')">Play →</button>
      </div>`;
    grid.appendChild(card);
  });

  const sbCard = document.createElement('div');
  sbCard.className = 'level-card';
  sbCard.onclick = () => App.goSandbox();
  sbCard.innerHTML = `
    <div class="card-top">
      <div class="card-icon">🔬</div>
      <div class="card-meta">
        <div class="card-num">Free Play</div>
        <div class="card-title">Sandbox</div>
        <span class="diff-badge" style="color:#a78bfa;background:#a78bfa18;border:1px solid #a78bfa40">Custom</span>
      </div>
    </div>
    <p class="card-teaser">Write your own PRM submission and run it through the real judge.</p>
    <div class="card-footer">
      <span class="card-subtitle">Open editor</span>
      <button class="btn btn-sm" style="border-color:#a78bfa;color:#a78bfa" onclick="event.stopPropagation();App.goSandbox()">Open →</button>
    </div>`;
  grid.appendChild(sbCard);
}

// --- Level screen ---

async function populateLevel(data) {
  $('level-nav-title').textContent = `Level ${data.number}: ${data.title}`;
  $('level-icon').textContent = data.icon;
  $('level-number').textContent = `Level ${data.number}`;
  $('level-title').textContent = data.title;
  $('level-subtitle').textContent = data.subtitle;

  const badge = $('level-badge');
  badge.textContent = data.difficulty;
  badge.style.cssText = `color:${data.difficulty_color};background:${data.difficulty_color}18;border:1px solid ${data.difficulty_color}40`;

  $('level-description').textContent = data.description;
  $('level-hint').textContent = data.gates_hint;
  $('level-theory').textContent = data.theory;
  $('level-code').innerHTML = highlight(data.code || '');

  $('results-empty').classList.remove('hidden');
  $('results-panel').classList.add('hidden');

  App.switchTab('mission', document.querySelector('.tab[data-tab="mission"]'));
}

// --- Results rendering ---

async function renderResults(data, { gatesId, verdictId, scoreBlockId, scoreValId, scoreBarId, actionsId }) {
  const passedAll = data.passed_all;

  const verdictEl = $(verdictId);
  verdictEl.textContent = passedAll ? '✓ ALL GATES PASSED' : '✗ FAILED';
  verdictEl.className = `verdict-badge ${passedAll ? 'verdict-pass' : 'verdict-fail'}`;

  const container = $(gatesId);
  container.innerHTML = '';

  for (let i = 0; i < (data.gates || []).length; i++) {
    await sleep(160);
    const g = data.gates[i];
    const status = g.passed === true ? 'pass' : g.passed === false ? 'fail' : 'skip';
    const pillLabel = g.passed === true ? 'PASS' : g.passed === false ? 'FAIL' : 'SKIP';

    const card = document.createElement('div');
    card.className = `gate-card ${status}`;
    card.innerHTML = `
      <div class="gate-row">
        <span class="gate-icon-el">${g.icon}</span>
        <span class="gate-short">${g.short}</span>
        <span class="gate-label">${g.label}</span>
        <span class="gate-pill">${pillLabel}</span>
      </div>
      <div class="gate-detail">${g.detail}</div>`;
    container.appendChild(card);
  }

  await sleep(200);

  if (scoreBlockId) {
    $(scoreBlockId).classList.remove('hidden');
    animateCount($(scoreValId), data.final_score || 0, 900);
    setTimeout(() => {
      const bar = $(scoreBarId);
      if (bar) bar.style.width = `${Math.min((data.final_score || 0) * 100, 100)}%`;
    }, 100);
  }

  if (actionsId) {
    $(actionsId).classList.remove('hidden');
    const curIdx = State.levels.findIndex(l => l.id === State.currentLevelId);
    const nextBtn = $('next-btn');
    if (nextBtn) nextBtn.style.display = curIdx < State.levels.length - 1 ? '' : 'none';
  }
}

// --- Walkthrough ---

function renderWTStep() {
  const step = WT_STEPS[State.wtStep];
  $('wt-content').innerHTML = `
    <div class="wt-step-icon">${step.icon}</div>
    <div class="wt-step-title">${step.title}</div>
    <div class="wt-step-body">${step.body}</div>
    ${step.visual || ''}
  `;

  $('wt-dots').innerHTML = WT_STEPS.map((_, i) =>
    `<div class="wt-dot${i === State.wtStep ? ' active' : ''}"></div>`
  ).join('');

  $('wt-prev').style.visibility = State.wtStep === 0 ? 'hidden' : 'visible';
  $('wt-next').textContent = State.wtStep >= WT_STEPS.length - 1 ? 'Enter the Lab →' : 'Next →';
}

// --- App ---

const App = {

  async init() {
    State.levels = await fetchLevels();
    showScreen('landing');
  },

  startGame() {
    if (!State.walkthroughDone) {
      State.wtStep = 0;
      renderWTStep();
      $('modal-walkthrough').classList.remove('hidden');
    } else {
      App.goMap();
    }
  },

  goMap() {
    renderMap();
    showScreen('map');
  },

  async goLevel(id) {
    State.currentLevelId = id;
    if (!State.currentLevelData || State.currentLevelData.id !== id) {
      State.currentLevelData = await fetchLevel(id);
    }
    await populateLevel(State.currentLevelData);
    showScreen('level');
  },

  goSandbox() {
    showScreen('sandbox');
    $('sandbox-empty').classList.remove('hidden');
    $('sandbox-panel').classList.add('hidden');
  },

  switchTab(name, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    const pane = $(`tab-${name}`);
    if (pane) { pane.classList.remove('hidden'); pane.classList.add('active'); }
    if (btn) btn.classList.add('active');
  },

  async runEval() {
    if (State.evalRunning) return;
    State.evalRunning = true;

    const btn = $('run-btn');
    btn.innerHTML = '<span class="spinner"></span> Evaluating…';
    btn.classList.add('loading');

    $('results-empty').classList.add('hidden');
    $('results-panel').classList.remove('hidden');
    $('gates-container').innerHTML = '';
    $('score-block').classList.add('hidden');
    $('results-actions').classList.add('hidden');
    $('results-verdict').textContent = '';
    $('results-verdict').className = 'verdict-badge';

    try {
      const data = await evaluateLevel(State.currentLevelId);
      if (data.error) {
        $('gates-container').innerHTML =
          `<div class="gate-card fail"><div class="gate-row"><span class="gate-pill">ERROR</span></div><div class="gate-detail">${data.detail || data.error}</div></div>`;
      } else {
        State.completed[State.currentLevelId] = {
          passed: data.passed_all,
          score: data.final_score,
          ts: Date.now(),
        };
        saveCompleted();
        await renderResults(data, {
          gatesId: 'gates-container',
          verdictId: 'results-verdict',
          scoreBlockId: 'score-block',
          scoreValId: 'score-value',
          scoreBarId: 'score-bar',
          actionsId: 'results-actions',
        });
      }
    } catch (e) {
      $('gates-container').innerHTML =
        `<div class="gate-card fail"><div class="gate-row"><span class="gate-pill">ERROR</span></div><div class="gate-detail">Network error: ${e.message}</div></div>`;
    }

    btn.innerHTML = '<span class="run-icon">▶</span> Run Evaluation';
    btn.classList.remove('loading');
    State.evalRunning = false;
  },

  async runSandbox() {
    if (State.sandboxRunning) return;
    const code = $('sandbox-code').value.trim();
    if (!code) { alert('Please write some code first.'); return; }

    State.sandboxRunning = true;
    const btn = document.querySelector('#screen-sandbox .btn-run');
    btn.innerHTML = '<span class="spinner"></span> Evaluating…';
    btn.classList.add('loading');

    $('sandbox-empty').classList.add('hidden');
    $('sandbox-panel').classList.remove('hidden');
    $('sandbox-gates').innerHTML = '';
    $('sandbox-score-block').classList.add('hidden');
    $('sandbox-verdict').textContent = '';
    $('sandbox-verdict').className = 'verdict-badge';

    try {
      const data = await evaluateSandbox(code);
      if (data.error) {
        $('sandbox-gates').innerHTML =
          `<div class="gate-card fail"><div class="gate-row"><span class="gate-pill">ERROR</span></div><div class="gate-detail">${data.detail || data.error}</div></div>`;
      } else {
        await renderResults(data, {
          gatesId: 'sandbox-gates',
          verdictId: 'sandbox-verdict',
          scoreBlockId: 'sandbox-score-block',
          scoreValId: 'sandbox-score-value',
          scoreBarId: 'sandbox-score-bar',
          actionsId: null,
        });
      }
    } catch (e) {
      $('sandbox-gates').innerHTML =
        `<div class="gate-card fail"><div class="gate-row"><span class="gate-pill">ERROR</span></div><div class="gate-detail">Network error: ${e.message}</div></div>`;
    }

    btn.innerHTML = '<span class="run-icon">▶</span> Run Evaluation';
    btn.classList.remove('loading');
    State.sandboxRunning = false;
  },

  loadTemplate() {
    $('sandbox-code').value = `"""
PRM Submission
"""
from __future__ import annotations
from typing import List


class MyPRM:
    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        # Return a score in [0, 1] per step.
        # Higher = more likely the prefix up to this step is correct.
        return [0.5] * len(step_texts)


def load_prm() -> MyPRM:
    return MyPRM()


MODEL_INFO = {
    "base": "qwen2.5-math-1.5b",
    "uses_lora": True,
    "trained_steps": 0,
    "labeling_method": "none",
}
`;
  },

  async showPrompt() {
    const data = await fetchPrompt();
    $('prompt-content').textContent = data.prompt || '';
    $('modal-prompt').classList.remove('hidden');
  },

  closePrompt() {
    $('modal-prompt').classList.add('hidden');
  },

  wtNext() {
    if (State.wtStep < WT_STEPS.length - 1) {
      State.wtStep++;
      renderWTStep();
    } else {
      $('modal-walkthrough').classList.add('hidden');
      State.walkthroughDone = true;
      localStorage.setItem('prm_wt_done', '1');
      App.goMap();
    }
  },

  wtPrev() {
    if (State.wtStep > 0) {
      State.wtStep--;
      renderWTStep();
    }
  },

  nextLevel() {
    const curIdx = State.levels.findIndex(l => l.id === State.currentLevelId);
    if (curIdx < State.levels.length - 1) {
      App.goLevel(State.levels[curIdx + 1].id);
    } else {
      App.goMap();
    }
  },
};

document.addEventListener('DOMContentLoaded', () => App.init());
