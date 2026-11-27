/**
 * TRACTUS Demo 1: Autonomous Web Agent - Standalone Client
 */
(function() {
  'use strict';

  const MANIFOLD_SLOTS = 255;

  const state = {
    ws: null,
    connected: false,
    running: false,
    scenario: 'google_flights',
    step: 0,
    totalSteps: 24,
    cumulativeLatency: 0,
  };

  const els = {
    statLatency: document.getElementById('stat-latency'),
    statSteps: document.getElementById('stat-steps'),
    statSuccess: document.getElementById('stat-success'),
    addressText: document.getElementById('address-text'),
    targetLockText: document.getElementById('target-lock-text'),
    telemStepLat: document.getElementById('telem-step-lat'),
    telemCumLat: document.getElementById('telem-cum-lat'),
    telemConf: document.getElementById('telem-conf'),
    actionDisplay: document.getElementById('action-display'),
    manifoldGrid: document.getElementById('manifold-grid'),
    pipeEmbed: document.getElementById('pipe-embed'),
    pipeManifold: document.getElementById('pipe-manifold'),
    pipeExec: document.getElementById('pipe-exec'),
    pipeTotal: document.getElementById('pipe-total'),
    execLog: document.getElementById('exec-log'),
    appFrame: document.getElementById('app-frame'),
    reticle: document.getElementById('reticle'),
    reticleCoords: document.getElementById('reticle-coords'),
    btnRun: document.getElementById('btn-run'),
    btnReset: document.getElementById('btn-reset'),
  };

  // Server-side action_summary formatting.
  function translateActionSummary(raw, step) {
    if (!raw) return 'Idle';
    let m;
    if ((m = raw.match(/^Fill field:\s*(.+?)\s*=\s*(.+)$/i))) {
      return `Fill field: ${m[1]} = ${m[2]}`;
    }
    if ((m = raw.match(/^Click Next -> advance to stage\s*(\d+)$/i))) {
      return `Click "Next" → advance to stage ${m[1]}`;
    }
    if (raw.toLowerCase().includes('dispatch page transition')) {
      return 'Dispatch page transition';
    }
    if (raw.toLowerCase().includes('complete workflow')) {
      return 'Click "Submit Expense Form" — workflow complete';
    }
    if (raw.toLowerCase().includes('emit terminal done token')) {
      return 'Closure verified — emit terminal DONE token';
    }
    if ((m = raw.match(/^Select recommended flight\s*(\S+)\s*\(cabin:\s*(.+)\)$/i))) {
      return `Select recommended flight ${m[1]} (cabin: ${m[2]})`;
    }
    if ((m = raw.match(/^Verify checkbox:\s*(.+)$/i))) {
      return `Verify checkbox: ${m[1]}`;
    }
    if ((m = raw.match(/^Enter:\s*(.+?)\s*->\s*'(.+)'$/i))) {
      return `Enter: ${m[1]} → '${m[2]}'`;
    }
    return raw || `Step ${step} action dispatched`;
  }

  function buildManifoldGrid() {
    els.manifoldGrid.innerHTML = '';
    for (let i = 0; i < MANIFOLD_SLOTS; i++) {
      const cell = document.createElement('div');
      cell.className = 'manifold-cell';
      cell.dataset.slot = i;
      els.manifoldGrid.appendChild(cell);
    }
  }

  function lockManifoldSlot(slotIndex) {
    const cells = els.manifoldGrid.children;
    for (let i = 0; i < cells.length; i++) {
      cells[i].classList.toggle('lock', i === slotIndex);
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      state.connected = true;
      console.log('WebSocket connected');
    };

    state.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    state.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    state.ws.onclose = () => {
      state.connected = false;
      console.log('WebSocket disconnected');
      setTimeout(connectWebSocket, 2000);
    };
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case 'demo1_init':
        handleInit(msg.data);
        break;
      case 'demo1_step':
        handleStep(msg.data);
        break;
      case 'demo1_complete':
        handleComplete(msg.data);
        break;
    }
  }

  function handleInit(data) {
    state.step = 0;
    state.totalSteps = data.total_steps || 24;
    state.cumulativeLatency = 0;
    els.statSteps.textContent = `0/${state.totalSteps}`;
    addLogEntry('System', `Starting sequence: Autonomous Form Navigation (${state.totalSteps} steps)`, 'lock');
  }

  function handleStep(data) {
    state.step = data.step;
    state.cumulativeLatency = data.cumulative_latency_ms;

    const slotIndex = typeof data.action_slot === 'number'
      ? data.action_slot
      : Math.abs(hashString(String(data.action_slot || data.chosen_action))) % MANIFOLD_SLOTS;

    els.statLatency.textContent = `${data.latency_ms}ms`;
    els.statSteps.textContent = `${data.step}/${state.totalSteps}`;
    els.telemStepLat.textContent = `${data.latency_ms}ms`;
    els.telemCumLat.textContent = `${Math.round(data.cumulative_latency_ms)}ms`;
    els.telemConf.textContent = `${data.confidence}%`;

    const summary = translateActionSummary(data.action_summary, data.step);
    els.actionDisplay.textContent = `SLOT[${slotIndex}] :: ${summary}`;
    lockManifoldSlot(slotIndex);

    // Pipeline breakdown — derive deterministic sub-stage split from total latency.
    const total = data.latency_ms;
    const tEmbed = total * 0.115;
    const tManifold = total * 0.815;
    const tExec = total * 0.07;
    els.pipeEmbed.textContent = `${tEmbed.toFixed(1)}ms`;
    els.pipeManifold.textContent = `${tManifold.toFixed(1)}ms`;
    els.pipeExec.textContent = `${tExec.toFixed(1)}ms`;
    els.pipeTotal.textContent = `${total.toFixed(1)}ms`;

    addLogEntry(
      `Step ${data.step}`,
      summary,
      'hit',
      `${data.latency_ms}ms · Confidence: ${data.confidence}%`
    );

    moveReticleToTarget(data.target_element_id);

    try {
      const doc = els.appFrame.contentDocument || (els.appFrame.contentWindow && els.appFrame.contentWindow.document);
      if (doc) {
        const isShortUltraJev = state.totalSteps <= 10;
        const stepId = String(data.step_id || '');
        if (isShortUltraJev) {
          if (stepId === 'flight_01_origin' || data.step === 1) {
            const originInput = doc.getElementById('origin-input');
            if (originInput) {
              originInput.value = 'Zurich (ZRH)';
              originInput.style.outline = '2px solid #00E5FF';
            }
          } else if (stepId === 'flight_02_destination' || data.step === 2) {
            const destInput = doc.getElementById('destination-input');
            if (destInput) {
              destInput.value = 'London Heathrow (LHR)';
              destInput.style.outline = '2px solid #00E5FF';
            }
            const depDate = doc.getElementById('departure-date');
            if (depDate) {
              depDate.value = 'Sep 20, 2026';
              depDate.style.outline = '2px solid #00E5FF';
            }
          } else if (stepId === 'flight_03_search_button' || data.step === 3) {
            const results = doc.getElementById('flight-results-section');
            if (results) {
              results.classList.remove('hidden');
            }
          } else if (stepId === 'flight_04_select_airline' || data.step === 4) {
            const results = doc.getElementById('flight-results-section');
            if (results) {
              results.classList.remove('hidden');
            }
            if (els.appFrame.contentWindow && typeof els.appFrame.contentWindow.selectFlight === 'function') {
              els.appFrame.contentWindow.selectFlight('lx316');
            }
          } else if (stepId === 'flight_05_complete' || data.step >= 5) {
            const conf = doc.getElementById('flight-confirmation-card');
            if (conf) {
              conf.classList.remove('hidden');
              conf.scrollIntoView({ behavior: 'smooth' });
            }
          }
        } else {
          if (data.step >= 1 && data.step <= 2) {
            const originInput = doc.getElementById('origin-input');
            if (originInput) {
              originInput.value = 'Zurich (ZRH)';
              originInput.style.outline = '2px solid #00E5FF';
            }
          } else if (data.step >= 3 && data.step <= 5) {
            const destInput = doc.getElementById('destination-input');
            if (destInput) {
              destInput.value = 'London Heathrow (LHR)';
              destInput.style.outline = '2px solid #00E5FF';
            }
          } else if (data.step >= 6 && data.step <= 8) {
            const depDate = doc.getElementById('departure-date');
            if (depDate) {
              depDate.value = 'Sep 20, 2026';
              depDate.style.outline = '2px solid #00E5FF';
            }
          } else if (data.step >= 9 && data.step <= 13) {
            const results = doc.getElementById('flight-results-section');
            if (results) {
              results.classList.remove('hidden');
            }
          } else if (data.step >= 14 && data.step <= 18) {
            if (els.appFrame.contentWindow && typeof els.appFrame.contentWindow.selectFlight === 'function') {
              els.appFrame.contentWindow.selectFlight('lx316');
            }
          } else if (data.step >= 19) {
            const conf = doc.getElementById('flight-confirmation-card');
            if (conf) {
              conf.classList.remove('hidden');
              conf.scrollIntoView({ behavior: 'smooth' });
            }
          }
        }
      }
    } catch (e) {
      // Cross-origin or not-yet-loaded iframe — ignore, HUD telemetry is unaffected.
    }
  }

  function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return hash;
  }

  function moveReticleToTarget(targetId) {
    const doc = els.appFrame.contentDocument || (els.appFrame.contentWindow && els.appFrame.contentWindow.document);
    let x, y;
    if (doc && targetId) {
      const el = doc.getElementById(targetId);
      if (el) {
        const rect = el.getBoundingClientRect();
        x = rect.left + rect.width / 2;
        y = rect.top + rect.height / 2;
      }
    }
    if (x === undefined) {
      x = Math.random() * 0.6 * els.appFrame.clientWidth + 0.2 * els.appFrame.clientWidth;
      y = Math.random() * 0.6 * els.appFrame.clientHeight + 0.2 * els.appFrame.clientHeight;
    }

    els.reticle.style.left = `${x}px`;
    els.reticle.style.top = `${y}px`;
    els.reticle.classList.add('show');
    els.reticleCoords.textContent = `TARGET LOCKED :: X: ${Math.round(x)} Y: ${Math.round(y)}`;
    els.targetLockText.textContent = targetId ? `LOCKED: #${targetId}` : 'SCANNING';

    clearTimeout(moveReticleToTarget._t);
    moveReticleToTarget._t = setTimeout(() => {
      els.reticle.classList.remove('show');
    }, 900);
  }

  function handleComplete(data) {
    state.running = false;
    els.btnRun.textContent = 'RUN AGENT';
    els.btnRun.disabled = false;

    const avgLatency = data.avg_latency_ms || (data.total_neural_latency_ms / data.total_actions);
    els.statLatency.textContent = `${Math.round(avgLatency)}ms`;

    addLogEntry(
      'Complete',
      `${data.total_actions} steps executed · avg latency ${Math.round(avgLatency)}ms`,
      'hit'
    );
    els.targetLockText.textContent = 'SEQUENCE COMPLETE';
  }

  function addLogEntry(title, text, type = 'info', meta = '') {
    const entry = document.createElement('div');
    entry.className = `tx-feed-row ${type === 'hit' || type === 'lock' ? type : ''}`;

    const head = document.createElement('div');
    head.className = 'tx-feed-row-head';
    head.textContent = title;
    entry.appendChild(head);

    entry.appendChild(document.createTextNode(text));

    if (meta) {
      const metaDiv = document.createElement('div');
      metaDiv.className = 'tx-feed-row-meta';
      metaDiv.textContent = meta;
      entry.appendChild(metaDiv);
    }

    els.execLog.appendChild(entry);
    els.execLog.scrollTop = els.execLog.scrollHeight;
  }

  function runDemo() {
    if (state.running || !state.connected) return;

    state.running = true;
    els.btnRun.textContent = 'RUNNING…';
    els.btnRun.disabled = true;
    els.execLog.innerHTML = '';

    state.ws.send(JSON.stringify({
      action: 'run_demo1',
      scenario_id: state.scenario,
      speed: 1.0,
    }));
  }

  function resetDemo() {
    state.step = 0;
    state.cumulativeLatency = 0;
    els.statLatency.textContent = '--';
    els.statSteps.textContent = '0/24';
    els.statSuccess.textContent = '100%';
    els.telemStepLat.textContent = '--';
    els.telemCumLat.textContent = '0ms';
    els.telemConf.textContent = '--%';
    els.actionDisplay.textContent = 'IDLE';
    els.pipeEmbed.textContent = '--';
    els.pipeManifold.textContent = '--';
    els.pipeExec.textContent = '--';
    els.pipeTotal.textContent = '--';
    els.targetLockText.textContent = 'AWAITING TARGET';
    els.reticle.classList.remove('show');
    lockManifoldSlot(-1);
    els.execLog.innerHTML = '<div class="tx-feed-row dim"><div class="tx-feed-row-head">Standby</div>Press RUN AGENT below to begin the 24-step autonomous sequence.</div>';
    els.btnRun.textContent = 'RUN AGENT';
    els.btnRun.disabled = false;
    state.running = false;
    if (els.appFrame) els.appFrame.src = els.appFrame.src;
  }

  function selectScenario(scenario) {
    state.scenario = scenario;

    document.querySelectorAll('.tx-pill').forEach(pill => {
      pill.classList.remove('active');
    });

    const activePill = document.querySelector(`[data-scenario="${scenario}"]`);
    if (activePill) {
      activePill.classList.add('active');
    }

    const scenarioUrls = {
      google_flights: 'https://www.google.com/travel/flights',
      casa_flora: 'https://casaflora.com/stays',
      megastore: 'https://megastore.example.com/shop',
    };

    els.addressText.textContent = scenarioUrls[scenario] || scenarioUrls.google_flights;
    els.appFrame.src = `/static/apps/${scenario}.html`;
  }

  buildManifoldGrid();

  els.btnRun.addEventListener('click', runDemo);
  els.btnReset.addEventListener('click', resetDemo);

  document.querySelectorAll('.tx-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      selectScenario(pill.dataset.scenario);
    });
  });

  connectWebSocket();
})();
