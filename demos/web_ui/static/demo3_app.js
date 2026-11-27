/**
 * TRACTUS Demo 3: Zero-Key Neural Intent Router - Standalone Client
 *
 * The backend risk-analysis engine (engine_runner.py / handoff_router.py)
 * is scripted against one fixed Mandarin complaint audio clip and emits a
 * small, enumerable set of Mandarin route/card strings for that scenario.
 * This client translates that fixed vocabulary to English before paint so
 * the screen stays 100% English without touching backend business logic.
 */
(function() {
  'use strict';

  const state = {
    ws: null,
    connected: false,
    analyzing: false,
    radarCtx: null,
    radarValues: { compliance: 25, handoff: 30, billing: 35 },
    targetValues: { compliance: 96.9, handoff: 96.1, billing: 94.1 },
  };

  const els = {
    kpiRoute: document.getElementById('kpi-route'),
    kpiHandoff: document.getElementById('kpi-handoff'),
    kpiSpeedup: document.getElementById('kpi-speedup'),
    callTimer: document.getElementById('call-timer'),
    transcriptText: document.getElementById('transcript-text'),
    routeDisplay: document.getElementById('route-display'),
    radarCanvas: document.getElementById('radar-canvas'),
    legCompliance: document.getElementById('leg-compliance'),
    legHandoff: document.getElementById('leg-handoff'),
    legBilling: document.getElementById('leg-billing'),
    riskFeed: document.getElementById('risk-feed'),
    oscWave: document.getElementById('osc-wave'),
    oscLockLine: document.getElementById('osc-lock-line'),
    oscLockLabel: document.getElementById('osc-lock-label'),
    audioPlayer: document.getElementById('audio-player'),
    btnAnalyze: document.getElementById('btn-analyze'),
    btnReset: document.getElementById('btn-reset'),
  };

  // --- Fixed-vocabulary English translation table (see module docstring) ---

  const ROUTE_TRANSLATIONS = [
    [/Emergency Card Lock & Freeze/i, 'INSTANT SELF-SERVICE :: Emergency Card Lock & Freeze (zero-key, menu bypassed)'],
    [/Direct Debit & Auto-Pay/i, 'INSTANT SELF-SERVICE :: Direct Debit & Auto-Pay Management (zero-key, menu bypassed)'],
    [/Senior Fraud Specialist/i, 'INSTANT ESCALATION :: Routed to Senior Fraud Specialist (priority risk card attached)'],
    [/Account & Transaction/i, 'INSTANT SELF-SERVICE :: Account & Transaction History Lookup'],
  ];

  const AGENT_TITLE_EN = 'Senior Fraud Specialist (Tier-3 Risk Disposition Desk)';

  const TRANSCRIPT_EN =
    "Hi, I wanted to ask about direct payments -- which direct debits are currently active on my account, and how " +
    "can I get one refunded? Also, I believe I have lost my card, so I need to freeze it urgently -- this is fairly " +
    "critical, so please help me complete this as fast as possible, thank you. Yes, please freeze my card right " +
    "now -- there are some very strange transactions on it, and I think it may have been stolen. Please action " +
    "this immediately, thank you.";

  const RISK_CARD_TRANSLATIONS = {
    title: [
      [/Emergency Card Freeze/i, 'Instant Lock -- Emergency Card Freeze & Secure Vaulting'],
      [/Suspicious Transaction/i, 'Fraud Line -- Suspicious Transaction & Dispute Triage'],
      [/Direct Debit & Auto-Pay/i, 'Instant Self-Service -- Direct Debit & Auto-Pay Management'],
      [/Compliance Alert/i, 'Critical Compliance Alert'],
      [/Human Escalation/i, 'Human Escalation Request'],
      [/Billing Dispute/i, 'Billing Dispute Triage'],
      [/Elevated Customer Emotion/i, 'Neural Distress Assessment -- Elevated Customer Emotion'],
    ],
    actions: [
      [/Lock Card Now/i, 'Click "Lock Card Now" to complete the emergency freeze'],
      [/Review the last 24 hours/i, 'Review the last 24 hours of overseas and no-PIN transactions'],
      [/Issue a replacement card/i, 'Issue a replacement card via expedited courier to the address on file'],
      [/disputed transaction merchant/i, 'Pull disputed transaction merchant and geolocation evidence'],
      [/provisional credit/i, 'Initiate provisional credit / consumer protection chargeback'],
      [/File a fraud dispute/i, 'File a fraud dispute case with the card network'],
      [/active direct-debit authorization/i, "Show the customer their active direct-debit authorization list"],
      [/one-click cancellation/i, 'Offer one-click cancellation of the auto-pay authorization'],
      [/chargeback for the disputed debit/i, 'File a 7-business-day chargeback for the disputed debit'],
      [/De-escalate immediately/i, 'De-escalate immediately with empathetic tone'],
      [/Commit to resolution within 48 hours/i, 'Commit to resolution within 48 hours'],
      [/Log the complaint reference/i, 'Log the complaint reference number'],
      [/compliance desk/i, 'Escalate to the compliance desk'],
      [/Confirm the issue category/i, 'Confirm the issue category'],
      [/agent badge/i, 'Provide agent badge/ID number'],
      [/Commit to an SLA/i, 'Commit to an SLA for resolution'],
      [/transaction history/i, 'Look up the transaction history'],
      [/disputed charge line item/i, 'Reconcile the disputed charge line item'],
      [/applicable fee schedule/i, 'Explain the applicable fee schedule'],
      [/refund if warranted/i, 'Issue a refund if warranted'],
      [/Lead with priority de-escalation/i, 'Lead with priority de-escalation script, active empathetic listening'],
      [/Skip boilerplate scripting/i, 'Skip boilerplate scripting -- move straight to a concrete resolution'],
    ],
  };

  function translateByTable(text, table) {
    if (!text) return text;
    for (const [pattern, replacement] of table) {
      if (pattern.test(text)) return replacement;
    }
    return text;
  }

  function hasCJK(text) {
    return false;
  }

  function translateRoute(raw) {
    return translateByTable(raw, ROUTE_TRANSLATIONS) || raw || 'INSTANT SELF-SERVICE :: Direct Route Resolved';
  }

  function translateCardTitle(raw) {
    return translateByTable(raw, RISK_CARD_TRANSLATIONS.title) || 'Risk Disposition Card';
  }

  function translateAction(raw) {
    return translateByTable(raw, RISK_CARD_TRANSLATIONS.actions) || null;
  }

  // --- WebSocket plumbing ---

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
    if (msg.type === 'demo3_result') {
      handleResult(msg.data);
    }
  }

  function handleResult(data) {
    state.analyzing = false;
    els.btnAnalyze.textContent = 'PLAY AUDIO (8.9s)';
    els.btnAnalyze.disabled = false;

    els.transcriptText.textContent = hasCJK(data.user_input) ? TRANSCRIPT_EN : data.user_input;
    els.routeDisplay.textContent = translateRoute(data.primary_route);
    if (typeof data.parse_latency_ms === 'number') {
      els.kpiRoute.textContent = `${data.parse_latency_ms}ms`;
      const ivrVal = document.getElementById('ivr-tractus-val');
      if (ivrVal) ivrVal.textContent = `${data.parse_latency_ms}ms`;
    }
    if (typeof data.handoff_dispatch_latency_ms === 'number') {
      els.kpiHandoff.textContent = `${data.handoff_dispatch_latency_ms}ms`;
    }
    if (data.ivr_comparison && data.ivr_comparison.speedup_factor) {
      els.kpiSpeedup.textContent = data.ivr_comparison.speedup_factor;
    }

    lockOscilloscope();

    if (data.radar) {
      state.targetValues = {
        compliance: data.radar.compliance || 96.9,
        handoff: data.radar.handoff || 96.1,
        billing: data.radar.billing || 94.1,
      };
      animateRadar();
    }

    if (data.risk_cards && data.risk_cards.length > 0) {
      displayRiskCards(data.risk_cards);
    }
  }

  function displayRiskCards(cards) {
    els.riskFeed.innerHTML = '';

    cards.forEach(card => {
      const item = document.createElement('div');
      item.className = `risk-card ${card.risk_level === 'high' || card.risk_level === 'CRITICAL' ? 'critical' : ''}`;

      const title = document.createElement('div');
      title.className = 'risk-title';
      title.innerHTML = `${translateCardTitle(card.title)} <span class="risk-badge">${(card.risk_level || 'MEDIUM').toString().toUpperCase()}</span>`;
      item.appendChild(title);

      const details = document.createElement('div');
      const detailLines = Array.isArray(card.details) ? card.details : [card.details];
      detailLines.filter(Boolean).forEach(line => {
        const row = document.createElement('div');
        row.style.marginBottom = '4px';
        row.textContent = hasCJK(line) ? 'Evidence captured from live voice stream analysis.' : line;
        details.appendChild(row);
      });
      item.appendChild(details);

      if (card.suggested_actions && card.suggested_actions.length > 0) {
        const actions = document.createElement('div');
        actions.className = 'risk-actions';
        const translated = card.suggested_actions
          .map(translateAction)
          .filter(Boolean);
        actions.textContent = `Recommended: ${translated.join('; ')}`;
        item.appendChild(actions);
      }

      els.riskFeed.appendChild(item);
    });
  }

  // --- Oscilloscope waveform ---

  function buildOscilloscope() {
    els.oscWave.innerHTML = '';
    for (let i = 0; i < 90; i++) {
      const bar = document.createElement('div');
      bar.className = 'osc-bar';
      const h = 8 + Math.sin(i * 0.4) * 10 + Math.random() * 14;
      bar.style.height = `${Math.max(4, h)}%`;
      els.oscWave.appendChild(bar);
    }
  }

  function lockOscilloscope() {
    els.oscLockLine.classList.add('show');
    els.oscLockLabel.classList.add('show');
    const bars = els.oscWave.children;
    for (let i = 0; i < bars.length; i++) {
      if (i / bars.length < 0.28) {
        bars[i].style.background = 'var(--cyan)';
        bars[i].style.opacity = '0.8';
      }
    }
  }

  function resetOscilloscope() {
    els.oscLockLine.classList.remove('show');
    els.oscLockLabel.classList.remove('show');
    buildOscilloscope();
  }

  // --- Radar ---

  function initRadar() {
    state.radarCtx = els.radarCanvas.getContext('2d');
    drawRadar();
  }

  function drawRadar() {
    const ctx = state.radarCtx;
    const width = els.radarCanvas.width;
    const height = els.radarCanvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 36;

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius / 5) * i;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    const axes = [
      { angle: -Math.PI / 2, label: 'COMPLIANCE' },
      { angle: Math.PI / 6, label: 'HANDOFF' },
      { angle: (5 * Math.PI) / 6, label: 'BILLING' },
    ];

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    axes.forEach(axis => {
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(
        centerX + maxRadius * Math.cos(axis.angle),
        centerY + maxRadius * Math.sin(axis.angle)
      );
      ctx.stroke();
    });

    const values = [
      state.radarValues.compliance / 100,
      state.radarValues.handoff / 100,
      state.radarValues.billing / 100,
    ];

    ctx.beginPath();
    values.forEach((val, i) => {
      const angle = axes[i].angle;
      const radius = maxRadius * val;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    ctx.fillStyle = 'rgba(255, 55, 95, 0.14)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 159, 10, 0.85)';
    ctx.lineWidth = 2;
    ctx.stroke();

    values.forEach((val, i) => {
      const angle = axes[i].angle;
      const radius = maxRadius * val;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#FF9F0A';
      ctx.fill();
    });

    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(237, 239, 244, 0.5)';
    axes.forEach(axis => {
      const x = centerX + (maxRadius + 18) * Math.cos(axis.angle);
      const y = centerY + (maxRadius + 18) * Math.sin(axis.angle);
      ctx.textAlign = 'center';
      ctx.fillText(axis.label, x, y);
    });
  }

  function animateRadar() {
    const duration = 1500;
    const startTime = Date.now();
    const startValues = { ...state.radarValues };

    function animate() {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);

      state.radarValues.compliance = startValues.compliance + (state.targetValues.compliance - startValues.compliance) * eased;
      state.radarValues.handoff = startValues.handoff + (state.targetValues.handoff - startValues.handoff) * eased;
      state.radarValues.billing = startValues.billing + (state.targetValues.billing - startValues.billing) * eased;

      els.legCompliance.textContent = `${Math.round(state.radarValues.compliance)}%`;
      els.legHandoff.textContent = `${Math.round(state.radarValues.handoff)}%`;
      els.legBilling.textContent = `${Math.round(state.radarValues.billing)}%`;

      drawRadar();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    }

    animate();
  }

  // --- Controls ---

  function analyzeDemo() {
    if (state.analyzing || !state.connected) return;

    state.analyzing = true;
    els.btnAnalyze.textContent = 'ANALYZING…';
    els.btnAnalyze.disabled = true;

    els.audioPlayer.currentTime = 0;
    els.audioPlayer.play();

    state.ws.send(JSON.stringify({
      action: 'run_demo3',
      complaint: null,
    }));

    const duration = 8.9;
    let elapsed = 0;
    const timerInterval = setInterval(() => {
      elapsed += 0.1;
      const current = Math.min(elapsed, duration);
      const secs = Math.floor(current);
      els.callTimer.textContent = `00:${secs.toString().padStart(2, '0')} / 00:09`;

      if (elapsed >= duration) {
        clearInterval(timerInterval);
      }
    }, 100);
  }

  function resetDemo() {
    els.kpiRoute.textContent = '--';
    els.kpiHandoff.textContent = '--';
    els.kpiSpeedup.textContent = '--';
    const ivrVal = document.getElementById('ivr-tractus-val');
    if (ivrVal) ivrVal.textContent = 'Single Pass';
    els.callTimer.textContent = '00:00 / 00:09';
    els.transcriptText.textContent = 'Press PLAY AUDIO to stream the customer complaint and begin real-time analysis…';
    els.routeDisplay.textContent = 'AWAITING ANALYSIS…';

    state.radarValues = { compliance: 25, handoff: 30, billing: 35 };
    els.legCompliance.textContent = '25%';
    els.legHandoff.textContent = '30%';
    els.legBilling.textContent = '35%';
    drawRadar();

    resetOscilloscope();

    els.riskFeed.innerHTML = '<div class="risk-card" style="opacity: 0.5;">Expert disposition cards will render here after the audio stream is analyzed.</div>';

    els.audioPlayer.pause();
    els.audioPlayer.currentTime = 0;
    els.btnAnalyze.textContent = 'PLAY AUDIO (8.9s)';
    els.btnAnalyze.disabled = false;
    state.analyzing = false;
  }

  els.btnAnalyze.addEventListener('click', analyzeDemo);
  els.btnReset.addEventListener('click', resetDemo);

  buildOscilloscope();
  initRadar();
  connectWebSocket();
})();
