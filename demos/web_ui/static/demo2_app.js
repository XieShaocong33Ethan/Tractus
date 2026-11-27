/**
 * TRACTUS Demo 2: Duplex Meeting Sentinel - Standalone Client
 */
(function() {
  'use strict';

  const LED_SEGMENTS = 20;

  const state = {
    ws: null,
    connected: false,
    playing: false,
    chunks: 0,
    totalChunks: 12,
  };

  const els = {
    kpiReject: document.getElementById('kpi-reject'),
    kpiLatency: document.getElementById('kpi-latency'),
    kpiTokens: document.getElementById('kpi-tokens'),
    timer: document.getElementById('timer'),
    mReject: document.getElementById('m-reject'),
    mLatency: document.getElementById('m-latency'),
    mTokens: document.getElementById('m-tokens'),
    mChunks: document.getElementById('m-chunks'),
    mCost: document.getElementById('m-cost'),
    ledMeter: document.getElementById('led-meter'),
    gateBarrier: document.getElementById('gate-barrier'),
    gateReadout: document.getElementById('gate-readout'),
    speechFeed: document.getElementById('speech-feed'),
    taskFeed: document.getElementById('task-feed'),
    meetingFrame: document.getElementById('meeting-frame'),
    audioPlayer: document.getElementById('audio-player'),
    btnPlay: document.getElementById('btn-play'),
    btnReset: document.getElementById('btn-reset'),
  };

  const participantNodes = {
    host: document.getElementById('node-sj'),
    'part-a': document.getElementById('node-dc'),
    'part-b': document.getElementById('node-er'),
    'part-c': document.getElementById('node-mv'),
  };

  function buildLedMeter() {
    els.ledMeter.innerHTML = '';
    for (let i = 0; i < LED_SEGMENTS; i++) {
      const seg = document.createElement('div');
      seg.className = 'tx-led-seg';
      els.ledMeter.appendChild(seg);
    }
  }

  function setLedMeter(fraction) {
    const segs = els.ledMeter.children;
    const lit = Math.round(fraction * segs.length);
    for (let i = 0; i < segs.length; i++) {
      segs[i].classList.toggle('on', i < lit);
      segs[i].classList.toggle('amber', i >= segs.length * 0.7 && i < lit);
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
      case 'demo2_init':
        handleInit(msg.data);
        break;
      case 'demo2_chunk':
        handleChunk(msg.data);
        break;
      case 'demo2_complete':
        handleComplete(msg.data);
        break;
    }
  }

  function handleInit(data) {
    state.chunks = 0;
    state.totalChunks = data.total_chunks || 12;
    els.speechFeed.innerHTML = '';
  }

  function handleChunk(data) {
    state.chunks = data.chunk_idx;

    const rejectRate = data.rejection_rate || 0;
    const tokensSaved = data.total_tokens_saved || 0;
    const latency = data.neural_latency_ms || 0;

    els.kpiReject.textContent = `${rejectRate}%`;
    els.kpiLatency.textContent = `${latency}ms`;
    els.kpiTokens.textContent = tokensSaved.toLocaleString();

    els.mReject.textContent = `${rejectRate}%`;
    els.mLatency.textContent = `${latency}ms`;
    els.mTokens.textContent = tokensSaved.toLocaleString();
    els.mChunks.textContent = `${data.chunk_idx}/${data.total_chunks}`;
    els.mCost.textContent = `$${(tokensSaved * 0.00003).toFixed(2)}`;
    setLedMeter(Math.min(tokensSaved / 20000, 1));

    highlightSpeaker(data.speaker_key);
    updateGate(data);
    addSpeechItem(data);
    updateMeetingFrame(data);

    if (data.action_item) {
      addTaskItem(data.action_item);
    }
  }

  function updateMeetingFrame(data) {
    if (!els.meetingFrame) return;
    try {
      const win = els.meetingFrame.contentWindow;
      if (!win) return;

      const spkKey = data.speaker_key === 'host' ? 'HOST' :
                     data.speaker_key === 'part-a' ? 'PARTICIPANT_1' :
                     data.speaker_key === 'part-b' ? 'PARTICIPANT_2' : 'PARTICIPANT_3';

      if (typeof win.setActiveSpeaker === 'function') {
        win.setActiveSpeaker(spkKey);
      }

      if (typeof win.addStreamChunk === 'function') {
        const isSilent = data.is_filler;
        const cat = data.decision_action || 'DISPATCH_TASK';
        const lat = data.neural_latency_ms || 0;
        const titleEn = translateTaskTitle(data.action_item ? data.action_item.title : '');
        const taskPayload = data.action_item ? {
          action: (data.decision_action === 'SUPPLEMENT_TASK' ? 'SUPPLEMENT' : 'DISPATCH'),
          task_id: data.action_item.id || 'TASK-101',
          title: titleEn,
          creator: data.speaker,
          tokens_saved: data.total_tokens_saved,
        } : null;

        win.addStreamChunk(spkKey, data.text, isSilent, cat, lat, '', taskPayload);
      }
    } catch (e) {
      // Cross-origin safe guard
    }
  }

  function updateGate(data) {
    els.gateBarrier.classList.remove('breach', 'absorb');
    if (data.is_filler) {
      els.gateBarrier.classList.add('absorb');
      els.gateReadout.className = 'gate-readout filtered';
      els.gateReadout.textContent = `FILTERED AT EDGE :: SILENT_OBSERVE · 0 TOKENS TO CLOUD`;
    } else if (data.action_item && data.action_item.id && els.taskFeed.querySelector(`[data-task-id="${data.action_item.id}"]`)) {
      els.gateBarrier.classList.add('breach');
      els.gateReadout.className = 'gate-readout supplement';
      els.gateReadout.textContent = `SUPPLEMENT_TASK :: CONTEXT MUTATED — #${data.action_item.id}`;
    } else if (data.action_item) {
      els.gateBarrier.classList.add('breach');
      els.gateReadout.className = 'gate-readout dispatch';
      els.gateReadout.textContent = `DISPATCH_TASK -> LLM BOT :: #${data.action_item.id || 'TASK'}`;
    } else {
      els.gateReadout.className = 'gate-readout';
      els.gateReadout.textContent = `MONITORING :: ${data.decision_action || 'PASSIVE_LISTEN'}`;
    }
  }

  function highlightSpeaker(speakerKey) {
    Object.values(participantNodes).forEach(node => {
      node.classList.remove('speaking');
    });
    if (participantNodes[speakerKey]) {
      participantNodes[speakerKey].classList.add('speaking');
    }
  }

  function addSpeechItem(data) {
    const item = document.createElement('div');
    item.className = `tx-feed-row ${data.is_filler ? 'dim' : 'hit'}`;

    const head = document.createElement('div');
    head.className = 'tx-feed-row-head';
    head.textContent = data.speaker || 'Unknown';
    item.appendChild(head);

    item.appendChild(document.createTextNode(data.text));

    const badge = document.createElement('div');
    badge.className = 'tx-feed-row-meta';
    badge.style.color = data.is_filler ? 'var(--ink-text-faint)' : 'var(--amber)';
    badge.textContent = data.is_filler ? 'SILENT_OBSERVE' : (data.decision_action || 'DISPATCH_TASK');
    item.appendChild(badge);

    const latency = document.createElement('div');
    latency.className = 'tx-feed-row-meta';
    latency.textContent = `${data.neural_latency_ms}ms · Confidence: ${data.confidence}%`;
    item.appendChild(latency);

    els.speechFeed.appendChild(item);
    els.speechFeed.scrollTop = els.speechFeed.scrollHeight;
  }

  function translateTaskTitle(title) {
    if (!title) return 'Actionable meeting task item';
    return title;
  }

  function addTaskItem(task) {
    if (els.taskFeed.querySelector('.task-card[style*="opacity"]')) {
      els.taskFeed.innerHTML = '';
    }

    const titleEn = translateTaskTitle(task.title);
    const taskId = task.id || (task.title ? task.title.split(' ')[0] : 'TASK');
    const existing = els.taskFeed.querySelector(`[data-task-id="${taskId}"]`);
    if (existing) {
      existing.className = 'task-card supplement';
      existing.innerHTML = `
        <div class="task-card-top">
          <span style="color: var(--mint);">#${taskId} · SUPPLEMENT_TASK</span>
          <span style="color: var(--mint);">CONTEXT REFINED</span>
        </div>
        <div class="task-card-title">${titleEn}</div>
        <div class="task-card-meta">Assignee: ${task.assignee} · Status: ${task.status} · Priority: ${task.priority || 'HIGH'}</div>
      `;
      return;
    }

    const item = document.createElement('div');
    item.className = 'task-card dispatch';
    item.setAttribute('data-task-id', taskId);
    item.innerHTML = `
      <div class="task-card-top">
        <span style="color: var(--cyan);">#${taskId} · DISPATCH_TASK</span>
        <span style="color: var(--cyan);">${task.status}</span>
      </div>
      <div class="task-card-title">${titleEn}</div>
      <div class="task-card-meta">Assignee: ${task.assignee} · Priority: ${task.priority || 'MEDIUM'}</div>
    `;
    els.taskFeed.appendChild(item);
  }

  function playDemo() {
    if (state.playing || !state.connected) return;

    state.playing = true;
    els.btnPlay.textContent = 'PLAYING…';
    els.btnPlay.disabled = true;
    els.speechFeed.innerHTML = '';
    els.taskFeed.innerHTML = '<div class="task-card" style="opacity: 0.5;">Monitoring stream…</div>';

    els.audioPlayer.currentTime = 0;
    els.audioPlayer.play();

    state.ws.send(JSON.stringify({
      action: 'run_demo2',
      speed: 1.0,
    }));

    const duration = 63.6;
    let elapsed = 0;
    const timerInterval = setInterval(() => {
      elapsed += 0.1;
      const current = Math.min(elapsed, duration);
      const mins = Math.floor(current / 60);
      const secs = Math.floor(current % 60);
      els.timer.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')} / 01:03`;

      if (elapsed >= duration) {
        clearInterval(timerInterval);
      }
    }, 100);
  }

  function handleComplete(data) {
    state.playing = false;
    els.btnPlay.textContent = 'PLAY AUDIO (63.6s)';
    els.btnPlay.disabled = false;

    els.kpiReject.textContent = `${data.rejection_rate}%`;
    els.kpiTokens.textContent = data.total_tokens_saved.toLocaleString();
    els.gateReadout.className = 'gate-readout';
    els.gateReadout.textContent = 'SESSION COMPLETE';
  }

  function resetDemo() {
    state.chunks = 0;
    els.kpiReject.textContent = '3/12 (25%)';
    els.kpiLatency.textContent = '--';
    els.kpiTokens.textContent = '0';
    els.mReject.textContent = '0%';
    els.mLatency.textContent = '--';
    els.mTokens.textContent = '0';
    els.mChunks.textContent = '0/12';
    els.mCost.textContent = '$0.00';
    setLedMeter(0);
    els.gateBarrier.classList.remove('breach', 'absorb');
    els.gateReadout.className = 'gate-readout';
    els.gateReadout.textContent = 'AWAITING AUDIO STREAM';
    els.timer.textContent = '00:00 / 01:03';
    els.speechFeed.innerHTML = '<div class="tx-feed-row dim"><div class="tx-feed-row-head">Standby</div>Press PLAY AUDIO to begin real-time duplex processing.</div>';
    els.taskFeed.innerHTML = '<div class="task-card" style="opacity: 0.5;">Structured task cards appear here when an explicit directive is detected.</div>';
    els.audioPlayer.pause();
    els.audioPlayer.currentTime = 0;
    els.btnPlay.textContent = 'PLAY AUDIO (63.6s)';
    els.btnPlay.disabled = false;
    state.playing = false;

    Object.values(participantNodes).forEach(node => {
      node.classList.remove('speaking');
    });

    try {
      if (els.meetingFrame && els.meetingFrame.contentWindow && typeof els.meetingFrame.contentWindow.resetStage === 'function') {
        els.meetingFrame.contentWindow.resetStage();
      }
    } catch (e) {}
  }

  buildLedMeter();

  els.btnPlay.addEventListener('click', playDemo);
  els.btnReset.addEventListener('click', resetDemo);

  connectWebSocket();
})();
