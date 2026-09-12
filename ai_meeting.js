/**
 * AirScribe AI Meeting Module
 * ============================
 * Frontend integration for AI-powered multi-speaker transcription.
 * 
 * This module works ALONGSIDE the existing MeetingRecorder.
 * It uses MediaRecorder API to capture audio chunks and sends them
 * to the AI backend (port 5001) for Whisper transcription + diarization.
 * 
 * IMPORTANT: Does NOT modify or replace any existing functionality.
 * Manual speaker naming is fully preserved.
 */

const AIMeeting = (() => {
    // ── Configuration ──
    const AI_API = 'http://localhost:5001';
    const CHUNK_INTERVAL_MS = 8000; // Send audio every 8 seconds
    const SPEAKER_COLORS = ['#a78bfa', '#34d399', '#4fd1ff', '#fbbf24', '#fb923c', '#f87171'];

    // ── State ──
    let isActive = false;
    let sessionId = null;
    let mediaRecorder = null;
    let audioStream = null;
    let chunkIndex = 0;
    let chunkTimer = null;
    let audioChunks = [];
    let speakerNames = {};    // index → name (synced with backend)
    let manualNames = {};     // index → name (user-set, never overwritten)
    let lastSpeakerIndex = -1;
    let aiAvailable = false;
    let pendingChunks = 0;
    let enrolledProfileNames = []; // Names of enrolled voice profiles

    // ── Feature #3: Timeline Fusion State ──
    let timelineEnabled = false;
    let timelinePollTimer = null;
    const TIMELINE_POLL_MS = 5000; // Poll timeline every 5 seconds
    let lastTimelineEventCount = 0;

    // ── AI Server Health Check ──
    async function checkAIServer() {
        try {
            const ctrl = new AbortController();
            const t = setTimeout(() => ctrl.abort(), 3000);
            const res = await fetch(`${AI_API}/ai/health`, { signal: ctrl.signal });
            clearTimeout(t);
            if (res.ok) {
                const data = await res.json();
                aiAvailable = data.status === 'ok';
                console.log('AI Meeting server:', aiAvailable ? 'CONNECTED' : 'ERROR', data);
                return aiAvailable;
            }
        } catch (e) {
            aiAvailable = false;
            console.warn('AI Meeting server not available');
        }
        return false;
    }

    // ── Start AI Meeting Session ──
    async function start(names = ['Speaker 1', 'Speaker 2']) {
        if (isActive) { console.warn('AI Meeting already active'); return false; }

        // Check AI server
        const serverOk = await checkAIServer();
        if (!serverOk) {
            showAIToast('AI server not running. Start it with: python ai_meeting_server.py', 'error');
            return false;
        }

        try {
            // Request microphone
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
            });
        } catch (e) {
            showAIToast('Microphone access denied. Please allow mic permission.', 'error');
            return false;
        }

        // Create backend session
        try {
            const res = await fetch(`${AI_API}/ai/session/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    speakerCount: names.length,
                    speakerNames: names,
                    useVoiceRecognition: true
                })
            });
            const data = await res.json();
            sessionId = data.sessionId;

            // Initialize speaker names
            manualNames = {};
            speakerNames = {};
            names.forEach((n, i) => {
                speakerNames[i] = n;
                if (n && !n.startsWith('Speaker ')) manualNames[i] = n;
            });
        } catch (e) {
            showAIToast('Failed to create AI session: ' + e.message, 'error');
            if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
            return false;
        }

        // Fetch enrolled voice profiles (for speaker recognition)
        await _fetchEnrolledProfiles();

        // Setup MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';

        mediaRecorder = new MediaRecorder(audioStream, { mimeType });
        audioChunks = [];
        chunkIndex = 0;
        pendingChunks = 0;
        lastSpeakerIndex = -1;
        isActive = true;

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.start(1000); // Collect data every second

        // Send chunks periodically
        chunkTimer = setInterval(() => sendCurrentChunk(), CHUNK_INTERVAL_MS);

        showAIPanel(true);
        updateAIStatus('🎙️ AI Listening...', 'listening');
        showAIToast('AI Meeting transcription started', 'info');
        console.log(`AI Meeting started — session: ${sessionId}`);

        // Feature #3: Start timeline polling
        startTimelinePolling();

        return true;
    }

    // ── Send Audio Chunk to Backend ──
    async function sendCurrentChunk() {
        if (!isActive || audioChunks.length === 0) return;

        const blob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
        audioChunks = []; // Clear buffer

        if (blob.size < 1000) return; // Skip tiny chunks (silence)

        const currentChunk = chunkIndex++;
        pendingChunks++;
        updateAIStatus('⏳ Processing...', 'processing');

        const formData = new FormData();
        formData.append('audio', blob, `chunk_${currentChunk}.webm`);
        formData.append('sessionId', sessionId);
        formData.append('chunkIndex', currentChunk);

        try {
            const res = await fetch(`${AI_API}/ai/transcribe`, { method: 'POST', body: formData });
            const data = await res.json();

            if (data.segments && data.segments.length > 0) {
                insertAISegments(data.segments);

                // Update speaker info from backend
                if (data.session && data.session.speakers) {
                    Object.values(data.session.speakers).forEach(s => {
                        if (!manualNames[s.index]) {
                            speakerNames[s.index] = s.label;
                        }
                    });
                    updateSpeakerPanel();
                }
            }

            pendingChunks--;
            if (isActive) updateAIStatus('🎙️ AI Listening...', 'listening');
        } catch (e) {
            pendingChunks--;
            console.error('AI transcription error:', e);
            if (isActive) updateAIStatus('⚠️ Error — retrying...', 'error');
        }
    }

    // ── Insert AI-Transcribed Segments into Quill Editor ──
    function insertAISegments(segments) {
        if (!window.quill || !segments.length) return;

        segments.forEach(seg => {
            const speakerIdx = seg.speakerIndex;
            const name = manualNames[speakerIdx] || seg.speaker;
            const color = SPEAKER_COLORS[speakerIdx % SPEAKER_COLORS.length];
            let text = seg.text.trim();
            if (!text) return;

            // Capitalize first letter
            text = text.charAt(0).toUpperCase() + text.slice(1);
            if (!/[.!?]$/.test(text)) text += '.';

            // Determine confidence indicator
            const confidence = seg.confidence || 0;
            let confidenceMark = '';
            if (enrolledProfileNames.length > 0 && confidence >= 0.70) {
                confidenceMark = ' ✓';  // Recognized speaker
            } else if (enrolledProfileNames.length > 0 && confidence > 0 && confidence < 0.70) {
                confidenceMark = ' ?';  // Uncertain match
            }

            const len = window.quill.getLength();

            // Insert speaker label if speaker changed
            if (speakerIdx !== lastSpeakerIndex) {
                const labelText = name + confidenceMark + ': ';
                window.quill.insertText(len - 1, '\n' + labelText, { bold: true, italic: true, color });
                lastSpeakerIndex = speakerIdx;
            }

            // Insert transcribed text
            const newLen = window.quill.getLength();
            window.quill.insertText(newLen - 1, text + ' ');

            // Scroll to bottom
            const editor = document.querySelector('.ql-editor');
            if (editor) editor.scrollTop = editor.scrollHeight;
        });

        // Update stats
        if (typeof updateEditorStats === 'function') updateEditorStats();
        if (typeof setSaveStatus === 'function') setSaveStatus('Unsaved...');
    }

    // ══════════════════════════════════════════════════
    //  FEATURE #3: TIMELINE FUSION API
    // ══════════════════════════════════════════════════

    /**
     * Add a typed note to the meeting timeline.
     * @param {string} content - The note text.
     * @param {number} [timestamp] - Epoch ms. Defaults to now.
     */
    async function addTimelineNote(content, timestamp) {
        if (!sessionId || !timelineEnabled) return null;
        try {
            const res = await fetch(`${AI_API}/ai/session/${sessionId}/timeline/note`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content,
                    timestamp: timestamp || Date.now()
                })
            });
            if (res.ok) {
                const data = await res.json();
                console.log('[Timeline] Note added:', data.event?.id);
                return data;
            }
        } catch (e) {
            console.debug('[Timeline] Note add failed:', e.message);
        }
        return null;
    }

    /**
     * Fetch the current unified timeline.
     * @param {string} [type] - Filter by event type.
     * @param {number} [since] - Only events after this epoch ms.
     * @returns {object|null} Timeline data.
     */
    async function getTimeline(type, since) {
        if (!sessionId) return null;
        try {
            let url = `${AI_API}/ai/session/${sessionId}/timeline`;
            const params = new URLSearchParams();
            if (type) params.set('type', type);
            if (since) params.set('since', String(since));
            if (params.toString()) url += '?' + params.toString();

            const res = await fetch(url);
            if (res.ok) return await res.json();
        } catch (e) {
            console.debug('[Timeline] Fetch failed:', e.message);
        }
        return null;
    }

    /**
     * Add a screen capture + optional OCR to the timeline.
     */
    async function addScreenCapture(captureTimestamp, ocrText, ocrCompletionTimestamp) {
        if (!sessionId || !timelineEnabled) return null;
        try {
            const body = { captureTimestamp };
            if (ocrText) body.ocrText = ocrText;
            if (ocrCompletionTimestamp) body.ocrCompletionTimestamp = ocrCompletionTimestamp;

            const res = await fetch(`${AI_API}/ai/session/${sessionId}/timeline/screen`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (res.ok) {
                const data = await res.json();
                console.log('[Timeline] Screen capture added:', data.screenEvent?.id);
                return data;
            }
        } catch (e) {
            console.debug('[Timeline] Screen capture add failed:', e.message);
        }
        return null;
    }

    /**
     * Start polling the timeline for updates.
     */
    function startTimelinePolling() {
        if (timelinePollTimer) return;
        timelineEnabled = true;
        lastTimelineEventCount = 0;

        timelinePollTimer = setInterval(async () => {
            const data = await getTimeline();
            if (data && data.counts) {
                const total = data.counts.total || 0;
                if (total > lastTimelineEventCount) {
                    lastTimelineEventCount = total;
                    updateTimelineIndicator(data.counts);
                }
            }
        }, TIMELINE_POLL_MS);
    }

    function stopTimelinePolling() {
        if (timelinePollTimer) {
            clearInterval(timelinePollTimer);
            timelinePollTimer = null;
        }
        timelineEnabled = false;
    }

    function updateTimelineIndicator(counts) {
        const el = document.getElementById('aiTimelineCounts');
        if (!el) return;
        const parts = [];
        if (counts.transcript) parts.push(`🎙️ ${counts.transcript}`);
        if (counts.note) parts.push(`📝 ${counts.note}`);
        if (counts.screen) parts.push(`📸 ${counts.screen}`);
        if (counts.ocr) parts.push(`🔍 ${counts.ocr}`);
        el.textContent = parts.join(' · ') || 'No events';
        el.title = `Timeline: ${counts.total || 0} total events`;
    }

    // ── Stop AI Meeting ──
    async function stop() {
        if (!isActive) return;
        isActive = false;

        clearInterval(chunkTimer);
        chunkTimer = null;
        stopTimelinePolling();

        // Send remaining audio
        if (audioChunks.length > 0) await sendCurrentChunk();

        // Stop MediaRecorder
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
        }
        mediaRecorder = null;

        // Stop mic
        if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }

        // Notify backend
        if (sessionId) {
            try {
                await fetch(`${AI_API}/ai/session/${sessionId}/stop`, { method: 'POST' });
            } catch (e) { /* ignore */ }
        }

        updateAIStatus('Stopped', 'stopped');
        setTimeout(() => showAIPanel(false), 3000);
        showAIToast('AI transcription stopped', 'success');
    }

    // ── Rename Speaker (preserves manual naming) ──
    async function renameSpeaker(index, name) {
        if (name && name.trim()) {
            manualNames[index] = name.trim();
            speakerNames[index] = name.trim();
        } else {
            delete manualNames[index];
            speakerNames[index] = `Speaker ${index + 1}`;
        }

        // Sync with backend
        if (sessionId) {
            try {
                await fetch(`${AI_API}/ai/session/${sessionId}/rename-speaker`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ speakerIndex: index, name: name?.trim() || '' })
                });
            } catch (e) { /* ignore */ }
        }

        updateSpeakerPanel();
    }

    // ══════════════════════════════════════════════════
    //  UI HELPERS
    // ══════════════════════════════════════════════════

    function showAIPanel(visible) {
        const panel = document.getElementById('aiMeetingPanel');
        if (panel) {
            if (visible) panel.classList.add('visible');
            else panel.classList.remove('visible');
        }
    }

    function updateAIStatus(text, state) {
        const el = document.getElementById('aiMeetingStatus');
        if (el) {
            el.textContent = text;
            el.className = 'ai-meeting-status ' + (state || '');
        }
    }

    function updateSpeakerPanel() {
        const container = document.getElementById('aiSpeakerList');
        if (!container) return;

        container.innerHTML = '';
        const maxIdx = Math.max(...Object.keys(speakerNames).map(Number), 0);

        for (let i = 0; i <= maxIdx; i++) {
            const name = manualNames[i] || speakerNames[i] || `Speaker ${i + 1}`;
            const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
            const isManual = i in manualNames;

            const row = document.createElement('div');
            row.className = 'ai-speaker-row';
            row.innerHTML = `
                <span class="ai-speaker-dot" style="background:${color}"></span>
                <input class="ai-speaker-input" value="${name}" data-idx="${i}"
                       placeholder="Speaker ${i + 1}" ${isManual ? 'style="color:var(--text-1);font-weight:600"' : ''}>
                <span class="ai-speaker-badge">${isManual ? '✎ Manual' : '🤖 AI'}</span>
            `;
            row.querySelector('input').addEventListener('change', (e) => {
                renameSpeaker(parseInt(e.target.dataset.idx), e.target.value);
            });
            container.appendChild(row);
        }
    }

    function showAIToast(msg, type) {
        if (typeof showToast === 'function') { showToast(msg, type); return; }
        console.log(`[AI Meeting] ${type}: ${msg}`);
    }

    // ── Fetch enrolled voice profiles ──
    async function _fetchEnrolledProfiles() {
        try {
            const res = await fetch(`${AI_API}/ai/voice/profiles`);
            if (res.ok) {
                const profiles = await res.json();
                enrolledProfileNames = profiles.map(p => p.userName);
                console.log(`[AI Meeting] ${enrolledProfileNames.length} enrolled voice profiles loaded`);
            }
        } catch (e) {
            enrolledProfileNames = [];
            console.debug('[AI Meeting] Could not fetch voice profiles (optional)');
        }
    }

    // ── Inject the AI Meeting UI Panel ──
    function injectUI() {
        if (document.getElementById('aiMeetingPanel')) return;

        const panel = document.createElement('div');
        panel.id = 'aiMeetingPanel';
        panel.className = 'ai-meeting-panel';
        panel.innerHTML = `
            <div class="ai-meeting-header">
                <div class="ai-meeting-title">
                    <i class="fa fa-brain"></i> AI Transcription
                </div>
                <div id="aiMeetingStatus" class="ai-meeting-status">Ready</div>
            </div>
            <div id="aiSpeakerList" class="ai-speaker-list"></div>
            <div id="aiTimelineCounts" class="ai-timeline-counts" style="font-size:11px;color:var(--text-2, #888);padding:4px 8px;text-align:center;"></div>
            <div class="ai-meeting-actions">
                <button class="ai-btn-stop" id="aiStopBtn" onclick="AIMeeting.stop()">
                    <i class="fa fa-stop"></i> Stop AI
                </button>
            </div>
        `;
        document.body.appendChild(panel);
    }

    // ── Inject CSS ──
    function injectStyles() {
        if (document.getElementById('aiMeetingStyles')) return;
        const link = document.createElement('link');
        link.id = 'aiMeetingStyles';
        link.rel = 'stylesheet';
        link.href = 'AirScribe_AI_Meeting/frontend/ai_meeting.css';
        document.head.appendChild(link);
    }

    // ── Initialize (call once on page load) ──
    function init() {
        injectStyles();
        injectUI();
        checkAIServer().then(ok => {
            if (ok) console.log('✅ AI Meeting module ready');
            else console.log('ℹ️ AI Meeting server not running — AI features disabled');
        });
    }

    // ── Public API ──
    return {
        init,
        start,
        stop,
        renameSpeaker,
        isActive: () => isActive,
        checkServer: checkAIServer,
        getSessionId: () => sessionId,
        getSpeakers: () => ({ ...speakerNames }),
        getManualNames: () => ({ ...manualNames }),
        // Feature #3: Timeline Fusion API
        addTimelineNote,
        getTimeline,
        addScreenCapture,
        isTimelineEnabled: () => timelineEnabled,
    };
})();

// Auto-initialize when script loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => AIMeeting.init());
} else {
    AIMeeting.init();
}
