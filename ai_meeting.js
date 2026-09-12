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
    let timelineReadyPromise = null;

    function ensureTimelineFusion() {
        if (window.TimelineFusion) return Promise.resolve(window.TimelineFusion);
        if (timelineReadyPromise) return timelineReadyPromise;

        timelineReadyPromise = new Promise((resolve, reject) => {
            const existing = document.querySelector('script[data-timeline-fusion]');
            if (existing) {
                existing.addEventListener('load', () => resolve(window.TimelineFusion));
                existing.addEventListener('error', reject);
                return;
            }
            const script = document.createElement('script');
            script.src = 'timeline_fusion.js';
            script.dataset.timelineFusion = 'true';
            script.onload = () => resolve(window.TimelineFusion);
            script.onerror = () => reject(new Error('Timeline Fusion module could not be loaded'));
            document.head.appendChild(script);
        });
        return timelineReadyPromise;
    }

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

        try { await ensureTimelineFusion(); } catch (e) { console.warn('[TimelineFusion] Optional module unavailable:', e); }

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

        // Reset one shared timeline for this meeting.
        if (window.TimelineFusion) {
            window.TimelineFusion.init(sessionId);
            if (window.quill) window.TimelineFusion.attachQuill(window.quill);
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
        return true;
    }

    // ── Send Audio Chunk to Backend ──
    async function sendCurrentChunk() {
        if (!isActive || audioChunks.length === 0) return;

        const blob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
        audioChunks = []; // Clear buffer

        if (blob.size < 1000) return; // Skip tiny chunks (silence)

        const currentChunk = chunkIndex++;
        const chunkStartSeconds = currentChunk * (CHUNK_INTERVAL_MS / 1000);
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
                // Whisper timestamps are relative to each 8-second chunk. Feed
                // the chunk origin into the unified meeting timeline so events
                // do not reset to 0s on every transcription request.
                if (window.TimelineFusion) {
                    data.segments.forEach(seg => window.TimelineFusion.addTranscript(seg, chunkStartSeconds));
                }

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

    // ── Stop AI Meeting ──
    async function stop() {
        if (!isActive) return;
        isActive = false;

        clearInterval(chunkTimer);
        chunkTimer = null;

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

        if (window.TimelineFusion) window.TimelineFusion.show(true);
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
    function showTimelineButton() {
        if (document.getElementById('timelineFusionOpenBtn')) return;
        const btn = document.createElement('button');
        btn.id = 'timelineFusionOpenBtn';
        btn.className = 'btn-secondary';
        btn.style.cssText = 'position:fixed;right:24px;bottom:calc(55vh + 34px);z-index:9051;font-size:12px;padding:8px 12px;display:none;';
        btn.textContent = 'Timeline';
        btn.addEventListener('click', () => window.TimelineFusion?.show(true));
        document.body.appendChild(btn);
    }

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
            <div class="ai-meeting-actions">
                <button class="ai-btn-stop" id="aiStopBtn" onclick="AIMeeting.stop()">
                    <i class="fa fa-stop"></i> Stop AI
                </button>
            </div>
        `;
        document.body.appendChild(panel);
        showTimelineButton();
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
        ensureTimelineFusion().catch(e => console.warn('[TimelineFusion] Disabled:', e));
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
        getTimeline: () => window.TimelineFusion ? window.TimelineFusion.getTimeline() : null,
        startScreenOCR: () => window.TimelineFusion ? window.TimelineFusion.startScreenCapture() : Promise.resolve(false),
        stopScreenOCR: () => window.TimelineFusion?.stopScreenCapture()
    };
})();

// Auto-initialize when script loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => AIMeeting.init());
} else {
    AIMeeting.init();
}
