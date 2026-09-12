/**
 * MeetScribe — Multi-Modal Timeline Fusion
 * -----------------------------------------
 * Client-side fusion of transcript, typed-note and screen/OCR events.
 * All timestamps are normalized to one monotonic meeting timeline.
 *
 * The engine deliberately keeps raw timestamps and corrected timestamps so
 * alignment decisions remain auditable for R&D/benchmarking.
 */

const TimelineFusion = (() => {
    const VERSION = '1.0.0';
    const MAX_DRIFT_SECONDS = 12;
    const OCR_SAMPLE_MS = 4000;

    let sessionId = null;
    let meetingStartedAt = null;
    let events = [];
    let eventIds = new Set();
    let noteTimer = null;
    let screenStream = null;
    let screenVideo = null;
    let screenTimer = null;
    let ocrBusy = false;
    let quillBound = false;

    function now() { return performance.now() / 1000; }

    function reset(id) {
        sessionId = id || null;
        meetingStartedAt = performance.now();
        events = [];
        eventIds = new Set();
        stopScreenCapture();
    }

    function relativeNow() {
        return meetingStartedAt == null ? 0 : (performance.now() - meetingStartedAt) / 1000;
    }

    function makeId(event) {
        return [event.source, event.type, event.timestamp, event.content || event.ocrText || ''].join('|');
    }

    function normalizeEvent(input) {
        const source = input.source || input.type || 'unknown';
        const type = input.type || source;
        const raw = Number(input.timestamp);
        const observed = Number(input.observedAt);
        const rawTimestamp = Number.isFinite(raw) ? raw : relativeNow();

        // If a producer reports both capture time and arrival/observation time,
        // use the capture time. Correct only small, explainable drift.
        let corrected = rawTimestamp;
        let correction = 0;
        let correctionReason = 'none';

        if (Number.isFinite(observed)) {
            const drift = observed - rawTimestamp;
            if (drift > 0 && drift <= MAX_DRIFT_SECONDS && input.correctDrift !== false) {
                // A late-arriving event should remain at its capture moment.
                // Keep the correction explicit rather than silently overwriting it.
                corrected = rawTimestamp;
                correction = -drift;
                correctionReason = 'late-arrival';
            }
        }

        return {
            id: input.id || makeId(input),
            timestamp: Math.max(0, corrected),
            rawTimestamp: Math.max(0, rawTimestamp),
            observedAt: Number.isFinite(observed) ? observed : null,
            correction,
            correctionReason,
            type,
            source,
            content: input.content || input.ocrText || '',
            metadata: input.metadata || {},
            confidence: Number.isFinite(Number(input.confidence)) ? Number(input.confidence) : 1,
            createdAt: new Date().toISOString()
        };
    }

    function add(input) {
        const event = normalizeEvent(input);
        if (eventIds.has(event.id)) return event;
        eventIds.add(event.id);
        events.push(event);
        events.sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id));
        render();
        return event;
    }

    function addTranscript(segment, chunkStartSeconds) {
        const start = Number(chunkStartSeconds || 0) + Number(segment.start || 0);
        const end = Number(chunkStartSeconds || 0) + Number(segment.end || segment.start || 0);
        return add({
            id: `transcript:${sessionId}:${chunkStartSeconds}:${segment.start}:${segment.end}:${segment.text}`,
            timestamp: start,
            type: 'transcript',
            source: 'audio',
            content: segment.text,
            confidence: segment.confidence,
            metadata: {
                speaker: segment.speaker,
                speakerIndex: segment.speakerIndex,
                end,
                chunkStart: Number(chunkStartSeconds || 0)
            }
        });
    }

    function addNote(content, timestamp, metadata = {}) {
        if (!content || !content.trim()) return null;
        return add({
            id: `note:${sessionId}:${timestamp}:${content.trim()}`,
            timestamp: Number(timestamp),
            type: 'note',
            source: 'typed-note',
            content: content.trim(),
            metadata
        });
    }

    function addOCR(text, captureTimestamp, observedAt, metadata = {}) {
        if (!text || !text.trim()) return null;
        return add({
            id: `ocr:${sessionId}:${captureTimestamp}:${text.trim()}`,
            timestamp: Number(captureTimestamp),
            observedAt: Number(observedAt),
            type: 'ocr',
            source: 'screen',
            content: text.trim(),
            metadata
        });
    }

    function getEvents() { return events.map(e => ({ ...e, metadata: { ...e.metadata } })); }

    function getTimeline() {
        return {
            version: VERSION,
            sessionId,
            generatedAt: new Date().toISOString(),
            events: getEvents()
        };
    }

    function clear() {
        events = [];
        eventIds = new Set();
        render();
    }

    function attachQuill(quill) {
        if (!quill || quillBound) return;
        quillBound = true;
        quill.on('text-change', (delta, oldDelta, source) => {
            if (source !== 'user') return;
            const inserted = (delta.ops || [])
                .filter(op => typeof op.insert === 'string')
                .map(op => op.insert.replace(/\n/g, ' '))
                .join(' ')
                .trim();
            if (!inserted) return;
            clearTimeout(noteTimer);
            noteTimer = setTimeout(() => addNote(inserted, relativeNow(), { editor: 'quill' }), 600);
        });
    }

    async function ensureTesseract() {
        if (window.Tesseract) return window.Tesseract;
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';
            script.onload = resolve;
            script.onerror = () => reject(new Error('Unable to load OCR engine'));
            document.head.appendChild(script);
        });
        return window.Tesseract;
    }

    async function captureOCRFrame() {
        if (!screenVideo || ocrBusy || !screenStream) return;
        ocrBusy = true;
        try {
            const canvas = document.createElement('canvas');
            canvas.width = screenVideo.videoWidth || 1280;
            canvas.height = screenVideo.videoHeight || 720;
            const ctx = canvas.getContext('2d', { alpha: false });
            ctx.drawImage(screenVideo, 0, 0, canvas.width, canvas.height);
            const captureTimestamp = relativeNow();
            const observedAt = relativeNow();
            const Tesseract = await ensureTesseract();
            const result = await Tesseract.recognize(canvas, 'eng', { logger: () => {} });
            const text = result?.data?.text || '';
            if (text.trim()) {
                addOCR(text, captureTimestamp, observedAt, {
                    width: canvas.width,
                    height: canvas.height,
                    confidence: result?.data?.confidence ?? null
                });
            }
        } catch (error) {
            console.warn('[TimelineFusion] OCR capture failed:', error);
        } finally {
            ocrBusy = false;
        }
    }

    async function startScreenCapture() {
        if (screenStream) return true;
        if (!navigator.mediaDevices?.getDisplayMedia) {
            console.warn('[TimelineFusion] Screen capture is not supported by this browser.');
            return false;
        }
        try {
            screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
            screenVideo = document.createElement('video');
            screenVideo.muted = true;
            screenVideo.playsInline = true;
            screenVideo.srcObject = screenStream;
            await screenVideo.play();
            const track = screenStream.getVideoTracks()[0];
            track?.addEventListener('ended', stopScreenCapture, { once: true });
            await captureOCRFrame();
            screenTimer = setInterval(captureOCRFrame, OCR_SAMPLE_MS);
            setStatus('Screen/OCR capture active');
            return true;
        } catch (error) {
            console.warn('[TimelineFusion] Screen capture cancelled/failed:', error);
            stopScreenCapture();
            return false;
        }
    }

    function stopScreenCapture() {
        if (screenTimer) clearInterval(screenTimer);
        screenTimer = null;
        if (screenStream) screenStream.getTracks().forEach(track => track.stop());
        screenStream = null;
        if (screenVideo) screenVideo.srcObject = null;
        screenVideo = null;
    }

    function setStatus(message) {
        const el = document.getElementById('timelineFusionStatus');
        if (el) el.textContent = message;
    }

    function render() {
        const container = document.getElementById('timelineFusionEvents');
        const count = document.getElementById('timelineFusionCount');
        if (!container) return;
        if (count) count.textContent = `${events.length} events`;
        container.innerHTML = events.slice(-40).map(event => {
            const t = event.timestamp.toFixed(2);
            const correction = event.correction ? ` · corrected ${event.correction.toFixed(2)}s` : '';
            return `<div class="timeline-fusion-event"><span class="timeline-fusion-time">${t}s</span><span class="timeline-fusion-type">${event.type}</span><span class="timeline-fusion-text">${escapeHtml(event.content).slice(0, 180)}</span><span class="timeline-fusion-correction">${correction}</span></div>`;
        }).join('');
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>\"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;' }[c]));
    }

    function injectUI() {
        if (document.getElementById('timelineFusionPanel')) return;
        const panel = document.createElement('div');
        panel.id = 'timelineFusionPanel';
        panel.style.cssText = 'position:fixed;right:24px;bottom:24px;width:420px;max-height:55vh;background:var(--bg-2,#111d32);border:1px solid var(--border-med,rgba(255,255,255,.12));border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.35);z-index:9050;overflow:hidden;font-family:DM Sans,sans-serif;display:none;';
        panel.innerHTML = `
          <div style="padding:12px 14px;border-bottom:1px solid var(--border,rgba(255,255,255,.08));display:flex;align-items:center;justify-content:space-between;gap:10px;">
            <div><strong>Unified Timeline</strong><div id="timelineFusionStatus" style="font-size:11px;color:var(--text-3,#71809b);margin-top:2px">Ready</div></div>
            <span id="timelineFusionCount" style="font-size:11px;color:var(--text-2,#8fa3c8)">0 events</span>
          </div>
          <div id="timelineFusionEvents" style="padding:8px 12px;max-height:32vh;overflow:auto"></div>
          <div style="padding:10px 12px;border-top:1px solid var(--border,rgba(255,255,255,.08));display:flex;gap:8px;">
            <button id="timelineOCRBtn" class="btn-secondary" style="font-size:11px;padding:7px 10px">Start Screen/OCR</button>
            <button id="timelineExportBtn" class="btn-secondary" style="font-size:11px;padding:7px 10px">Export Timeline</button>
          </div>`;
        document.body.appendChild(panel);
        document.getElementById('timelineOCRBtn').addEventListener('click', async () => {
            if (screenStream) { stopScreenCapture(); setStatus('Screen/OCR capture stopped'); }
            else { await startScreenCapture(); }
            document.getElementById('timelineOCRBtn').textContent = screenStream ? 'Stop Screen/OCR' : 'Start Screen/OCR';
        });
        document.getElementById('timelineExportBtn').addEventListener('click', exportTimeline);
    }

    function show(visible = true) {
        const panel = document.getElementById('timelineFusionPanel');
        if (panel) panel.style.display = visible ? 'block' : 'none';
    }

    function exportTimeline() {
        const blob = new Blob([JSON.stringify(getTimeline(), null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `meetscribe-timeline-${sessionId || 'session'}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    function init(id) {
        reset(id);
        injectUI();
        show(true);
        if (window.quill) attachQuill(window.quill);
        setStatus('Listening for transcript and note events');
    }

    return {
        init, reset, add, addTranscript, addNote, addOCR,
        attachQuill, startScreenCapture, stopScreenCapture,
        getEvents, getTimeline, clear, show
    };
})();

window.TimelineFusion = TimelineFusion;
