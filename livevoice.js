/**
 * AirScribe AI LiveVoice — Accurate Real-Time Voice Typing
 * ==========================================================
 * 
 * Enhances the Live Meeting feature with precise voice-to-text.
 * 
 * KEY ACCURACY FIXES:
 *   - Only inserts FINAL results (isFinal === true)
 *   - Prevents duplicate words via last-text tracking
 *   - Auto-restarts recognition when it stops
 *   - Smooth sentence flow with proper spacing/capitalization
 *   - Debounced interim preview (visual only, never inserted)
 * 
 * PRESERVES:
 *   - All existing MeetingRecorder functionality
 *   - Manual speaker name assignment
 *   - Existing voice record/stop buttons
 * 
 * USAGE: Just include this script in AirScribe_v3.html.
 *        It auto-initializes and adds a toggle button to the Voice Input section.
 */

const LiveVoice = (() => {
    'use strict';

    // ── State ──
    let recognition = null;
    let isListening = false;
    let restartCount = 0;
    const MAX_RESTARTS = 100; // Allow very long sessions

    // Accuracy tracking
    let lastFinalText = '';       // Last inserted final text (prevents duplicates)
    let lastInsertTime = 0;       // Timestamp of last insertion
    let totalWordsInserted = 0;   // Stats counter

    // Interim display
    let interimTimeout = null;
    const INTERIM_DEBOUNCE = 80;  // ms — smooth interim updates

    // ── Check browser support ──
    function isSupported() {
        return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    }

    // ── Create recognition instance with accuracy settings ──
    function createRecognition() {
        const SpeechR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechR) return null;

        const rec = new SpeechR();
        rec.lang = 'en-US';
        rec.continuous = true;         // Don't stop after one phrase
        rec.interimResults = true;     // Show live preview
        rec.maxAlternatives = 1;       // Single best result

        rec.onresult = handleResult;
        rec.onerror = handleError;
        rec.onend = handleEnd;

        return rec;
    }

    // ══════════════════════════════════════════════════
    //  RESULT HANDLER — The core accuracy logic
    // ══════════════════════════════════════════════════
    function handleResult(event) {
        if (!isListening) return;

        restartCount = 0; // Reset on any successful result
        let finalText = '';
        let interimText = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const transcript = result[0].transcript;

            if (result.isFinal) {
                // ── ONLY insert FINAL results ──
                finalText += transcript;
            } else {
                interimText += transcript;
            }
        }

        // Show interim preview (visual only — never inserted into editor)
        if (interimText && !finalText) {
            showInterimPreview(interimText);
            updateStatusText('Listening...');
        }

        // Insert final text with duplicate prevention
        if (finalText) {
            clearInterimPreview();
            insertAccurateText(finalText);
        }
    }

    // ── Insert text with accuracy safeguards ──
    function insertAccurateText(rawText) {
        let text = rawText.trim();
        if (!text) return;

        // ── DUPLICATE PREVENTION ──
        // Check if this exact text was just inserted
        if (text === lastFinalText && (Date.now() - lastInsertTime) < 2000) {
            console.log('[LiveVoice] Skipping duplicate:', text);
            return;
        }

        // Check if this text starts with the end of the last text (partial overlap)
        if (lastFinalText && text.length > 3) {
            const overlapCheck = lastFinalText.slice(-Math.min(20, lastFinalText.length));
            if (text.startsWith(overlapCheck)) {
                text = text.slice(overlapCheck.length).trim();
                if (!text) return;
            }
        }

        // ── FORMAT TEXT ──
        // Capitalize first letter
        text = text.charAt(0).toUpperCase() + text.slice(1);

        // Add period if no ending punctuation
        if (!/[.!?,;:]$/.test(text)) {
            text += '.';
        }

        // Add space after
        text += ' ';

        // ── INSERT INTO QUILL ──
        if (window.quill) {
            const len = window.quill.getLength();
            window.quill.insertText(len - 1, text);

            // Scroll to bottom
            const editor = document.querySelector('.ql-editor');
            if (editor) editor.scrollTop = editor.scrollHeight;

            // Update stats
            if (typeof updateEditorStats === 'function') updateEditorStats();
            if (typeof setSaveStatus === 'function') setSaveStatus('Unsaved...');
        }

        // Track for duplicate prevention
        lastFinalText = rawText.trim();
        lastInsertTime = Date.now();
        totalWordsInserted += text.split(/\s+/).length;

        updateStatusText('Inserted: ' + text.trim().substring(0, 40) + (text.length > 40 ? '...' : ''));
        updateWordCount();
    }

    // ── Interim preview (visual feedback only) ──
    function showInterimPreview(text) {
        clearTimeout(interimTimeout);
        interimTimeout = setTimeout(() => {
            const el = document.getElementById('liveVoiceInterim');
            if (el) {
                const preview = text.length > 60 ? '...' + text.slice(-60) : text;
                el.textContent = preview;
                el.style.opacity = '1';
            }
        }, INTERIM_DEBOUNCE);
    }

    function clearInterimPreview() {
        clearTimeout(interimTimeout);
        const el = document.getElementById('liveVoiceInterim');
        if (el) {
            el.textContent = '';
            el.style.opacity = '0';
        }
    }

    // ══════════════════════════════════════════════════
    //  ERROR & AUTO-RESTART
    // ══════════════════════════════════════════════════
    function handleError(event) {
        if (!isListening) return;
        console.warn('[LiveVoice] Error:', event.error);

        switch (event.error) {
            case 'not-allowed':
            case 'permission-denied':
                showToastMsg('Microphone permission denied. Allow mic access in browser settings.', 'error');
                stop();
                return;
            case 'audio-capture':
                showToastMsg('No microphone found. Please connect a microphone.', 'error');
                stop();
                return;
            case 'no-speech':
                updateStatusText('No speech detected — still listening...');
                return; // Let handleEnd restart
            case 'network':
                updateStatusText('Network issue — reconnecting...');
                return;
            case 'aborted':
                return;
            default:
                updateStatusText('Error: ' + event.error);
        }
    }

    function handleEnd() {
        if (!isListening) return;

        restartCount++;
        if (restartCount > MAX_RESTARTS) {
            showToastMsg('Voice typing stopped — too many restarts. Click to restart.', 'error');
            stop();
            return;
        }

        // Auto-restart with fresh instance (prevents Chrome stale state)
        setTimeout(() => {
            if (!isListening) return;
            try {
                recognition = createRecognition();
                if (recognition) {
                    recognition.start();
                    updateStatusText('Listening...');
                }
            } catch (e) {
                console.warn('[LiveVoice] Restart failed:', e);
                if (isListening && restartCount < MAX_RESTARTS) {
                    setTimeout(handleEnd, 800); // Retry after longer delay
                }
            }
        }, 250);
    }

    // ══════════════════════════════════════════════════
    //  START / STOP
    // ══════════════════════════════════════════════════
    function start() {
        if (!isSupported()) {
            showToastMsg('Speech recognition not supported. Use Chrome or Edge.', 'error');
            return false;
        }
        if (isListening) return true; // Already running

        recognition = createRecognition();
        if (!recognition) return false;

        // Reset state
        isListening = true;
        restartCount = 0;
        lastFinalText = '';
        lastInsertTime = 0;
        totalWordsInserted = 0;

        try {
            recognition.start();
        } catch (e) {
            showToastMsg('Failed to start voice typing: ' + e.message, 'error');
            isListening = false;
            return false;
        }

        // UI
        updateToggleButton(true);
        updateStatusText('Listening...');
        showToastMsg('Voice typing started', 'info');
        return true;
    }

    function stop() {
        if (!isListening) return;
        isListening = false;

        clearInterimPreview();

        if (recognition) {
            try { recognition.onend = null; } catch (e) { /* ignore */ }
            try { recognition.stop(); } catch (e) { /* ignore */ }
            recognition = null;
        }

        // UI
        updateToggleButton(false);
        updateStatusText('Stopped');
        if (totalWordsInserted > 0) {
            showToastMsg(`Voice typing stopped — ${totalWordsInserted} words captured`, 'success');
        }
    }

    function toggle() {
        if (isListening) stop();
        else start();
    }

    // ══════════════════════════════════════════════════
    //  UI — Injected into the existing Voice Input section
    // ══════════════════════════════════════════════════
    function updateToggleButton(active) {
        const btn = document.getElementById('liveVoiceBtn');
        if (!btn) return;
        if (active) {
            btn.classList.add('active');
            btn.innerHTML = '<i class="fa fa-microphone-slash"></i> Stop Voice Typing';
        } else {
            btn.classList.remove('active');
            btn.innerHTML = '<i class="fa fa-keyboard"></i> Voice Typing';
        }
    }

    function updateStatusText(text) {
        const el = document.getElementById('liveVoiceStatus');
        if (el) el.textContent = text;
    }

    function updateWordCount() {
        const el = document.getElementById('liveVoiceWords');
        if (el) el.textContent = totalWordsInserted + ' words';
    }

    function showToastMsg(msg, type) {
        if (typeof showToast === 'function') { showToast(msg, type); return; }
        console.log(`[LiveVoice][${type}] ${msg}`);
    }

    // ── Inject UI into the page ──
    function injectUI() {
        // Find the VOICE INPUT section in the editor sidebar
        const voiceSection = findVoiceSection();
        if (!voiceSection) {
            console.warn('[LiveVoice] Voice Input section not found — will retry');
            setTimeout(injectUI, 2000);
            return;
        }

        // Check if already injected
        if (document.getElementById('liveVoiceBtn')) return;

        // Create the voice typing UI block
        const container = document.createElement('div');
        container.id = 'liveVoiceContainer';
        container.innerHTML = `
            <button class="voice-btn livevoice" id="liveVoiceBtn" onclick="LiveVoice.toggle()" 
                    style="background:rgba(167,139,250,0.12);color:#a78bfa;border-color:rgba(167,139,250,0.25);width:100%;margin-top:8px">
                <i class="fa fa-keyboard"></i> Voice Typing
            </button>
            <div id="liveVoiceStatusBar" style="display:flex;align-items:center;gap:8px;margin-top:6px;min-height:20px">
                <div id="liveVoiceIndicator" style="width:6px;height:6px;border-radius:50%;background:#4a6080;flex-shrink:0;transition:all 0.3s"></div>
                <span id="liveVoiceStatus" style="font-size:11px;color:var(--text-3,#4a6080)">Ready</span>
                <span id="liveVoiceWords" style="font-size:10px;color:var(--text-3,#4a6080);margin-left:auto"></span>
            </div>
            <div id="liveVoiceInterim" style="font-size:11px;color:#a78bfa;font-style:italic;min-height:14px;margin-top:2px;opacity:0;transition:opacity 0.2s;word-break:break-word"></div>
        `;

        // Insert after existing voice controls
        const voiceControls = voiceSection.querySelector('.voice-controls');
        if (voiceControls) {
            voiceControls.parentNode.insertBefore(container, voiceControls.nextSibling);
        } else {
            voiceSection.appendChild(container);
        }

        // Add CSS for active state
        injectStyles();
    }

    function findVoiceSection() {
        // Find the es-section containing "VOICE INPUT"
        const sections = document.querySelectorAll('.es-section');
        for (const sec of sections) {
            const label = sec.querySelector('.es-label');
            if (label && label.textContent.trim() === 'VOICE INPUT') return sec;
        }
        return null;
    }

    function injectStyles() {
        if (document.getElementById('liveVoiceStyles')) return;
        const style = document.createElement('style');
        style.id = 'liveVoiceStyles';
        style.textContent = `
            #liveVoiceBtn.active {
                background: rgba(167,139,250,0.2) !important;
                color: #a78bfa !important;
                border-color: #a78bfa !important;
                animation: lv-pulse 1.5s infinite;
            }
            #liveVoiceBtn.active + #liveVoiceStatusBar #liveVoiceIndicator,
            #liveVoiceBtn.active ~ #liveVoiceStatusBar #liveVoiceIndicator {
                background: #34d399 !important;
                box-shadow: 0 0 6px rgba(52,211,153,0.5);
            }
            @keyframes lv-pulse {
                0%, 100% { box-shadow: 0 0 0 0 rgba(167,139,250,0.3); }
                50% { box-shadow: 0 0 0 6px rgba(167,139,250,0); }
            }
        `;
        document.head.appendChild(style);
    }

    // ── Initialize ──
    function init() {
        // Wait for DOM to be ready and Quill to exist
        const tryInject = () => {
            if (document.querySelector('.es-section')) {
                injectUI();
            } else {
                setTimeout(tryInject, 1000);
            }
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => setTimeout(tryInject, 1500));
        } else {
            setTimeout(tryInject, 1500); // Wait for AirScribe app init
        }
    }

    // Auto-initialize
    init();

    // ── Public API ──
    return {
        start,
        stop,
        toggle,
        isActive: () => isListening
    };
})();
