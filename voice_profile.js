/**
 * AirScribe — Voice Profile Module
 * ===================================
 * Frontend for voice enrollment and profile management.
 *
 * Purpose:
 *   Allows users to record their voice (~30s) to create a voice profile.
 *   The profile is sent to the AI server for embedding extraction and
 *   stored in MongoDB. During meetings, enrolled profiles are used to
 *   automatically identify speakers by name.
 *
 * How it connects:
 *   - Injected into the AirScribe_v3.html Voice Profile view.
 *   - Communicates with the AI server at http://localhost:5001/ai/voice/*.
 *   - Uses the same IIFE pattern as AIMeeting and LiveVoice modules.
 *
 * How to debug:
 *   - All actions log to console via [VoiceProfile] prefix.
 *   - The AI server logs enrollment/deletion operations.
 *   - Check MongoDB voice_profiles collection for stored profiles.
 *
 * IMPORTANT: Does NOT modify any existing AirScribe functionality.
 */

const VoiceProfile = (() => {
    'use strict';

    // ── Configuration ──
    const AI_API = 'http://localhost:5001';
    const ENROLLMENT_DURATION = 30; // seconds
    const MIC_BAR_COUNT = 12;

    // ── State ──
    let mediaRecorder = null;
    let audioStream = null;
    let audioChunks = [];
    let isRecording = false;
    let recordingTimer = null;
    let elapsedSeconds = 0;
    let micAnalyser = null;
    let micAnimFrame = null;
    let profiles = [];
    let currentUserId = '';
    let currentUserName = '';

    // ══════════════════════════════════════════════════
    //  INITIALIZATION
    // ══════════════════════════════════════════════════

    /**
     * Initialize the Voice Profile module.
     * Fetches user info and renders the UI skeleton.
     */
    function init() {
        // Get current user info from auth
        _fetchUserInfo();
    }

    /**
     * Load and display all enrolled voice profiles.
     * Called when the user navigates to the Voice Profile view.
     */
    async function loadProfiles() {
        await _fetchUserInfo();
        _renderUI();
        await _fetchProfiles();
    }

    // ══════════════════════════════════════════════════
    //  RECORDING
    // ══════════════════════════════════════════════════

    /**
     * Start recording the user's voice for enrollment.
     */
    async function startRecording() {
        if (isRecording) return;

        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
        } catch (e) {
            _showToast('Microphone access denied. Please allow mic permission.', 'error');
            console.error('[VoiceProfile] Mic access denied:', e);
            return;
        }

        // Setup MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';

        mediaRecorder = new MediaRecorder(audioStream, { mimeType });
        audioChunks = [];
        elapsedSeconds = 0;
        isRecording = true;

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.start(1000); // Collect data every second

        // Start mic level visualization
        _startMicVisualization();

        // Start countdown timer
        recordingTimer = setInterval(() => {
            elapsedSeconds++;
            _updateProgress(elapsedSeconds);

            if (elapsedSeconds >= ENROLLMENT_DURATION) {
                stopRecording();
            }
        }, 1000);

        _updateRecordingUI('recording');
        _showToast('Recording your voice... Please speak naturally for 30 seconds.', 'info');
        console.log('[VoiceProfile] Recording started');
    }

    /**
     * Stop recording and send the audio for enrollment.
     */
    async function stopRecording() {
        if (!isRecording) return;
        isRecording = false;

        // Stop timer
        clearInterval(recordingTimer);
        recordingTimer = null;

        // Stop mic visualization
        _stopMicVisualization();

        // Stop MediaRecorder
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            // Wait for the last data chunk
            await new Promise(resolve => {
                mediaRecorder.onstop = resolve;
                mediaRecorder.stop();
            });
        }

        // Stop mic stream
        if (audioStream) {
            audioStream.getTracks().forEach(t => t.stop());
            audioStream = null;
        }

        // Check if we have enough audio
        if (elapsedSeconds < 10) {
            _showToast('Recording too short. Please record for at least 15 seconds.', 'error');
            _updateRecordingUI('idle');
            return;
        }

        // Send to AI server for enrollment
        _updateRecordingUI('processing');
        await _enrollVoice();
    }

    /**
     * Cancel an active recording without enrolling.
     */
    function cancelRecording() {
        if (!isRecording) return;
        isRecording = false;

        clearInterval(recordingTimer);
        recordingTimer = null;
        _stopMicVisualization();

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
        }
        if (audioStream) {
            audioStream.getTracks().forEach(t => t.stop());
            audioStream = null;
        }

        audioChunks = [];
        _updateRecordingUI('idle');
        _showToast('Recording cancelled', 'info');
    }

    // ══════════════════════════════════════════════════
    //  API CALLS
    // ══════════════════════════════════════════════════

    /**
     * Send the recorded audio to the AI server for enrollment.
     */
    async function _enrollVoice() {
        const blob = new Blob(audioChunks, {
            type: mediaRecorder?.mimeType || 'audio/webm'
        });
        audioChunks = [];

        if (blob.size < 5000) {
            _showToast('Audio too short or empty. Please try again.', 'error');
            _updateRecordingUI('idle');
            return;
        }

        const formData = new FormData();
        formData.append('audio', blob, 'enrollment.webm');
        formData.append('userId', currentUserId);
        formData.append('userName', currentUserName);

        try {
            _updateRecordingUI('processing');

            const res = await fetch(`${AI_API}/ai/voice/enroll`, {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (res.ok && data.success) {
                _showToast(`Voice profile for "${data.userName}" enrolled successfully!`, 'success');
                _updateRecordingUI('complete');
                // Refresh profile list
                setTimeout(() => _fetchProfiles(), 500);
            } else {
                _showToast(data.error || 'Enrollment failed. Please try again.', 'error');
                _updateRecordingUI('idle');
            }
        } catch (e) {
            console.error('[VoiceProfile] Enrollment error:', e);
            _showToast(
                'Could not connect to AI server. Make sure it\'s running on port 5001.',
                'error'
            );
            _updateRecordingUI('idle');
        }
    }

    /**
     * Fetch all enrolled profiles for the current user.
     */
    async function _fetchProfiles() {
        try {
            const url = currentUserId
                ? `${AI_API}/ai/voice/profiles?userId=${encodeURIComponent(currentUserId)}`
                : `${AI_API}/ai/voice/profiles`;

            const res = await fetch(url);
            if (res.ok) {
                profiles = await res.json();
                _renderProfileList();
            }
        } catch (e) {
            console.warn('[VoiceProfile] Could not fetch profiles:', e);
            profiles = [];
            _renderProfileList();
        }
    }

    /**
     * Delete a voice profile.
     */
    async function deleteProfile(profileId) {
        if (!confirm('Delete this voice profile? This cannot be undone.')) return;

        try {
            const res = await fetch(`${AI_API}/ai/voice/profile/${profileId}`, {
                method: 'DELETE'
            });

            if (res.ok) {
                _showToast('Voice profile deleted', 'success');
                await _fetchProfiles();
                _updateRecordingUI('idle');
            } else {
                const data = await res.json();
                _showToast(data.error || 'Failed to delete profile', 'error');
            }
        } catch (e) {
            _showToast('Could not connect to AI server', 'error');
        }
    }

    /**
     * Test the microphone.
     */
    async function testMicrophone() {
        const statusEl = document.getElementById('vpMicTestStatus');
        const indicatorEl = document.getElementById('vpMicIndicator');

        if (!statusEl) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            statusEl.textContent = 'Recording test...';
            if (indicatorEl) indicatorEl.className = 'vp-mic-indicator active';

            // Record 2 seconds
            const recorder = new MediaRecorder(stream);
            const chunks = [];
            recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

            recorder.start();
            await new Promise(r => setTimeout(r, 2000));
            recorder.stop();
            stream.getTracks().forEach(t => t.stop());

            await new Promise(r => { recorder.onstop = r; });

            const blob = new Blob(chunks, { type: recorder.mimeType });
            const formData = new FormData();
            formData.append('audio', blob, 'mictest.webm');

            statusEl.textContent = 'Analyzing...';

            const res = await fetch(`${AI_API}/ai/voice/test-mic`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                if (data.isUsable) {
                    statusEl.textContent = `Microphone working (volume: ${Math.round(data.volumeLevel * 100)}%)`;
                    if (indicatorEl) indicatorEl.className = 'vp-mic-indicator active';
                } else {
                    statusEl.textContent = data.message || 'Audio quality issue';
                    if (indicatorEl) indicatorEl.className = 'vp-mic-indicator error';
                }
            } else {
                statusEl.textContent = 'Test failed — AI server error';
                if (indicatorEl) indicatorEl.className = 'vp-mic-indicator error';
            }
        } catch (e) {
            statusEl.textContent = e.name === 'NotAllowedError'
                ? 'Microphone permission denied'
                : 'Could not access microphone';
            if (indicatorEl) indicatorEl.className = 'vp-mic-indicator error';
        }
    }

    // ══════════════════════════════════════════════════
    //  USER INFO
    // ══════════════════════════════════════════════════

    async function _fetchUserInfo() {
        try {
            const res = await fetch('/auth/status');
            const data = await res.json();
            if (data.authenticated && data.user) {
                currentUserId = data.user.id || '';
                currentUserName = data.user.name || data.user.email || '';
            }
        } catch (e) {
            // Fallback for local dev without auth
            currentUserId = 'local_user';
            currentUserName = 'User';
        }
    }

    // ══════════════════════════════════════════════════
    //  MIC VISUALIZATION
    // ══════════════════════════════════════════════════

    function _startMicVisualization() {
        if (!audioStream) return;

        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(audioStream);
            micAnalyser = audioCtx.createAnalyser();
            micAnalyser.fftSize = 64;
            source.connect(micAnalyser);

            const dataArray = new Uint8Array(micAnalyser.frequencyBinCount);

            const animate = () => {
                if (!isRecording) return;
                micAnalyser.getByteFrequencyData(dataArray);

                const bars = document.querySelectorAll('.vp-mic-bar');
                bars.forEach((bar, i) => {
                    const value = dataArray[i * 2] || 0;
                    const height = Math.max(3, (value / 255) * 30);
                    bar.style.height = `${height}px`;
                });

                micAnimFrame = requestAnimationFrame(animate);
            };

            animate();
        } catch (e) {
            console.warn('[VoiceProfile] Mic visualization failed:', e);
        }
    }

    function _stopMicVisualization() {
        if (micAnimFrame) {
            cancelAnimationFrame(micAnimFrame);
            micAnimFrame = null;
        }
        micAnalyser = null;

        // Reset bars
        const bars = document.querySelectorAll('.vp-mic-bar');
        bars.forEach(bar => { bar.style.height = '3px'; });
    }

    // ══════════════════════════════════════════════════
    //  UI RENDERING
    // ══════════════════════════════════════════════════

    function _renderUI() {
        const container = document.getElementById('voiceProfileContent');
        if (!container) return;

        // Build mic bars HTML
        let micBars = '';
        for (let i = 0; i < MIC_BAR_COUNT; i++) {
            micBars += '<div class="vp-mic-bar"></div>';
        }

        container.innerHTML = `
            <div class="vp-container">
                <!-- Enrollment Section -->
                <div class="vp-section">
                    <div class="vp-section-title">
                        <i class="fa fa-waveform-lines"></i> Voice Enrollment
                    </div>
                    <div class="vp-section-desc">
                        Record your voice for 30 seconds. Speak naturally — read a paragraph, 
                        describe your day, or talk about anything. This creates a unique voice 
                        fingerprint for automatic speaker recognition in meetings.
                    </div>

                    <div class="vp-enrollment-card" id="vpEnrollmentCard">
                        <div id="vpRecordVisual">
                            <div class="vp-record-icon" id="vpRecordIcon">
                                <i class="fa fa-microphone" id="vpRecordIconI"></i>
                            </div>
                        </div>

                        <div class="vp-mic-level" id="vpMicLevel" style="display:none">
                            ${micBars}
                        </div>

                        <div class="vp-enrollment-label" id="vpEnrollLabel">
                            Ready to Record
                        </div>
                        <div class="vp-enrollment-hint" id="vpEnrollHint">
                            Click the button below to start recording your voice sample.
                        </div>

                        <div class="vp-btn-group" id="vpBtnGroup">
                            <button class="vp-btn vp-btn-primary" id="vpRecordBtn" onclick="VoiceProfile.startRecording()">
                                <i class="fa fa-microphone"></i> Start Recording
                            </button>
                        </div>
                    </div>

                    <div class="vp-info-box">
                        <i class="fa fa-shield-halved"></i>
                        <p>
                            Your voice recording is processed instantly and never stored permanently. 
                            Only a mathematical fingerprint (embedding) is saved — your actual voice 
                            cannot be reconstructed from it.
                        </p>
                    </div>
                </div>

                <!-- Mic Test Section -->
                <div class="vp-section">
                    <div class="vp-section-title">
                        <i class="fa fa-volume-high"></i> Microphone Test
                    </div>
                    <div class="vp-mic-test">
                        <div class="vp-mic-indicator" id="vpMicIndicator"></div>
                        <div class="vp-mic-test-status" id="vpMicTestStatus">
                            Click "Test" to check your microphone
                        </div>
                        <button class="vp-btn vp-btn-ghost" onclick="VoiceProfile.testMicrophone()" style="padding:6px 14px;font-size:12px">
                            <i class="fa fa-play"></i> Test
                        </button>
                    </div>
                </div>

                <!-- Enrolled Profiles Section -->
                <div class="vp-section">
                    <div class="vp-section-title">
                        <i class="fa fa-users"></i> Enrolled Profiles
                    </div>
                    <div class="vp-section-desc">
                        Voice profiles enrolled on this account. During meetings, these profiles 
                        are used to automatically identify and label speakers by name.
                    </div>
                    <div id="vpProfileList">
                        <div class="vp-empty">
                            <i class="fa fa-user-plus"></i>
                            <p>Loading profiles...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function _renderProfileList() {
        const container = document.getElementById('vpProfileList');
        if (!container) return;

        if (!profiles || profiles.length === 0) {
            container.innerHTML = `
                <div class="vp-empty">
                    <i class="fa fa-user-plus"></i>
                    <p>No voice profiles enrolled yet.<br>
                    Record your voice above to enable automatic speaker recognition.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = profiles.map(p => {
            const date = p.updatedAt || p.createdAt || '';
            const formattedDate = date ? new Date(date).toLocaleDateString('en-US', {
                day: '2-digit', month: 'short', year: 'numeric'
            }) : '';
            const duration = p.sampleDuration ? `${Math.round(p.sampleDuration)}s sample` : '';

            return `
                <div class="vp-profile-item">
                    <div class="vp-profile-avatar">
                        <i class="fa fa-user"></i>
                    </div>
                    <div class="vp-profile-info">
                        <div class="vp-profile-name">${_escHtml(p.userName || 'Unknown')}</div>
                        <div class="vp-profile-meta">
                            <span><i class="fa fa-clock" style="margin-right:3px"></i>${formattedDate}</span>
                            ${duration ? `<span><i class="fa fa-waveform-lines" style="margin-right:3px"></i>${duration}</span>` : ''}
                            <span class="vp-status enrolled"><i class="fa fa-check-circle"></i> Enrolled</span>
                        </div>
                    </div>
                    <div class="vp-profile-actions">
                        <button class="vp-profile-action-btn" title="Re-record voice"
                                onclick="VoiceProfile.startRecording()">
                            <i class="fa fa-rotate"></i>
                        </button>
                        <button class="vp-profile-action-btn danger" title="Delete profile"
                                onclick="VoiceProfile.deleteProfile('${p._id}')">
                            <i class="fa fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }

    function _updateRecordingUI(state) {
        const card = document.getElementById('vpEnrollmentCard');
        const icon = document.getElementById('vpRecordIcon');
        const iconI = document.getElementById('vpRecordIconI');
        const label = document.getElementById('vpEnrollLabel');
        const hint = document.getElementById('vpEnrollHint');
        const btnGroup = document.getElementById('vpBtnGroup');
        const micLevel = document.getElementById('vpMicLevel');
        const recordVisual = document.getElementById('vpRecordVisual');

        if (!card) return;

        switch (state) {
            case 'recording':
                card.className = 'vp-enrollment-card recording';
                icon.className = 'vp-record-icon recording';
                iconI.className = 'fa fa-stop';
                label.textContent = `Recording... ${elapsedSeconds}s / ${ENROLLMENT_DURATION}s`;
                hint.textContent = 'Speak naturally. Read something aloud or talk about your day.';
                if (micLevel) micLevel.style.display = 'flex';

                // Show progress ring
                _showProgressRing(recordVisual);

                btnGroup.innerHTML = `
                    <button class="vp-btn vp-btn-danger" onclick="VoiceProfile.stopRecording()">
                        <i class="fa fa-stop"></i> Stop & Enroll
                    </button>
                    <button class="vp-btn vp-btn-ghost" onclick="VoiceProfile.cancelRecording()">
                        Cancel
                    </button>
                `;
                break;

            case 'processing':
                card.className = 'vp-enrollment-card recording';
                icon.className = 'vp-record-icon';
                iconI.className = 'fa fa-spinner fa-spin';
                label.textContent = 'Processing your voice...';
                hint.textContent = 'Extracting voice fingerprint. This may take a few seconds.';
                if (micLevel) micLevel.style.display = 'none';
                _hideProgressRing(recordVisual);

                btnGroup.innerHTML = `
                    <button class="vp-btn vp-btn-primary" disabled>
                        <i class="fa fa-spinner fa-spin"></i> Processing...
                    </button>
                `;
                break;

            case 'complete':
                card.className = 'vp-enrollment-card complete';
                icon.className = 'vp-record-icon complete';
                iconI.className = 'fa fa-check';
                label.textContent = 'Voice Profile Enrolled!';
                hint.textContent = `Your voice will now be automatically recognized in meetings.`;
                if (micLevel) micLevel.style.display = 'none';
                _hideProgressRing(recordVisual);

                btnGroup.innerHTML = `
                    <button class="vp-btn vp-btn-ghost" onclick="VoiceProfile.startRecording()">
                        <i class="fa fa-rotate"></i> Re-record
                    </button>
                `;
                break;

            default: // idle
                card.className = 'vp-enrollment-card';
                icon.className = 'vp-record-icon';
                iconI.className = 'fa fa-microphone';
                label.textContent = 'Ready to Record';
                hint.textContent = 'Click the button below to start recording your voice sample.';
                if (micLevel) micLevel.style.display = 'none';
                _hideProgressRing(recordVisual);

                btnGroup.innerHTML = `
                    <button class="vp-btn vp-btn-primary" onclick="VoiceProfile.startRecording()">
                        <i class="fa fa-microphone"></i> Start Recording
                    </button>
                `;
        }
    }

    function _updateProgress(seconds) {
        const label = document.getElementById('vpEnrollLabel');
        if (label && isRecording) {
            label.textContent = `Recording... ${seconds}s / ${ENROLLMENT_DURATION}s`;
        }

        // Update progress ring
        const progress = document.getElementById('vpProgressRingProgress');
        if (progress) {
            const circumference = 2 * Math.PI * 34; // radius = 34
            const offset = circumference - (seconds / ENROLLMENT_DURATION) * circumference;
            progress.style.strokeDashoffset = offset;
        }

        const text = document.getElementById('vpProgressText');
        if (text) {
            text.textContent = `${ENROLLMENT_DURATION - seconds}`;
        }
    }

    function _showProgressRing(container) {
        if (!container || document.getElementById('vpProgressContainer')) return;

        const circumference = 2 * Math.PI * 34;
        const ring = document.createElement('div');
        ring.id = 'vpProgressContainer';
        ring.className = 'vp-progress-container';
        ring.innerHTML = `
            <svg class="vp-progress-ring" width="80" height="80">
                <circle class="vp-progress-ring-circle" cx="40" cy="40" r="34"/>
                <circle class="vp-progress-ring-progress" id="vpProgressRingProgress"
                        cx="40" cy="40" r="34"
                        style="stroke-dasharray:${circumference};stroke-dashoffset:${circumference}"/>
            </svg>
            <div class="vp-progress-text" id="vpProgressText">${ENROLLMENT_DURATION}</div>
        `;

        // Replace the icon with the progress ring
        const icon = document.getElementById('vpRecordIcon');
        if (icon) icon.style.display = 'none';
        container.insertBefore(ring, container.firstChild);
    }

    function _hideProgressRing(container) {
        const ring = document.getElementById('vpProgressContainer');
        if (ring) ring.remove();

        const icon = document.getElementById('vpRecordIcon');
        if (icon) icon.style.display = 'flex';
    }

    // ══════════════════════════════════════════════════
    //  HELPERS
    // ══════════════════════════════════════════════════

    function _showToast(msg, type) {
        if (typeof showToast === 'function') {
            showToast(msg, type);
        } else {
            console.log(`[VoiceProfile] ${type}: ${msg}`);
        }
    }

    function _escHtml(str) {
        return String(str || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ── Auto-initialize ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => init());
    } else {
        init();
    }

    // ── Public API ──
    return {
        init,
        loadProfiles,
        startRecording,
        stopRecording,
        cancelRecording,
        deleteProfile,
        testMicrophone
    };
})();
