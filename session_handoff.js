/**
 * AirScribe Session Handoff — Feature #7
 * =========================================
 * Cross-device session handoff protocol (frontend).
 *
 * Integrates with the existing AIMeeting module.
 * Provides pairing, handoff initiation, snapshot transfer,
 * acknowledgement, commit, and session restore.
 */

const SessionHandoff = (() => {
    const AI_API = 'http://localhost:5001';

    // ── Device Identity ──
    let _deviceId = localStorage.getItem('airscribe_device_id');
    if (!_deviceId) {
        _deviceId = 'dev_' + crypto.randomUUID().slice(0, 12);
        localStorage.setItem('airscribe_device_id', _deviceId);
    }

    // ── State ──
    let _pairCode = null;
    let _handoffToken = null;
    let _targetSessionId = null;
    let _role = null;          // 'device_a' | 'device_b'
    let _phase = 'IDLE';
    let _snapshot = null;
    let _pollTimer = null;
    let _statusCallback = null;

    // ══════════════════════════════════════════
    //  DEVICE A — Source Device
    // ══════════════════════════════════════════

    /**
     * Create a pairing code for the current session.
     * @param {string} sessionId
     * @returns {Promise<{pair_code, expires_at}|null>}
     */
    async function createPairCode(sessionId) {
        try {
            const res = await fetch(`${AI_API}/ai/handoff/pair`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId, deviceId: _deviceId }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _pairCode = data.pair_code;
            _targetSessionId = sessionId;
            _role = 'device_a';
            _phase = 'PAIR_REQUESTED';
            _updateStatus('pairing', `Pair code: ${_pairCode}`);

            // Poll for Device B to join
            _startStatusPolling(sessionId);
            return data;
        } catch (e) {
            _updateStatus('error', 'Failed to create pair code: ' + e.message);
            return null;
        }
    }

    /**
     * Initiate handoff — freeze session, create snapshot.
     * Called by Device A after Device B has paired.
     */
    async function initiateHandoff(sessionId) {
        try {
            _phase = 'INITIATING';
            _updateStatus('syncing', 'Creating session snapshot...');

            const res = await fetch(`${AI_API}/ai/handoff/initiate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId, deviceId: _deviceId }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _phase = 'SNAPSHOT_CREATED';
            _updateStatus('syncing', 'Snapshot created — waiting for Device B...');
            return data;
        } catch (e) {
            _updateStatus('error', 'Handoff initiation failed: ' + e.message);
            return null;
        }
    }

    /**
     * Commit handoff — Device B becomes active.
     * Called by Device A after receiving ACK.
     */
    async function commitHandoff(sessionId) {
        try {
            _phase = 'COMMITTING';
            _updateStatus('syncing', 'Committing handoff...');

            const res = await fetch(`${AI_API}/ai/handoff/commit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId, deviceId: _deviceId }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _phase = 'COMMITTED';
            _stopStatusPolling();
            _updateStatus('success',
                `Handoff complete (${data.handoff_duration_ms}ms). Device B is now active.`);
            return data;
        } catch (e) {
            _updateStatus('error', 'Commit failed: ' + e.message);
            return null;
        }
    }

    // ══════════════════════════════════════════
    //  DEVICE B — Target Device
    // ══════════════════════════════════════════

    /**
     * Join using a pair code.
     * @param {string} pairCode
     * @returns {Promise<{session_id, handoff_token}|null>}
     */
    async function joinWithCode(pairCode) {
        try {
            const res = await fetch(`${AI_API}/ai/handoff/join`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pairCode, deviceId: _deviceId }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _handoffToken = data.handoff_token;
            _targetSessionId = data.session_id;
            _role = 'device_b';
            _phase = 'PAIRED';
            _updateStatus('paired', 'Paired! Waiting for session transfer...');

            // Poll for snapshot availability
            _startSnapshotPolling();
            return data;
        } catch (e) {
            _updateStatus('error', 'Join failed: ' + e.message);
            return null;
        }
    }

    /**
     * Retrieve the session snapshot from the server.
     */
    async function fetchSnapshot() {
        if (!_targetSessionId || !_handoffToken) return null;
        try {
            const res = await fetch(`${AI_API}/ai/handoff/snapshot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sessionId: _targetSessionId,
                    handoffToken: _handoffToken,
                }),
            });
            const data = await res.json();
            if (data.error) return null;

            _snapshot = data.snapshot;
            _phase = 'SNAPSHOT_RECEIVED';
            _updateStatus('syncing', 'Snapshot received — validating...');
            return _snapshot;
        } catch (e) {
            return null;
        }
    }

    /**
     * Validate snapshot and send acknowledgement.
     */
    async function acknowledgeSnapshot() {
        if (!_snapshot) { _updateStatus('error', 'No snapshot to acknowledge'); return null; }

        // Verify checksum locally
        const checksum = _computeChecksum(_snapshot);
        if (checksum !== _snapshot.checksum) {
            _updateStatus('error', 'Snapshot integrity check failed');
            return null;
        }

        try {
            const res = await fetch(`${AI_API}/ai/handoff/acknowledge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sessionId: _targetSessionId,
                    handoffToken: _handoffToken,
                    deviceId: _deviceId,
                    checksum: checksum,
                }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _phase = 'ACK_SENT';
            _updateStatus('syncing', 'Acknowledged — waiting for commit...');

            // Poll for commit
            _startCommitPolling();
            return data;
        } catch (e) {
            _updateStatus('error', 'Acknowledge failed: ' + e.message);
            return null;
        }
    }

    /**
     * Restore the session from snapshot on Device B.
     * After this, Device B can continue recording.
     */
    async function restoreSession() {
        if (!_snapshot) { _updateStatus('error', 'No snapshot to restore'); return null; }

        try {
            _updateStatus('syncing', 'Restoring session...');
            const res = await fetch(`${AI_API}/ai/handoff/restore`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    snapshot: _snapshot,
                    deviceId: _deviceId,
                }),
            });
            const data = await res.json();
            if (data.error) { _updateStatus('error', data.error); return null; }

            _phase = 'RESTORED';
            _stopStatusPolling();
            _updateStatus('success', 'Session restored! You can continue recording.');
            return data;
        } catch (e) {
            _updateStatus('error', 'Restore failed: ' + e.message);
            return null;
        }
    }

    // ══════════════════════════════════════════
    //  ABORT
    // ══════════════════════════════════════════

    async function abort(reason) {
        _stopStatusPolling();
        if (_targetSessionId) {
            try {
                await fetch(`${AI_API}/ai/handoff/abort`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sessionId: _targetSessionId, reason }),
                });
            } catch (e) { /* best effort */ }
        }
        _reset();
        _updateStatus('idle', 'Handoff aborted');
    }

    // ══════════════════════════════════════════
    //  POLLING
    // ══════════════════════════════════════════

    function _startStatusPolling(sessionId) {
        _stopStatusPolling();
        _pollTimer = setInterval(async () => {
            try {
                const res = await fetch(`${AI_API}/ai/handoff/status/${sessionId}`);
                const status = await res.json();

                if (status.phase === 'PAIRED' && _role === 'device_a' && _phase === 'PAIR_REQUESTED') {
                    _phase = 'PAIRED';
                    _updateStatus('paired', 'Device B connected! Ready to transfer.');
                }
                if (status.phase === 'ACK_RECEIVED' && _role === 'device_a') {
                    _phase = 'ACK_RECEIVED';
                    _updateStatus('syncing', 'Device B acknowledged — committing...');
                    // Auto-commit
                    const result = await commitHandoff(sessionId);
                    if (result && result.success) {
                        _stopStatusPolling();
                    }
                }
                if (status.phase === 'FAILED') {
                    _phase = 'FAILED';
                    _stopStatusPolling();
                    _updateStatus('error', status.failure_reason || 'Handoff failed');
                }
            } catch (e) { /* ignore poll errors */ }
        }, 1500);
    }

    function _startSnapshotPolling() {
        _stopStatusPolling();
        _pollTimer = setInterval(async () => {
            const snap = await fetchSnapshot();
            if (snap) {
                _stopStatusPolling();
                // Auto-acknowledge and wait for commit
                await acknowledgeSnapshot();
            }
        }, 1500);
    }

    function _startCommitPolling() {
        _stopStatusPolling();
        _pollTimer = setInterval(async () => {
            try {
                const res = await fetch(`${AI_API}/ai/handoff/status/${_targetSessionId}`);
                const status = await res.json();
                if (status.phase === 'COMMITTED') {
                    _stopStatusPolling();
                    _phase = 'COMMITTED';
                    _updateStatus('syncing', 'Handoff committed — restoring session...');
                    await restoreSession();
                }
                if (status.phase === 'FAILED') {
                    _stopStatusPolling();
                    _phase = 'FAILED';
                    _updateStatus('error', status.failure_reason || 'Handoff failed');
                }
            } catch (e) { /* ignore */ }
        }, 1500);
    }

    function _stopStatusPolling() {
        if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    }

    // ══════════════════════════════════════════
    //  HELPERS
    // ══════════════════════════════════════════

    function _computeChecksum(snapshot) {
        const parts = [
            snapshot.session_id,
            String(snapshot.sequence_number),
            String((snapshot.transcript || []).length),
            String((snapshot.timeline_events || []).length),
            String(snapshot.chunks_processed),
            String(snapshot.handoff_epoch),
        ];
        // SHA-256 in browser
        const raw = parts.join('|');
        // Use a simple hash since crypto.subtle is async
        let hash = 0;
        for (let i = 0; i < raw.length; i++) {
            const chr = raw.charCodeAt(i);
            hash = ((hash << 5) - hash) + chr;
            hash |= 0;
        }
        // Match Python's hashlib output format — need actual SHA-256
        return _sha256Sync(raw).slice(0, 16);
    }

    function _sha256Sync(str) {
        // Synchronous SHA-256 using a basic implementation
        // For production, use crypto.subtle — here we use a compact JS SHA-256
        const utf8 = new TextEncoder().encode(str);
        // Inline SHA-256 (minimal implementation)
        const K = [
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
        ];
        let h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,
            h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
        const l = utf8.length;
        const bits = l * 8;
        // Padding
        const padded = new Uint8Array(Math.ceil((l + 9) / 64) * 64);
        padded.set(utf8);
        padded[l] = 0x80;
        const dv = new DataView(padded.buffer);
        dv.setUint32(padded.length - 4, bits, false);
        // Process blocks
        const w = new Int32Array(64);
        for (let off = 0; off < padded.length; off += 64) {
            for (let i = 0; i < 16; i++) w[i] = dv.getInt32(off + i * 4, false);
            for (let i = 16; i < 64; i++) {
                const s0 = (ror(w[i-15],7))^(ror(w[i-15],18))^(w[i-15]>>>3);
                const s1 = (ror(w[i-2],17))^(ror(w[i-2],19))^(w[i-2]>>>10);
                w[i] = (w[i-16]+s0+w[i-7]+s1)|0;
            }
            let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
            for (let i=0;i<64;i++){
                const S1=(ror(e,6))^(ror(e,11))^(ror(e,25));
                const ch=(e&f)^((~e)&g);
                const t1=(h+S1+ch+K[i]+w[i])|0;
                const S0=(ror(a,2))^(ror(a,13))^(ror(a,22));
                const maj=(a&b)^(a&c)^(b&c);
                const t2=(S0+maj)|0;
                h=g;g=f;f=e;e=(d+t1)|0;d=c;c=b;b=a;a=(t1+t2)|0;
            }
            h0=(h0+a)|0;h1=(h1+b)|0;h2=(h2+c)|0;h3=(h3+d)|0;
            h4=(h4+e)|0;h5=(h5+f)|0;h6=(h6+g)|0;h7=(h7+h)|0;
        }
        return [h0,h1,h2,h3,h4,h5,h6,h7].map(v=>(v>>>0).toString(16).padStart(8,'0')).join('');
    }
    function ror(n,b){return(n>>>b)|(n<<(32-b));}

    function _reset() {
        _pairCode = null;
        _handoffToken = null;
        _targetSessionId = null;
        _role = null;
        _phase = 'IDLE';
        _snapshot = null;
        _stopStatusPolling();
    }

    function _updateStatus(state, message) {
        console.log(`[Handoff] ${state}: ${message}`);
        // Update UI
        const el = document.getElementById('handoffStatus');
        if (el) {
            el.textContent = message;
            el.className = 'handoff-status handoff-' + state;
        }
        const phaseEl = document.getElementById('handoffPhase');
        if (phaseEl) phaseEl.textContent = _phase;

        // Show/hide appropriate buttons
        _updateUI();

        if (_statusCallback) _statusCallback(state, message, _phase);
    }

    function _updateUI() {
        const pairBtn = document.getElementById('handoffPairBtn');
        const joinArea = document.getElementById('handoffJoinArea');
        const transferBtn = document.getElementById('handoffTransferBtn');
        const abortBtn = document.getElementById('handoffAbortBtn');
        const codeDisplay = document.getElementById('handoffCodeDisplay');

        if (pairBtn) pairBtn.style.display = _phase === 'IDLE' ? '' : 'none';
        if (joinArea) joinArea.style.display = _phase === 'IDLE' ? '' : 'none';
        if (abortBtn) abortBtn.style.display =
            (_phase !== 'IDLE' && _phase !== 'COMMITTED' && _phase !== 'RESTORED') ? '' : 'none';

        if (transferBtn) {
            transferBtn.style.display = (_phase === 'PAIRED' && _role === 'device_a') ? '' : 'none';
        }
        if (codeDisplay) {
            codeDisplay.style.display = _pairCode && _role === 'device_a' ? '' : 'none';
            codeDisplay.textContent = _pairCode || '';
        }
    }

    // ══════════════════════════════════════════
    //  UI INJECTION
    // ══════════════════════════════════════════

    function injectUI() {
        if (document.getElementById('handoffPanel')) return;

        const panel = document.createElement('div');
        panel.id = 'handoffPanel';
        panel.className = 'handoff-panel';
        panel.innerHTML = `
            <div class="handoff-header">
                <span><i class="fa fa-exchange-alt"></i> Session Handoff</span>
                <button class="handoff-close-btn" onclick="document.getElementById('handoffPanel').classList.remove('visible')">&times;</button>
            </div>
            <div class="handoff-body">
                <div id="handoffStatus" class="handoff-status handoff-idle">Ready</div>
                <div id="handoffPhase" class="handoff-phase">IDLE</div>
                <div id="handoffCodeDisplay" class="handoff-code" style="display:none"></div>

                <button id="handoffPairBtn" class="handoff-btn handoff-btn-primary"
                    onclick="SessionHandoff.pair()">
                    <i class="fa fa-link"></i> Generate Pair Code
                </button>

                <div id="handoffJoinArea" class="handoff-join-area">
                    <input id="handoffCodeInput" class="handoff-input"
                        placeholder="Enter pair code" maxlength="6" pattern="[0-9]{6}">
                    <button class="handoff-btn handoff-btn-secondary"
                        onclick="SessionHandoff.join()">
                        <i class="fa fa-sign-in-alt"></i> Join
                    </button>
                </div>

                <button id="handoffTransferBtn" class="handoff-btn handoff-btn-primary" style="display:none"
                    onclick="SessionHandoff.transfer()">
                    <i class="fa fa-paper-plane"></i> Transfer Session
                </button>

                <button id="handoffAbortBtn" class="handoff-btn handoff-btn-danger" style="display:none"
                    onclick="SessionHandoff.abort('User cancelled')">
                    <i class="fa fa-times"></i> Cancel Handoff
                </button>
            </div>
        `;
        document.body.appendChild(panel);

        // Inject styles
        if (!document.getElementById('handoffStyles')) {
            const style = document.createElement('style');
            style.id = 'handoffStyles';
            style.textContent = `
                .handoff-panel {
                    position: fixed; right: 16px; bottom: 16px; width: 320px;
                    background: var(--bg-2, #1e1e2e); border: 1px solid var(--border, #333);
                    border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                    z-index: 9100; display: none; font-family: inherit;
                    overflow: hidden;
                }
                .handoff-panel.visible { display: block; }
                .handoff-header {
                    display: flex; justify-content: space-between; align-items: center;
                    padding: 12px 16px; background: var(--bg-3, #252536);
                    border-bottom: 1px solid var(--border, #333);
                    font-weight: 600; font-size: 14px; color: var(--text-1, #e0e0e0);
                }
                .handoff-close-btn {
                    background: none; border: none; color: var(--text-2, #888);
                    font-size: 20px; cursor: pointer; line-height: 1;
                }
                .handoff-body { padding: 16px; display: flex; flex-direction: column; gap: 10px; }
                .handoff-status {
                    padding: 8px 12px; border-radius: 8px; font-size: 13px;
                    text-align: center; font-weight: 500;
                }
                .handoff-idle { background: var(--bg-3, #252536); color: var(--text-2, #888); }
                .handoff-pairing { background: #1a365d; color: #63b3ed; }
                .handoff-paired { background: #1c4532; color: #68d391; }
                .handoff-syncing { background: #2a2060; color: #a78bfa; }
                .handoff-success { background: #1c4532; color: #68d391; }
                .handoff-error { background: #4a1d1d; color: #fc8181; }
                .handoff-phase { font-size: 11px; color: var(--text-3, #666); text-align: center; }
                .handoff-code {
                    font-size: 32px; font-weight: 700; text-align: center;
                    letter-spacing: 8px; color: #a78bfa; padding: 12px;
                    background: var(--bg-3, #252536); border-radius: 8px;
                    font-family: monospace;
                }
                .handoff-btn {
                    padding: 10px 16px; border: none; border-radius: 8px;
                    font-size: 13px; font-weight: 600; cursor: pointer;
                    display: flex; align-items: center; justify-content: center; gap: 8px;
                    transition: opacity 0.2s;
                }
                .handoff-btn:hover { opacity: 0.85; }
                .handoff-btn-primary { background: #a78bfa; color: #fff; }
                .handoff-btn-secondary { background: #4fd1ff; color: #111; }
                .handoff-btn-danger { background: #f87171; color: #fff; }
                .handoff-join-area { display: flex; gap: 8px; }
                .handoff-input {
                    flex: 1; padding: 10px 12px; border: 1px solid var(--border, #333);
                    border-radius: 8px; background: var(--bg-3, #252536);
                    color: var(--text-1, #e0e0e0); font-size: 16px;
                    letter-spacing: 4px; text-align: center; font-family: monospace;
                }
            `;
            document.head.appendChild(style);
        }
    }

    // ══════════════════════════════════════════
    //  PUBLIC API
    // ══════════════════════════════════════════

    function pair() {
        const sessionId = window.AIMeeting?.getSessionId();
        if (!sessionId) {
            _updateStatus('error', 'No active meeting session');
            return;
        }
        createPairCode(sessionId);
    }

    function join() {
        const input = document.getElementById('handoffCodeInput');
        const code = input?.value?.trim();
        if (!code || code.length !== 6) {
            _updateStatus('error', 'Enter a 6-digit pair code');
            return;
        }
        joinWithCode(code);
    }

    function transfer() {
        if (!_targetSessionId) return;
        initiateHandoff(_targetSessionId);
    }

    function showPanel() {
        injectUI();
        document.getElementById('handoffPanel')?.classList.add('visible');
    }

    function hidePanel() {
        document.getElementById('handoffPanel')?.classList.remove('visible');
    }

    return {
        // Core API
        createPairCode,
        joinWithCode,
        initiateHandoff,
        commitHandoff,
        fetchSnapshot,
        acknowledgeSnapshot,
        restoreSession,
        abort,

        // Convenience
        pair,
        join,
        transfer,
        showPanel,
        hidePanel,
        injectUI,

        // State
        getDeviceId: () => _deviceId,
        getPhase: () => _phase,
        getRole: () => _role,
        getSnapshot: () => _snapshot,
        getTargetSessionId: () => _targetSessionId,
        onStatus: (cb) => { _statusCallback = cb; },
    };
})();

// Auto-inject when loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => SessionHandoff.injectUI());
} else {
    SessionHandoff.injectUI();
}
