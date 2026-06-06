from .base import render_template

_EXTRA_CSS = '''
.state-banner {
    padding: 10px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    text-align: center;
    margin-bottom: 10px;
}
.state-banner.lane { background: rgba(63, 185, 80, 0.12); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }
.state-banner.stop { background: rgba(248, 81, 73, 0.12); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); }
.reason { font-size: 12px; color: var(--text-secondary); margin-top: 6px; word-break: break-word; }
'''

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img src="{{ url_for('video') }}" class="stream" alt="Object Detection Stream">
        </div>
        <div class="controls-section">
            <div class="card">
                <div class="card-header">Robot State</div>
                <div id="state-banner" class="state-banner lane">LANE FOLLOWING</div>
                <div class="reason" id="reason">—</div>
            </div>

            <div class="card">
                <div class="card-header">Detection Stats</div>
                <div class="stats-grid">
                    <div class="stat-box"><div class="stat-value" id="kept">0</div><div class="stat-label">kept</div></div>
                    <div class="stat-box"><div class="stat-value" id="raw">0</div><div class="stat-label">raw</div></div>
                    <div class="stat-box"><div class="stat-value" id="latency">0</div><div class="stat-label">ms</div></div>
                    <div class="stat-box"><div class="stat-value" id="frames">0</div><div class="stat-label">det frames</div></div>
                </div>
                <div class="config-item">
                    <span class="config-label">Frames skipped</span>
                    <span class="config-value" id="skip">0</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">Control</div>
                <button class="button success" onclick="postJSON('/start', {})">Start</button>
                <button class="button danger" onclick="postJSON('/stop', {})">Stop</button>
                <button class="button" onclick="postJSON('/reset', {})">Reset</button>
            </div>
        </div>
    </div>
'''

_EXTRA_JS = '''
async function refresh() {
    try {
        const r = await fetch('/status');
        const s = await r.json();
        const banner = document.getElementById('state-banner');
        if (s.state === 'obstacle_present') {
            banner.className = 'state-banner stop';
            banner.textContent = 'OBSTACLE PRESENT';
        } else {
            banner.className = 'state-banner lane';
            banner.textContent = 'LANE FOLLOWING';
        }
        document.getElementById('reason').textContent  = s.stop_reason || '—';
        document.getElementById('kept').textContent    = s.kept_count;
        document.getElementById('raw').textContent     = s.raw_count;
        document.getElementById('latency').textContent = Math.round(s.detector_latency_ms);
        document.getElementById('frames').textContent  = s.frames_processed;
        document.getElementById('skip').textContent    = s.frames_skipped;
    } catch (e) {}
}
setInterval(refresh, 300);
refresh();
'''


OBJECT_DETECTION_TEMPLATE = render_template(
    title="Object Detection",
    subtitle="Lane following with obstacle stopping",
    content_html=_CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=_EXTRA_JS,
)
