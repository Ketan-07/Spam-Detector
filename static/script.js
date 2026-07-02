const messageInput = document.getElementById('messageInput');
const charCount = document.getElementById('charCount');
const sortBtn = document.getElementById('sortBtn');
const trayHam = document.getElementById('trayHam');
const traySpam = document.getElementById('traySpam');
const envelope = document.getElementById('envelope');
const resultPanel = document.getElementById('resultPanel');
const resultVerdict = document.getElementById('resultVerdict');
const resultDetail = document.getElementById('resultDetail');
const stampMark = document.getElementById('stampMark');
const errorText = document.getElementById('errorText');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const API_BASE = ''; // same-origin; change if the API is hosted separately

function updateCharCount() {
  charCount.textContent = `${messageInput.value.length} / 5000`;
}
messageInput.addEventListener('input', updateCharCount);
updateCharCount();

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    messageInput.value = chip.dataset.sample;
    updateCharCount();
    messageInput.focus();
  });
});

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error();
    statusDot.className = 'status-dot ok';
    statusText.textContent = 'desk online';
  } catch {
    statusDot.className = 'status-dot err';
    statusText.textContent = 'desk offline';
  }
}
checkHealth();

function resetTrays() {
  trayHam.classList.remove('active');
  traySpam.classList.remove('active');
  envelope.classList.remove('moving', 'to-ham', 'to-spam');
}

async function sortMessage() {
  const message = messageInput.value.trim();
  errorText.hidden = true;
  resultPanel.hidden = true;

  if (!message) {
    errorText.textContent = 'Write or paste a message first.';
    errorText.hidden = false;
    return;
  }

  sortBtn.disabled = true;
  sortBtn.querySelector('.sort-btn-label').textContent = 'Sorting…';
  resetTrays();

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const isSpam = data.is_spam;

    // animate envelope + tray
    void envelope.offsetWidth; // restart animation
    envelope.classList.add('moving', isSpam ? 'to-spam' : 'to-ham');
    setTimeout(() => {
      (isSpam ? traySpam : trayHam).classList.add('active');
    }, 700);

    stampMark.className = `stamp ${isSpam ? 'spam' : 'ham'}`;
    stampMark.textContent = isSpam ? 'SPAM' : 'HAM';
    resultVerdict.textContent = isSpam
      ? 'Flagged as spam'
      : 'Looks legitimate';
    resultDetail.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    setTimeout(() => {
      resultPanel.hidden = false;
    }, 300);
  } catch (err) {
    errorText.textContent = err.message || 'Something went wrong contacting the API.';
    errorText.hidden = false;
  } finally {
    sortBtn.disabled = false;
    sortBtn.querySelector('.sort-btn-label').textContent = 'Sort message';
  }
}

sortBtn.addEventListener('click', sortMessage);
messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) sortMessage();
});
