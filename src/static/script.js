const SAMPLE_RATE = 16_000;

// prettier-ignore
const target_by_words = [
  ['you', [['j', 0.24160443037974685, 0.2617381329113924], ['u', 0.2617381329113924, 0.281871835443038]]],
  ['gotta', [['ɡ', 0.34227294303797473, 0.36240664556962027], ['ɑ', 0.38254034810126586, 0.40267405063291145], ['t', 0.4429414556962025, 0.4630751582278481], ['ʌ', 0.4832088607594937, 0.5033425632911394]]],
  ['stay', [['s', 0.6845458860759495, 0.7046795886075949], ['t', 0.7650806962025317, 0.7852143987341773], ['eɪ', 0.8053481012658229, 0.8254818037974684]]],
  ['alert', [['æ', 1.0872199367088609, 1.1073536392405066], ['l', 1.369091772151899, 1.3892254746835444], ['ɜ˞', 1.4093591772151899, 1.4294928797468356], ['t', 1.6106962025316458, 1.630829905063291]]],
  ['all', [['ɔ', 2.0536376582278484, 2.073771360759494], ['l', 2.1945735759493674, 2.214707278481013]]],
  ['the', [['ð', 2.27510838607595, 2.295242088607595], ['ʌ', 2.295242088607595, 2.3153757911392407]]],
  ['time', [['t', 2.416044303797469, 2.436178006329114], ['aɪ', 2.4764454113924055, 2.4965791139240507], ['m', 2.7985846518987345, 2.8187183544303798]]],
];

const wordsEl = document.getElementById('scored_words');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const playButton = document.getElementById('play-button');
const statusEl = document.getElementById('status');
const scoreEl = document.getElementById('score');
const feedbackAreaEl = document.getElementById('feedback-area');

let audioContext = null;
let audioInput = null;
let audioWorkletNode = null;
let mediaStream = null;
let userAudioBuffer = null;
let storedAudioChunks = [];

function setStatus(message) {
  statusEl.textContent = message;
}

function renderWordPlaceholders() {
  wordsEl.innerHTML = target_by_words
    .map(
      ([word]) => `
        <button
          type="button"
          class="word-pill"
          data-word="${word}"
          disabled
          style="margin-right: 8px; margin-bottom: 8px; padding: 10px 14px; border-radius: 999px; border: 1px solid #d4d4d8; background: #fff;"
        >
          ${word}
        </button>
      `,
    )
    .join('');
}

function combineAudioChunks(audioChunks) {
  if (audioChunks.length === 0) {
    return null;
  }

  const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;

  for (const chunk of audioChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function createAudioBuffer(audio) {
  if (!audioContext || !audio) {
    return null;
  }

  const buffer = audioContext.createBuffer(1, audio.length, SAMPLE_RATE);
  buffer.getChannelData(0).set(audio);
  return buffer;
}

function writeAsciiString(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function encodeWav(audio) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + audio.length * bytesPerSample);
  const view = new DataView(buffer);

  writeAsciiString(view, 0, 'RIFF');
  view.setUint32(4, 36 + audio.length * bytesPerSample, true);
  writeAsciiString(view, 8, 'WAVE');
  writeAsciiString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAsciiString(view, 36, 'data');
  view.setUint32(40, audio.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < audio.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, audio[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

async function playAudio(start = 0, end = null) {
  if (!audioContext || !userAudioBuffer) {
    return;
  }

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  const source = audioContext.createBufferSource();
  source.buffer = userAudioBuffer;
  source.connect(audioContext.destination);

  if (end === null) {
    source.start(0, start);
  } else {
    source.start(0, start, Math.max(0, end - start));
  }

  await new Promise((resolve) => {
    source.onended = resolve;
  });
}

async function cleanupRecording() {
  if (audioWorkletNode) {
    audioWorkletNode.disconnect();
    audioWorkletNode = null;
  }

  if (audioInput) {
    audioInput.disconnect();
    audioInput = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

async function startRecording() {
  await cleanupRecording();
  storedAudioChunks = [];
  userAudioBuffer = null;
  feedbackAreaEl.innerHTML = '';
  scoreEl.textContent = '';
  playButton.disabled = true;
  renderWordPlaceholders();
  setStatus('Requesting microphone access...');

  let usingLocalResampler = false;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE },
  });

  if (!audioContext || audioContext.state === 'closed') {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
        latencyHint: 'interactive',
      });
      audioInput = audioContext.createMediaStreamSource(mediaStream);
    } catch (error) {
      usingLocalResampler = true;
      if (audioContext) {
        await audioContext.close();
      }
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        latencyHint: 'interactive',
      });
      audioInput = audioContext.createMediaStreamSource(mediaStream);
    }
  } else {
    audioInput = audioContext.createMediaStreamSource(mediaStream);
  }

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  await audioContext.audioWorklet.addModule('./WavWorklet.js');
  if (usingLocalResampler) {
    await audioContext.audioWorklet.addModule('./libsamplerate.worklet.js');
  }

  audioWorkletNode = new AudioWorkletNode(audioContext, 'wav-worklet');
  audioWorkletNode.port.postMessage({
    type: 'init',
    sourceSampleRate: audioContext.sampleRate,
    targetSampleRate: SAMPLE_RATE,
  });
  audioInput.connect(audioWorkletNode);
  audioWorkletNode.connect(audioContext.destination);
  audioWorkletNode.port.onmessage = (event) => {
    storedAudioChunks.push(event.data);
  };

  setStatus('Recording...');
}

function wordColor(score) {
  if (score > 0.8) return '#10b981';
  if (score > 0.5) return '#86efac';
  if (score > 0.3) return '#fde68a';
  return '#fecaca';
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderAnalysis(result) {
  scoreEl.textContent = Math.round(result.average_score * 100);

  wordsEl.innerHTML = result.words
    .map(
      (word, index) => `
        <button
          type="button"
          data-word-index="${index}"
          style="margin-right: 8px; margin-bottom: 8px; padding: 10px 14px; border-radius: 999px; border: 1px solid #d4d4d8; background: ${wordColor(word.score)};"
        >
          ${escapeHtml(word.text)}
        </button>
      `,
    )
    .join('');

  document.querySelectorAll('[data-word-index]').forEach((button) => {
    button.addEventListener('click', async () => {
      const { start, end } = result.words[Number(button.dataset.wordIndex)];
      const originalText = button.textContent;
      button.disabled = true;
      button.textContent = 'Playing...';
      await playAudio(start, end);
      button.textContent = originalText;
      button.disabled = false;
    });
  });

  if (result.feedback.length === 0) {
    feedbackAreaEl.innerHTML = '<p>No major pronunciation issues found.</p>';
    return;
  }

  feedbackAreaEl.innerHTML = `
    <h3>Feedback</h3>
    <div>
      ${result.feedback
        .map(
          (item) => `
            <section style="margin-bottom: 20px; padding: 16px; border: 1px solid #e5e7eb; border-radius: 16px;">
              <h4 style="margin-top: 0;">${escapeHtml(item.caption)}</h4>
              <p>${escapeHtml(item.details)}</p>
              <p><strong>Relevant words:</strong> ${item.words.map(escapeHtml).join(', ')}</p>
              ${
                item.video
                  ? `<p><a href="${item.video}" target="_blank" rel="noreferrer">Open video explanation</a></p>`
                  : ''
              }
            </section>
          `,
        )
        .join('')}
    </div>
  `;
}

async function analyzeRecording() {
  const audio = combineAudioChunks(storedAudioChunks);
  if (!audio || audio.length === 0) {
    setStatus('No audio captured.');
    return;
  }

  userAudioBuffer = createAudioBuffer(audio);
  playButton.disabled = false;

  const formData = new FormData();
  formData.append('file', encodeWav(audio), 'recording.wav');

  setStatus('Uploading for analysis...');
  const response = await fetch(
    `/analyze_file?target_by_words=${encodeURIComponent(JSON.stringify(target_by_words))}&topk=5`,
    {
      method: 'POST',
      body: formData,
    },
  );

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error);
  }

  const result = await response.json();
  renderAnalysis(result);
  setStatus('Analysis complete.');
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  stopButton.disabled = false;

  try {
    await startRecording();
  } catch (error) {
    setStatus(`Unable to start recording: ${error.message}`);
    startButton.disabled = false;
    stopButton.disabled = true;
  }
});

stopButton.addEventListener('click', async () => {
  stopButton.disabled = true;
  startButton.disabled = false;

  try {
    setStatus('Finishing recording...');
    await cleanupRecording();
    await analyzeRecording();
  } catch (error) {
    setStatus(`Analysis failed: ${error.message}`);
  }
});

playButton.addEventListener('click', async () => {
  const originalText = playButton.textContent;
  playButton.disabled = true;
  playButton.textContent = 'Playing...';
  await playAudio();
  playButton.textContent = originalText;
  playButton.disabled = false;
});

renderWordPlaceholders();
