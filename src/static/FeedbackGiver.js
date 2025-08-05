const SAMPLE_RATE = 16_000;

function combineAudioChunks(audio_chunks) {
  if (audio_chunks.length === 0) {
    return null;
  }

  const totalLength = audio_chunks.reduce((sum, arr) => sum + arr.length, 0);
  const merged = new Float32Array(totalLength);

  let chunkPosition = 0;
  for (const chunk of audio_chunks) {
    merged.set(chunk, chunkPosition);
    chunkPosition += chunk.length;
  }

  if (merged.length !== totalLength) {
    throw new Error('merged length does not match total length');
  }
  return merged;
}

export class FeedbackGiver {
  constructor(
    target,
    target_by_word,
    on_transcription,
    on_word_spoken,
    serverorigin = location.origin,
    serverhost = location.host,
  ) {
    this.serverorigin = serverorigin;
    this.serverhost = serverhost;

    this.target = target;
    this.target_by_word = target_by_word;
    this.target_timestamped = target;
    this.speech_timestamped = [];
    this.speech_transcript = [];
    this.on_transcription = on_transcription;
    this.word_phone_pairings = null;

    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;
    this.audioInput = null;
    this.mediaStream = null;

    this.userAudioBuffer = null;
    this.store_audio_chunks = [];

    this.on_word_spoken = on_word_spoken;
    this.words = [];
    this.are_words_correct = [];
    for (const [word, _] of target_by_word) {
      this.words.push(word);
      this.are_words_correct.push(false);
    }
    this.next_word_ix = 0;
    this.recognition = null;
  }

  #setTranscription(transcription) {
    const parsed = JSON.parse(transcription);
    this.speech_transcript = parsed.speech_transcript || [];
    this.speech_timestamped = parsed.speech_timestamps || [];
    this.on_transcription(this.speech_transcript);
  }

  async getWordPhonePairings() {
    // Only make the request if we have valid speech data
    if (!this.speech_timestamped || this.speech_timestamped.length === 0) {
      return [];
    }

    const res = await fetch(
      `${this.serverorigin}/pair_by_words?target_timestamped=${encodeURIComponent(JSON.stringify(this.target_timestamped))}&target_by_words=${encodeURIComponent(JSON.stringify(this.target_by_word))}&speech_timestamped=${encodeURIComponent(JSON.stringify(this.speech_timestamped))}`,
    );
    this.word_phone_pairings = await res.json();
    return this.word_phone_pairings;
  }

  async computeWordPhonePairings() {
    // If we already have the pairings, return them
    if (this.word_phone_pairings) {
      return this.word_phone_pairings;
    }

    // Only compute if we have speech data
    if (!this.speech_timestamped || this.speech_timestamped.length === 0) {
      return [];
    }

    // Otherwise compute them
    return await this.getWordPhonePairings();
  }

  /** @returns {[number[], number]} res[0] = scores for each word, res[1] = average score */
  async getCER() {
    const res = await fetch(
      `${this.serverorigin}/score_words_cer?word_phone_pairings=${encodeURIComponent(JSON.stringify(await this.computeWordPhonePairings()))}`,
    );
    return await res.json();
  }

  /** @returns {[number[], number]} res[0] = scores for each word, res[1] = average score */
  async getWFED() {
    const res = await fetch(
      `${this.serverorigin}/score_words_wfed?word_phone_pairings=${encodeURIComponent(JSON.stringify(await this.computeWordPhonePairings()))}`,
    );
    return await res.json();
  }

  async getFeedback() {
    const res = await fetch(
      `${this.serverorigin}/user_phonetic_errors?word_phone_pairings=${encodeURIComponent(JSON.stringify(await this.computeWordPhonePairings()))}`,
    );
    return await res.json();
  }

  #preparePlayback() {
    const audio = combineAudioChunks(this.store_audio_chunks);
    this.userAudioBuffer = this.audioContext.createBuffer(1, audio.length, SAMPLE_RATE);
    this.userAudioBuffer.getChannelData(0).set(audio);
  }

  async playUserAudio(start_timestamp = 0, end_timestamp = null) {
    const source = this.audioContext.createBufferSource();
    source.buffer = this.userAudioBuffer;
    source.connect(this.audioContext.destination);

    if (end_timestamp === null) {
      source.start(0, start_timestamp);
    } else {
      source.start(0, start_timestamp, end_timestamp - start_timestamp);
    }

    await new Promise(resolve => {
      source.onended = resolve;
    });
  }

  async #cleanupRecording() {
    // Stop processing audio
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
      this.audioWorkletNode = null;
    }

    // Stop and clean up audio input
    if (this.audioInput) {
      this.audioInput.disconnect();
      this.audioInput = null;
    }

    // Stop media stream tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => {
        track.stop();
      });
      this.mediaStream = null;
    }

    // Stop word recognition
    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.stop();
      this.recognition = null;
    }

    // Ask the server to close the websocket connection when it has finished processing all audio up until the "stop" message
    if (this.socket) {
      await new Promise(resolve => {
        this.socket.onclose = resolve;
        this.socket.send('stop');
        this.socket = null;
      });
    }
  }

  async start() {
    await this.#cleanupRecording();
    this.store_audio_chunks = [];
    this.userAudioBuffer = null;
    this.speech_transcript = [];
    this.speech_timestamped = [];
    this.word_phone_pairings = null;

    // Open WebSocket connection
    this.socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${this.serverhost}/stream`,
    );

    // Handle incoming transcriptions
    this.socket.onmessage = async event => {
      // Immediately process each transcription update so consumers can react (e.g. update word coloring)
      this.#setTranscription(event.data);
    };

    // Start capturing audio (microphone)
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // Create AudioContext if it doesn't exist or is closed
    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
        latencyHint: 'interactive',
      });
    }
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    // Load the AudioWorkletProcessor (which handles audio processing)
    await this.audioContext.audioWorklet.addModule(`${this.serverorigin}/WavWorklet.js`);
    this.audioWorkletNode = new AudioWorkletNode(this.audioContext, 'wav-worklet');

    // Connect the audio input to the AudioWorkletNode
    this.audioInput = this.audioContext.createMediaStreamSource(stream);
    this.audioInput.connect(this.audioWorkletNode);

    // Connect the AudioWorkletNode to the audio context destination
    this.audioWorkletNode.connect(this.audioContext.destination);

    // Connect AudioWorkletNode to process audio and send to WebSocket
    this.audioWorkletNode.port.onmessage = event => {
      const chunk = event.data;
      this.store_audio_chunks.push(chunk); // continuously store the audio chunks
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(chunk);
      }
    };

    this.#startWordTranscription();
  }

  async stop() {
    await this.#cleanupRecording();
    this.#preparePlayback();
  }

  #startWordTranscription() {
    this.next_word_ix = 0;
    for (let i = 0; i < this.words.length; i++) {
      this.are_words_correct[i] = false;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    this.recognition = new SpeechRecognition();
    this.recognition.lang = 'en-US';
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;
    this.recognition.start();
    this.recognition.onend = () => this.recognition.start();
    const finalWords = [];
    let numCorrect = 0;
    this.recognition.onresult = event => {
      const wordlist = [...event.results]
        .map(result => result[0].transcript.split(' '))
        .reduce((a, b) => a.concat(b))
        .filter(w => w.length > 0);
      const isFinal = [...event.results].at(-1).isFinal;
      if (isFinal) {
        finalWords.push(...wordlist);
        wordlist.length = 0;
      }

      const allWords = finalWords.concat(wordlist);
      if (
        allWords[0].toLowerCase().replace(/[^a-z]/g, '') !=
          this.words[0].toLowerCase().replace(/[^a-z]/g, '') &&
        allWords.length > 0
      ) {
        allWords.shift();
      }
      numCorrect = 0;
      for (let i = 0; i < allWords.length; i++) {
        const word = allWords[i];
        const target = this.words[i];

        if (
          word.toLowerCase().replace(/[^a-z]/g, '') === target.toLowerCase().replace(/[^a-z]/g, '')
        ) {
          this.are_words_correct[i] = true;
          numCorrect++;
        } else {
          this.are_words_correct[i] = false;
        }
        this.next_word_ix = i + 1;
        if (this.next_word_ix < this.words.length) {
          this.on_word_spoken(
            this.words,
            this.are_words_correct,
            this.next_word_ix,
            Math.round((1000 * numCorrect) / this.next_word_ix) / 10,
            false,
          );
        } else {
          this.recognition.onend = null;
          this.recognition.stop();
          this.on_word_spoken(
            this.words,
            this.are_words_correct,
            this.next_word_ix,
            Math.round((1000 * numCorrect) / this.words.length) / 10,
            true,
          );
        }
      }
    };
  }
}
