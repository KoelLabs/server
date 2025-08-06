/** @typedef {string} Phone */
/** @typedef {string} Word */
/** @typedef {[Phone, number, number]} TimestampedPhone */
/** @typedef {Array<[Word, TimestampedPhone[]]>} TimestampedPhonesByWord */
/** @typedef {Array<[TimestampedPhone, TimestampedPhone]>} TimestampedPhonePairings */

/**
 * @typedef {object} ExampleWord
 * @property {string} word the word with one or more sections surrounded by * that exemplify the phoneme, e.g., "l*aa*"
 * @property {string} phonetic_spelling an American English "spelling" of the word, e.g., "L-AA"
 */
/**
 * @typedef {object} PhoneDescription
 * @property {string} phoneme the IPA representation of the phoneme
 * @property {string[]} description e.g., ["front open unrounded vowel"]
 * @property {string} explanation a written explanation of how to pronounce this phoneme
 * @property {ExampleWord[]} examples a possibly empty list of example words with this phoneme
 * @property {string} phonetic_spelling an American English "spelling" of the phoneme
 * @property {string} video a source video that explains the phoneme
 */
/**
 * @typedef {object} Mistake represents a target phoneme being mispronounced as one of a collection of phonemes
 * @property {Phone} target the target phoneme being mispronounced ("-" if this is an insertion error)
 * @property {Phone[]} speech the phonemes it is mispronounced as ("-" for the deletion error)
 * @property {Word[]} words the words where this mispronunciation occurs
 * @property {Array<[Word, TimestampedPhonePairings, number[]]>} occurences_by_word for each word, a tuple of (word, paired target vs speech phonemes, severities)
 * @property {PhoneDescription | null} target_description description of the target phoneme (or null iff this is an insertion error)
 * @property {Array<PhoneDescription | null>} speech_description description(s) of the phonemes the target is mispronounced as (a null represent the deletion error)
 * @property {number} frequency count of occurences
 * @property {number} total_severity sum of severities (each between 0 and 1)
 */
/**
 * @typedef {object} Feedback
 * @property {Mistake[]} topk_mistakes_by_target
 * @property {Mistake[]} topk_insertion_mistakes
 * @property {Mistake[]} topk_deletion_mistakes
 * @property {Mistake[]} topk_substitution_mistakes
 * @property {Array<[Word, number, number]>} spoken_word_timestamps
 */
/** @typedef {[Array<[Word, number]>, number]} WordScores scores for each word and the average score */

/** @typedef {(transcription: TimestampedPhone[]) => void} TranscriptionCallback */
/** @typedef {(words: Word[], are_words_correct: boolean[], next_word_ix: number, percentage_correct: number, is_done: boolean) => void} WordSpokenCallback */

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
  /** @param {TimestampedPhonesByWord} target_by_word @param {TranscriptionCallback} on_transcription @param {WordSpokenCallback} on_word_spoken */
  constructor(
    target_by_word,
    on_transcription,
    on_word_spoken,
    serverorigin = location.origin,
    serverhost = location.host,
  ) {
    this.serverorigin = serverorigin;
    this.serverhost = serverhost;

    /** @type {TimestampedPhonesByWord} */
    this.target_by_word = target_by_word;
    /** @type {TimestampedPhone[]} */
    this.transcription = [];
    /** @type {TranscriptionCallback} */
    this.on_transcription = on_transcription;

    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;
    this.audioInput = null;
    this.mediaStream = null;

    this.userAudioBuffer = null;
    this.store_audio_chunks = [];

    /** @type {WordSpokenCallback} */
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

  /** @param {TimestampedPhone[]} transcription */
  #setTranscription(transcription) {
    this.transcription = transcription;
    this.on_transcription(transcription);
  }

  /** @returns {Promise<WordScores>} */
  async getCER() {
    if (this.transcription.length === 0) throw new Error('No transcription');
    const res = await fetch(
      `${this.serverorigin}/score_words_cer?target_by_words=${encodeURIComponent(JSON.stringify(this.target_by_word))}&speech=${encodeURIComponent(JSON.stringify(this.transcription))}`,
    );
    return await res.json();
  }

  /** @returns {Promise<Feedback>} */
  async getFeedback() {
    if (this.transcription.length === 0) throw new Error('No transcription');
    const res = await fetch(
      `${this.serverorigin}/top_phonetic_errors?target_by_words=${encodeURIComponent(JSON.stringify(this.target_by_word))}&speech=${encodeURIComponent(JSON.stringify(this.transcription))}`,
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
    this.transcription = [];

    // Open WebSocket connection
    this.socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${this.serverhost}/stream`,
    );

    // Handle incoming transcriptions
    this.socket.onmessage = async event => {
      // Immediately process each transcription update so consumers can react (e.g. update word coloring)
      this.#setTranscription(JSON.parse(event.data));
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
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.store_audio_chunks.push(chunk);
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
        if (!target) continue;

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
          if (this.recognition) {
            this.recognition.onend = null;
            this.recognition.stop();
          }
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
