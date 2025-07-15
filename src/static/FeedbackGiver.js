const serverorigin = location.origin;
const serverhost = location.host;

export class FeedbackGiver {
  constructor(target, target_by_word, on_transcription, on_word_spoken) {
    this.target = target;
    this.target_by_word = target_by_word;
    this.transcription = "";
    this.on_transcription = on_transcription;
    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;

    this.on_word_spoken = on_word_spoken;
    this.words = [];
    this.are_words_correct = [];
    for (const [word, _] of target_by_word) {
      this.words.push(word);
      this.are_words_correct.push(false);
    }
    this.next_word_ix = 0;
    this.recognition = null;

    // Collect Float32Array chunks coming from the AudioWorklet so that we can
    // send the full utterance to the backend /feedback endpoint after the
    // recording is finished.
    this._audioChunks = [];
  }

  #setTranscription(transcription) {
    this.transcription = transcription;
    this.on_transcription(this.transcription);
  }


  async getCER() {
    const res = await fetch(
      `${serverorigin}/score_words_cer?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [scoredWords, overall] = data;
    return [scoredWords, overall];
  }

  async getWFED() {
    const res = await fetch(
      `${serverorigin}/score_words_wfed?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [scoredWords, overall] = data;
    return [scoredWords, overall];
  }

  async getPhonemeNaturalLanguageFeedback() {
    const res = await fetch(
      `${serverorigin}/phoneme_written_feedback?target=${encodeURIComponent(this.target)}&speech=${encodeURIComponent(this.transcription)}`
    );
    return await res.json();
  }

  async getUserPhoneticErrors() {
    console.log("getUserPhoneticErrors called with transcription:", this.transcription);
    const res = await fetch(
      `${serverorigin}/user_phonetic_errors?target=${encodeURIComponent(this.target)}&tbw=${encodeURIComponent(JSON.stringify(this.target_by_word))}&speech=${encodeURIComponent(this.transcription)}`
    );
    console.log("res", res);
    return await res.json();
      
  }

  async start() {
    // Clear previous transcription
    this.#setTranscription("");

    // Open WebSocket connection
    this.socket = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${serverhost}/stream`
    );

    // Handle incoming transcriptions
    this.socket.onmessage = async (event) => {
      this.#setTranscription(event.data);
    };

    // Start capturing audio
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // Create an AudioContext
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
      latencyHint: "interactive",
    });

    // Load the AudioWorkletProcessor (which handles audio processing)
    await this.audioContext.audioWorklet.addModule(
      `${serverorigin}/WavWorklet.js`
    );

    // Create the AudioWorkletNode
    this.audioWorkletNode = new AudioWorkletNode(
      this.audioContext,
      "wav-worklet"
    );

    // Connect the audio input to the AudioWorkletNode
    const audioInput = this.audioContext.createMediaStreamSource(stream);
    audioInput.connect(this.audioWorkletNode);
    console.log("Audio input connected to AudioWorkletNode");

    // Connect the AudioWorkletNode to the audio context destination
    this.audioWorkletNode.connect(this.audioContext.destination);
    console.log("AudioWorkletNode connected to destination");

    // Connect AudioWorkletNode to process audio and send to WebSocket
    this.audioWorkletNode.port.onmessage = (event) => {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(event.data);
      }
    };

    this.#startWordTranscription();
  }

  async stop() {
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
    }
    if (this.socket) {
      this.socket.close();
    }
    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.stop();
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    // Optionally, you can await this.feedback() here or call it externally.
  }

  // Concatenate the recorded Float32Array chunks into a single Float32Array
  _getFullAudio() {
    const totalLength = this._audioChunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this._audioChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  /**
   * Upload the full recording to `/feedback` and return the JSON response.
   * Call this after `stop()` finishes so that recording is complete.
   */
  async feedback() {
    // Assemble audio
    const audioFloat32 = this._getFullAudio();

    // Prepare multipart/form-data payload
    const formData = new FormData();
    formData.append("target", this.target.join(""));
    formData.append("tbw", JSON.stringify(this.target_by_word));
    formData.append(
      "audio",
      new Blob([audioFloat32.buffer], { type: "application/octet-stream" }),
      "audio.raw"
    );

    const res = await fetch(`${serverorigin}/feedback`, {
      method: "POST",
      body: formData,
    });

    return await res.json();
  }

  #startWordTranscription() {
    this.next_word_ix = 0;
    for (let i = 0; i < this.words.length; i++) {
      this.are_words_correct[i] = false;
    }

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    this.recognition = new SpeechRecognition();
    this.recognition.lang = "en-US";
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;
    this.recognition.start();
    this.recognition.onend = () => this.recognition.start();
    const finalWords = [];
    let numCorrect = 0;
    this.recognition.onresult = (event) => {
      const wordlist = [...event.results]
        .map((result) => result[0].transcript.split(" "))
        .reduce((a, b) => a.concat(b))
        .filter((w) => w.length > 0);
      const isFinal = [...event.results].at(-1).isFinal;
      if (isFinal) {
        finalWords.push(...wordlist);
        wordlist.length = 0;
      }

      const allWords = finalWords.concat(wordlist);
      if (
        allWords[0].toLowerCase().replace(/[^a-z]/g, "") !=
          this.words[0].toLowerCase().replace(/[^a-z]/g, "") &&
        allWords.length > 0
      ) {
        allWords.shift();
      }
      numCorrect = 0;
      for (let i = 0; i < allWords.length; i++) {
        const word = allWords[i];
        const target = this.words[i];

        if (
          word.toLowerCase().replace(/[^a-z]/g, "") ===
          target.toLowerCase().replace(/[^a-z]/g, "")
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
            false
          );
        } else {
          this.recognition.onend = null;
          this.recognition.stop();
          this.on_word_spoken(
            this.words,
            this.are_words_correct,
            this.next_word_ix,
            Math.round((1000 * numCorrect) / this.words.length) / 10,
            true
          );
        }
      }
    };
  }
}
