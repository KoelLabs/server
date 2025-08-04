const serverorigin = location.origin;
const serverhost = location.host;

export class FeedbackGiver {
  constructor(target, target_by_word, on_transcription, on_word_spoken) {
    this.target_transcript = target;
    this.target_by_word = target_by_word;
    this.target_timestamped = target; // Store the target timestamped data
    this.speech_timestamped = []; // Will be populated from transcription
    this.speech_transcript = [];
    this.on_transcription = on_transcription;
    this.word_phone_pairings = null; // Store the computed pairings

    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;
    this.store_audio_chunks = [];
    this.userAudioBuffer = null;
    this.stored_audio = null;
    this.audioInput = null;
    this.mediaStream = null;

    this.on_word_spoken = on_word_spoken;
    this.words = [];
    this.are_words_correct = [];
    for (const [word, _] of target_by_word) {
      this.words.push(word);
      this.are_words_correct.push(false);
    }
    this.next_word_ix = 0;
    this.recognition = null;

    // Add properties to accumulate transcription data
    this.latest_transcription_data = "";
  }

  #setTranscription(transcription) {
    try {
      console.log("transcription", transcription);

      // Handle empty or invalid input
      if (!transcription || transcription === "" || typeof transcription !== "string") {
        console.log("Empty or invalid transcription data, skipping");
        return;
      }

      const parsed = JSON.parse(transcription);
      this.speech_transcript = parsed.speech_transcript || [];
      this.speech_timestamped = parsed.speech_timestamps || []; // Store the timestamped speech data
      console.log("this.speech_transcript from setTranscription", this.speech_transcript);
      this.on_transcription(this.speech_transcript);
    } catch (error) {
      console.error("Failed to parse transcription JSON in #setTranscription:", error);
      console.error("Raw transcription data:", transcription);
    }
  }

  #combineAudioChunks() {
    if (!this.store_audio_chunks || this.store_audio_chunks.length === 0) {
      console.log("no audio chunks to merge");
      return null;
    }
    const totalLength = this.store_audio_chunks.reduce((sum, arr) => sum + arr.length, 0); // grabs all samples across all the chunks
    const merged = new Float32Array(totalLength); // allocate the float
    let chunkPosition = 0;
    for (const chunk of this.store_audio_chunks) {
      merged.set(chunk, chunkPosition);
      chunkPosition += chunk.length;
    }
    if (merged.length !== totalLength) {
      throw new Error("merged length does not match total length");
    }
    return merged;
  }

  async getWordPhonePairings() {
    try {
      console.log("getting word phone pairings");

      // Only make the request if we have valid speech data
      if (!this.speech_timestamped || this.speech_timestamped.length === 0) {
        console.log("No speech data available, skipping word phone pairings request");
        return [];
      }

      console.log("speech_timestamped", this.speech_timestamped);
      const res = await fetch(
        `${serverorigin}/pair_by_words?target_timestamped=${encodeURIComponent(JSON.stringify(this.target_timestamped))}&target_by_words=${encodeURIComponent(JSON.stringify(this.target_by_word))}&speech_timestamped=${encodeURIComponent(JSON.stringify(this.speech_timestamped))}`
      );
      const result = await res.json();
      console.log("ARUNA result from getWordPhonePairings", result);
      this.word_phone_pairings = result; // Store the result
      return result;
    } catch (error) {
      console.error("Error in getWordPhonePairings:", error);
      console.error(error.stack);
      return [];
    }
  }

  async computeWordPhonePairings() {
    // If we already have the pairings, return them
    if (this.word_phone_pairings) {
      console.log("word_phone_pairings already exists, returning them");
      return this.word_phone_pairings;
    }

    // Only compute if we have speech data
    if (!this.speech_timestamped || this.speech_timestamped.length === 0) {
      console.log("No speech data available for computing word phone pairings");
      return [];
    }

    // Otherwise compute them
    return await this.getWordPhonePairings();
  }

  clearWordPhonePairings() {
    this.word_phone_pairings = null;
  }

  async getCER() {
    try {
      // Ensure we have the word phone pairings
      if (!this.word_phone_pairings) {
        await this.computeWordPhonePairings();
      }

      const res = await fetch(
        `${serverorigin}/score_words_cer?word_phone_pairings=${encodeURIComponent(JSON.stringify(this.word_phone_pairings))}`
      );
      const data = await res.json();
      console.log("data", data);
      const [scoredWords, overall] = data;
      return [scoredWords, overall];
    } catch (error) {
      console.error("Error in getCER:", error);
      return [[], 0];
    }
  }

  async getWFED() {
    try {
      // Ensure we have the word phone pairings
      if (!this.word_phone_pairings) {
        await this.computeWordPhonePairings();
      }

      const res = await fetch(
        `${serverorigin}/score_words_wfed?word_phone_pairings=${encodeURIComponent(JSON.stringify(this.word_phone_pairings))}`
      );
      const data = await res.json();
      const [scoredWords, overall] = data;
      return [scoredWords, overall];
    } catch (error) {
      console.error("Error in getWFED:", error);
      return [[], 0];
    }
  }
  async getFeedback() {
    try {
      // Ensure we have the word phone pairings
      if (!this.word_phone_pairings) {
        await this.computeWordPhonePairings();
      }

      const res = await fetch(
        `${serverorigin}/user_phonetic_errors?word_phone_pairings=${encodeURIComponent(JSON.stringify(this.word_phone_pairings))}`
      );
      console.log("res", res);
      return await res.json();
    } catch (error) {
      console.error("Error in getFeedback:", error);
      return [];
    }
  }

  async getUserPhoneticErrors() {
    try {

      // Ensure we have the word phone pairings
      if (!this.word_phone_pairings) {
        await this.computeWordPhonePairings();
      }
      if (!this.word_phone_pairings || this.word_phone_pairings.length === 0) {
        console.log("No word phone pairings available");
        return {};
      }

      const res = await fetch(
        `${serverorigin}/user_phonetic_errors?word_phone_pairings=${encodeURIComponent(JSON.stringify(this.word_phone_pairings))}`
      );
      return await res.json();
    } catch (error) {
      console.error("Error in getUserPhoneticErrors:", error);
      return {};
    }
  }
  preparePlayback() {
    this.userAudioBuffer = null;
    console.log("buffer gets cleared: ", this.userAudioBuffer);

    this.userAudioBuffer = this.audioContext.createBuffer(1, this.stored_audio.length, 16000);
    this.userAudioBuffer.getChannelData(0).set(this.stored_audio);
    console.log("userAudioBuffer length from prep: ", this.userAudioBuffer.length);
  }
  async playUserAudio(onPlaybackEnd = () => { }) {
    const source = this.audioContext.createBufferSource();
    source.buffer = this.userAudioBuffer;
    console.log("userAudioBuffer length from playing: ", this.userAudioBuffer.length);
    source.connect(this.audioContext.destination);
    source.start();
    // close the audio context after we have nicely played back the users audio and wait for buffer to close
    source.onended = () => {
      source.suspend();
      onPlaybackEnd();
    };
  }
  async playTimestampedAudio(start_timestamp, end_timestamp) {
    if (!this.userAudioBuffer) {
      console.log("no user audio buffer to play");
      return;
    }


    const source = this.audioContext.createBufferSource();
    source.buffer = this.userAudioBuffer;
    source.connect(this.audioContext.destination);

    const offset = start_timestamp;              // where in the buffer to begin
    const duration = end_timestamp - start_timestamp;

    source.start(0, offset, duration);             // play immediately
    source.onended = () => source.disconnect();

    console.log(`played ${duration}s from ${offset}s`);
  }
  async #cleanupRecording() {
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

    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.stop();
      this.recognition = null;
    }

  }

  async start() {
    await this.#cleanupRecording();
    this.store_audio_chunks = [];

    // Clear previous transcription
    this.speech_transcript = [];
    this.speech_timestamped = [];
    this.stored_audio = null;
    this.clearWordPhonePairings(); // Clear cached pairings for new recording

    // Reset latest transcription
    this.latest_transcription_data = "";

    // Open WebSocket connection
    this.socket = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${serverhost}/stream`
    );

    // Handle incoming transcriptions
    this.socket.onmessage = async (event) => {
      // Accumulate transcription data instead of immediately processing it
      // We'll process the final transcription when recording stops
      console.log("event.data", event.data);

      // Always keep the most recent transcription snapshot
      this.latest_transcription_data = event.data;
    };

    // Start capturing audio (microphone)
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // Create an AudioContext for usage throughout
    // Create AudioContext if it doesn't exist or is closed
    if (!this.audioContext || this.audioContext.state === 'closed') {
      console.log("audioContext is closed or doesnt exist, creating new one");
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
        latencyHint: "interactive",
      });
    }
    if (this.audioContext.state === 'suspended') {
      console.log("audioContext is suspended, resuming");
      await this.audioContext.resume();
    }
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
    this.audioInput = this.audioContext.createMediaStreamSource(stream);
    this.audioInput.connect(this.audioWorkletNode);

    // Connect the AudioWorkletNode to the audio context destination
    this.audioWorkletNode.connect(this.audioContext.destination);

    // Connect AudioWorkletNode to process audio and send to WebSocket
    this.audioWorkletNode.port.onmessage = (event) => {
      const chunk_to_store = event.data;
      this.store_audio_chunks.push(chunk_to_store); // continuously store the audio chunks
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(chunk_to_store);
      }
    };

    this.#startWordTranscription();
  }

  async stop() {
    // Use the latest transcription snapshot we've received so far.
    if (this.latest_transcription_data) {
      this.#setTranscription(this.latest_transcription_data);
    } else {
      console.warn("No transcription data received from server before stop");
    }
    await this.#cleanupRecording();
    // merging logic of audio chunks when user stops recording
    if (this.store_audio_chunks.length > 0) {
      this.stored_audio = this.#combineAudioChunks();
      console.log("stored_audio! stored audio length: ", this.stored_audio.length);
    } else {
      console.log("no audio chunks to merge");
    }
    this.store_audio_chunks = [];
    if (this.stored_audio) {
      this.preparePlayback();
    } else {
      console.log("no audio to play back");
    }

    if (this.socket) {
      return new Promise((resolve) => {
        this.socket.onclose = resolve;
        this.socket.send("stop")
        this.socket = null;
      });
    }
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