class WavWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = this.handleMessage.bind(this);
    this.sourceSampleRate = null;
    this.targetSampleRate = null;
    this.needsResampling = false;
    this.resampleBuffer = [];
    this.resampleRatio = 1.0;
    // LibSampleRate
    this.resampler = null;
    this.resamplerInitialized = false;
    this.libSampleRateAvailable = false;
  }

  process(inputs, outputs, parameters) {
    // convert to single channel but keep float32, post to main thread
    const input = inputs[0]; // Input is a 2D array: channels x samples
    if (input.length > 0) {
      const numChannels = input.length;
      const numSamples = input[0].length;

      // Create an array to hold mixed single channel data
      const mixedData = new Float32Array(numSamples);

      // Mix all channels to a single channel by averaging (or summing if preferred)
      for (let i = 0; i < numSamples; i++) {
        let sum = 0;
        for (let j = 0; j < numChannels; j++) {
          sum += input[j][i]; // Add the sample from each channel
        }
        mixedData[i] = sum / numChannels; // Average the samples from all channels
      }

      // Resample if needed (for Firefox compatibility)
      const finalData = this.resample(mixedData);

      // send to main thread
      this.port.postMessage(finalData);
    }

    // Return true to keep the processor alive
    return true;
  }

  initializeLibSampleRate() {}

  async handleMessage(event) {
    if (event.data.type === "init") {
      const data = event.data;
      this.sourceSampleRate = data.sourceSampleRate;
      this.targetSampleRate = data.targetSampleRate;
      this.needsResampling = data.sourceSampleRate !== data.targetSampleRate;
      this.resampleRatio = data.sourceSampleRate / data.targetSampleRate;

      // Check if LibSampleRate is available in the global scope
      if (typeof globalThis.LibSampleRate !== "undefined") {
        this.libSampleRateAvailable = true;
        console.log('Using LibSampleRate for resampling');
      } else {
        console.warn(
          "LibSampleRate not available, falling back to linear interpolation:",
        );
        this.libSampleRateAvailable = false;
      }

      // Initialize libsamplerate-js resampler if available and resampling is needed
      if (this.needsResampling && this.libSampleRateAvailable) {
        try {
          const { create, ConverterType } = globalThis.LibSampleRate;
          this.resampler = await create(
            1, // mono channel
            this.sourceSampleRate,
            this.targetSampleRate,
            { converterType: ConverterType.SRC_SINC_FASTEST },
          );
          this.resamplerInitialized = true;
          console.log("libsamplerate-js resampler initialized successfully");
        } catch (error) {
          console.warn(
            "Failed to initialize libsamplerate-js, falling back to linear interpolation:",
            error,
          );
          this.resamplerInitialized = false;
        }
      }
    }
  }

  resample(inputData) {
    if (!this.needsResampling) {
      return inputData;
    }

    // Handle empty input buffer.
    if (inputData.length === 0) {
      return inputData;
    }

    // Use libsamplerate-js if available and initialized
    if (this.resamplerInitialized && this.resampler) {
      try {
        console.log(inputData);
        return this.resampler.full(inputData);
      } catch (error) {
        console.warn(
          "libsamplerate-js resampling failed, falling back to linear interpolation:",
          error,
        );
        // Fall through to linear interpolation
      }
    }

    // Fallback: Linear interpolation (original implementation)
    return this.linearResample(inputData);
  }

  linearResample(inputData) {
    // Calculate the output buffer length.
    // This formula correctly maps the endpoints of the input and output buffers,
    // ensuring that the last sample of the output corresponds to the last sample of the input.
    //
    // It's derived from the relationship: (outputLength - 1) * resampleRatio ≈ (inputData.length - 1).
    // This prevents both reading past the input buffer during interpolation (when upsampling)
    // and dropping samples from the end of the buffer (when downsampling).
    const outputLength =
      Math.floor((inputData.length - 1) / this.resampleRatio) + 1;
    const outputData = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * this.resampleRatio;
      const sourceIndexInt = Math.floor(sourceIndex);
      const fraction = sourceIndex - sourceIndexInt;

      if (sourceIndexInt + 1 < inputData.length) {
        // Linear interpolation
        outputData[i] =
          inputData[sourceIndexInt] * (1 - fraction) +
          inputData[sourceIndexInt + 1] * fraction;
      } else {
        outputData[i] = inputData[sourceIndexInt];
      }
    }

    return outputData;
  }
}

registerProcessor("wav-worklet", WavWorklet);
