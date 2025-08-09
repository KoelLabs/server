class WavWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = this.handleMessage.bind(this);
    this.sourceSampleRate = null;
    this.targetSampleRate = null;
    this.needsResampling = false;
    this.resampleBuffer = [];
    this.resampleRatio = 1.0;
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

  handleMessage(event) {
    if (event.data.type === 'init') {
      this.sourceSampleRate = event.data.sourceSampleRate;
      this.targetSampleRate = event.data.targetSampleRate;
      this.needsResampling = this.sourceSampleRate !== this.targetSampleRate;
      this.resampleRatio = this.sourceSampleRate / this.targetSampleRate;
    }
  }

  resample(inputData) {
    if (!this.needsResampling) {
      return inputData;
    }

    const outputLength = Math.floor(inputData.length / this.resampleRatio);
    const outputData = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * this.resampleRatio;
      const sourceIndexInt = Math.floor(sourceIndex);
      const fraction = sourceIndex - sourceIndexInt;
      
      if (sourceIndexInt + 1 < inputData.length) {
        // Linear interpolation
        outputData[i] = inputData[sourceIndexInt] * (1 - fraction) + 
                       inputData[sourceIndexInt + 1] * fraction;
      } else {
        outputData[i] = inputData[sourceIndexInt];
      }
    }
    
    return outputData;
  }
}

registerProcessor('wav-worklet', WavWorklet);
