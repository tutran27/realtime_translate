class Pcm16Resampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetRate;
    this.chunkSamples = options.processorOptions.chunkSamples;
    this.ratio = sampleRate / this.targetRate;
    this.pendingInput = new Float32Array(0);
    this.readIndex = 0;
    this.output = [];
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") {
        this.flush();
      }
    };
  }

  emit(force) {
    while (
      this.output.length >= this.chunkSamples ||
      (force && this.output.length)
    ) {
      const size = force ? this.output.length : this.chunkSamples;
      const pcm = new Int16Array(size);
      for (let index = 0; index < size; index++) {
        const sample = Math.max(-1, Math.min(1, this.output[index]));
        pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.output.splice(0, size);
      this.port.postMessage(
        { type: "pcm", buffer: pcm.buffer },
        [pcm.buffer],
      );
      if (!force) {
        break;
      }
    }
  }

  flush() {
    this.emit(true);
    this.port.postMessage({ type: "flushed" });
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0].length) {
      return true;
    }

    const length = channels[0].length;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < channels.length; channel++) {
      const input = channels[channel];
      for (let index = 0; index < length; index++) {
        mono[index] += input[index] / channels.length;
      }
    }

    const combined = new Float32Array(
      this.pendingInput.length + mono.length,
    );
    combined.set(this.pendingInput);
    combined.set(mono, this.pendingInput.length);

    while (this.readIndex + 1 < combined.length) {
      const low = Math.floor(this.readIndex);
      const fraction = this.readIndex - low;
      this.output.push(
        combined[low] * (1 - fraction) +
          combined[low + 1] * fraction,
      );
      this.readIndex += this.ratio;
    }

    const consumed = Math.floor(this.readIndex);
    const retainFrom = Math.min(consumed, combined.length - 1);
    this.pendingInput = combined.slice(retainFrom);
    this.readIndex -= retainFrom;
    this.emit(false);
    return true;
  }
}

registerProcessor("pcm16-resampler", Pcm16Resampler);
