export interface VadRegion {
  start: number;
  end: number;
}

/** Current /analyze response shape (existing backend) */
export interface AnalyzeResponseLegacy {
  duration_s: number;
  sample_rate: number;
  waveform: number[];
  mel_spectrogram: number[];
  mel_n_mels: number;
  mel_n_frames: number;
  vad_regions: VadRegion[];
}

/** Extended /analyze response (after backend update) */
export interface AnalyzeResponse extends AnalyzeResponseLegacy {
  stft_magnitude: number[];
  stft_n_frames: number;
  stft_n_freq_bins: number;
  rms_energy: number[];
  power_spectrum: number[];
}

export interface AnalyzerState {
  status: "idle" | "loading" | "ready" | "error";
  data: AnalyzeResponse | null;
  error: string | null;
  fileName: string | null;
}
