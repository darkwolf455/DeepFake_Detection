# VeriSense: System Limitations, Robustness Boundaries & Ethical Use

## 1. Probabilistic Nature of Deepfake Detection
Detection models output calibrated likelihoods rather than absolute mathematical proofs. VeriSense analyzes artifacts in spatial frequency distributions, temporal sequence continuity, voice acoustic timbre, and lip-audio synchronization. No automated classifier is 100% accurate.

## 2. Key Technical Limitations & Robustness Boundaries

### A. Generalization to Unseen Generators
- Models trained on specific diffusion models (e.g. Stable Diffusion 1.5/2.1, Midjourney v4) may experience performance drops when exposed to newer latent diffusion architectures, flux models, or proprietary autoencoders.
- Deepfake video models trained on FaceForensics++ (autoencoder-based face swaps) can have lower sensitivity against photorealistic 3D Gaussian Splatting or neural radiance fields (NeRFs) without retraining.

### B. Severe Social Media Compression & Re-encoding
- Platforms like WhatsApp, Twitter/X, and Instagram heavily downscale, blur, and apply lossy JPEG/H.264 compression with high CRF values.
- Lossy compression frequently destroys subtle high-frequency artifacts in facial textures and audio harmonics.
- Although VeriSense incorporates compression augmentations (random JPEG quality 30-100, blur, downsampling, codec simulation), heavily degraded media is flagged as `inconclusive`.

### C. Physical Audio Reverberation & Background Noise
- Audio branches trained on clean speech may suffer elevated false-positive rates when encountering heavy background music, cafeteria chatter, or acoustic room reflections.
- RawBoost and room impulse response simulations are integrated, but noisy audio should be interpreted cautiously.

### D. Adversarial Perturbations & Anti-Forensics
- Malicious actors can apply adversarial noise, patch attacks, or post-processing filters (e.g., subtle Gaussian grain, color grade shifts) designed to trick CLIP or Wav2Vec2 embeddings.

## 3. Ethical Guidance & Responsible Deployment
1. **Never Sole Evidence**: VeriSense verdicts should NEVER be used as the sole grounds for legal prosecution, employment termination, defamation allegations, or journalistic attribution without corroborating human forensic review.
2. **Mitigating False Accusations**: In legal and human rights contexts, a false positive (flagging authentic media as fake) can cause irreparable personal harm. The `inconclusive` confidence band is deliberately calibrated to mitigate overconfident false verdicts.
3. **Data Privacy**: The backend architecture is designed with immediate temporary file cleanup (`finally` block unlinks). Do not store or redistribute uploaded media without consent.
