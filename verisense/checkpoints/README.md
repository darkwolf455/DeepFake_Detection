# VeriSense Checkpoint Directory

Store trained PyTorch model weights here. The backend and evaluation pipelines look for these filenames by default:

- `image_best.pt` : CLIP ViT-L/14 visual classifier head (and optional LoRA adapters)
- `video_best.pt` : Temporal Transformer / BiLSTM sequence model
- `audio_best.pt` : Wav2Vec2-XLS-R layer weighting weights and attentive pooling head
- `fusion_calibrated.pt` : Multimodal late-fusion classifier + temperature parameter
- `syncnet.pt` : Audio-visual lip-sync embedding model

### Graceful Fallback Mode
If one or more checkpoint files are not present in this folder, the backend server continues running and logs a warning for that branch. The available branches will process uploaded media seamlessly.
