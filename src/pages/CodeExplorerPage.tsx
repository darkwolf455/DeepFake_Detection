import React, { useState } from 'react';
import {
  Code2,
  FileCode,
  Folder,
  FolderOpen,
  Copy,
  Check,
  Download,
  Terminal,
  BookOpen,
  FileText,
} from 'lucide-react';

interface RepoFile {
  path: string;
  category: string;
  description: string;
  snippet: string;
}

const REPO_FILES: RepoFile[] = [
  {
    path: 'verisense/README.md',
    category: 'Documentation',
    description: 'Complete setup, dataset download instructions, and training order.',
    snippet: `# VeriSense: Multimodal Deepfake Detection System
## Production PyTorch System for Images, Videos, and Audio

1. Setup:
   conda env create -f environment.yml
   conda activate verisense

2. Prepare Data & Identity-Protected Splits:
   python scripts/prepare_image_data.py --input-dir data/raw/genimage
   python scripts/prepare_video_data.py --video-dir data/raw/faceforensics
   python scripts/prepare_audio_data.py --input-dir data/raw/asvspoof2019
   python scripts/make_splits.py

3. Train Image Branch:
   python src/image/train.py --config configs/image.yaml

4. Train Video Branch:
   python src/video/train.py --config configs/video.yaml

5. Train Audio Branch:
   python src/audio/train.py --config configs/audio.yaml

6. Train Calibrated Late Fusion:
   python src/fusion/build_features.py --config configs/fusion.yaml
   python src/fusion/train_fusion.py --config configs/fusion.yaml

7. Launch FastAPI Backend:
   cd backend && uvicorn app:app --host 0.0.0.0 --port 8000`,
  },
  {
    path: 'verisense/configs/image.yaml',
    category: 'Configuration',
    description: 'CLIP ViT-L/14 backbone, Albumentations compression parameters, and training settings.',
    snippet: `model:
  name: "openai/clip-vit-large-patch14"
  pretrained: true
  freeze_backbone: true
  head_type: "mlp"          # "linear" or "mlp"
  hidden_dim: 512
  dropout: 0.2
  use_lora: false           # Set true to enable PEFT LoRA fine-tuning
  lora_r: 8
  lora_alpha: 16

data:
  image_size: 224
  batch_size: 32            # Fits 8-12GB GPU VRAM
  class_balanced_sampling: true

augmentation:
  jpeg_compression_min: 30
  jpeg_compression_max: 100
  gaussian_blur_prob: 0.3
  horizontal_flip_prob: 0.5

training:
  epochs: 15
  lr: 0.0001
  label_smoothing: 0.05
  gradient_accumulation_steps: 2
  use_amp: true`,
  },
  {
    path: 'verisense/configs/video.yaml',
    category: 'Configuration',
    description: 'MediaPipe face extraction, 32-frame sampling, and Temporal Transformer configuration.',
    snippet: `face_extractor:
  backend: "mediapipe"
  confidence_threshold: 0.65
  margin_ratio: 0.25
  target_face_size: 224

temporal_model:
  type: "transformer"       # "transformer" or "bilstm"
  input_dim: 768
  d_model: 512
  nhead: 8
  num_layers: 4
  dim_feedforward: 1024
  max_frames: 32
  pooling: "attention"

training:
  epochs: 20
  lr: 0.0002
  use_amp: true`,
  },
  {
    path: 'verisense/configs/audio.yaml',
    category: 'Configuration',
    description: 'Wav2Vec2 XLS-R 300M, RawBoost augmentations, and focal loss settings.',
    snippet: `model:
  backbone_name: "facebook/wav2vec2-xls-r-300m"
  sample_rate: 16000
  freeze_feature_extractor: true
  unfreeze_last_n_layers: 2
  aggregate_hidden_states: true
  classifier_type: "attentive_pooling"

audio_processing:
  target_sample_rate: 16000
  crop_duration_seconds: 4.0

augmentation:
  use_rawboost: true
  additive_noise_prob: 0.4
  codec_simulation_prob: 0.4

training:
  epochs: 18
  lr: 0.00005
  loss_type: "focal"
  monitor_metric: "val_eer"`,
  },
  {
    path: 'verisense/src/image/model.py',
    category: 'Image Branch',
    description: 'CLIP ViT-L/14 model architecture with frozen backbone and trainable MLP head.',
    snippet: `import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

class CLIPDeepfakeDetector(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze_backbone=True, head_type="mlp", hidden_dim=512, dropout=0.2, use_lora=False):
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        embed_dim = self.backbone.config.projection_dim

        if freeze_backbone and not use_lora:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 2))

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return self.head(outputs.image_embeds)`,
  },
  {
    path: 'verisense/src/video/model.py',
    category: 'Video Branch',
    description: 'Temporal Transformer Encoder with sinusoidal positional encoding and attentive pooling.',
    snippet: `import torch
import torch.nn as nn

class VideoTemporalDetector(nn.Module):
    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.2, max_frames=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames + 16)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooler = AttentionPooling(d_model)
        self.video_classifier = nn.Linear(d_model, 2)
        self.frame_classifier = nn.Linear(d_model, 2)

    def forward(self, embeddings, mask=None):
        x = self.pos_encoder(self.input_proj(embeddings))
        encoded_seq = self.sequence_encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        video_feat = self.pooler(encoded_seq, mask=mask)
        return self.video_classifier(video_feat), self.frame_classifier(encoded_seq)`,
  },
  {
    path: 'verisense/src/audio/model.py',
    category: 'Audio Branch',
    description: 'Wav2Vec2 XLS-R layer-weighted sum with attentive temporal pooling.',
    snippet: `import torch
import torch.nn as nn
from transformers import AutoModel

class Wav2Vec2DeepfakeDetector(nn.Module):
    def __init__(self, backbone_name="facebook/wav2vec2-xls-r-300m", unfreeze_last_n_layers=2, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        num_layers = self.backbone.config.num_hidden_layers + 1
        embed_dim = self.backbone.config.hidden_size

        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.pooler = AttentivePooling(embed_dim, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_values):
        outputs = self.backbone(input_values, output_hidden_states=True)
        norm_weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(outputs.hidden_states, dim=0)
        weighted_features = torch.sum(stacked * norm_weights.view(-1, 1, 1, 1), dim=0)
        return self.head(self.pooler(weighted_features))`,
  },
  {
    path: 'verisense/src/fusion/train_fusion.py',
    category: 'Late Fusion',
    description: 'Calibrated late-fusion engine with Platt temperature scaling and missing modality masks.',
    snippet: `class CalibratedLateFusion:
    def __init__(self, model_type="mlp", hidden_dim=32, temperature=1.0, inconclusive_range=(0.40, 0.65)):
        self.temperature = temperature
        self.inconclusive_lower, self.inconclusive_upper = inconclusive_range
        self.model = MultimodalFusionMLP(in_features=8, hidden_dim=hidden_dim)

    def prepare_input_vector(self, scores_dict):
        # Encodes [s_img, s_vid, s_aud, s_sync, mask_img, mask_vid, mask_aud, mask_sync]
        modalities = ["image", "video", "audio", "av_sync"]
        feat, masks = [], []
        for m in modalities:
            val = scores_dict.get(m)
            if val is not None and not np.isnan(val):
                feat.append(float(val)); masks.append(1.0)
            else:
                feat.append(0.5); masks.append(0.0) # Uninformative prior
        return np.array(feat + masks, dtype=np.float32)

    def predict(self, scores_dict):
        x = self.prepare_input_vector(scores_dict)
        logit = self.model(torch.from_numpy(x).unsqueeze(0)).item()
        calibrated_prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        
        if self.inconclusive_lower <= calibrated_prob <= self.inconclusive_upper:
            verdict = "inconclusive"
        elif calibrated_prob > self.inconclusive_upper:
            verdict = "manipulated"
        else:
            verdict = "authentic"
        return {"calibrated_prob": calibrated_prob, "verdict": verdict}`,
  },
  {
    path: 'verisense/backend/app.py',
    category: 'FastAPI Backend',
    description: 'FastAPI server with magic-byte validation, concurrency semaphore, and graceful fallback.',
    snippet: `from fastapi import FastAPI, UploadFile, File
from backend.preprocessing import SafeTempFile
from backend.inference import InferenceManager

app = FastAPI(title="VeriSense Backend", version="1.0.0")
inference_manager = InferenceManager()

@app.on_event("startup")
async def startup():
    inference_manager.load_models()

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    async with SafeTempFile(file, max_bytes=150 * 1024 * 1024) as (path, mime, modality, sha256):
        return await inference_manager.analyze_file(path, modality, sha256)`,
  },
];

export const CodeExplorerPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<RepoFile>(REPO_FILES[0]);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadFile = () => {
    const blob = new Blob([selectedFile.snippet], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile.path.split('/').pop() || 'code.py';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider">
              Python 3.10+ • PyTorch 2.x
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white mt-1 flex items-center gap-2.5">
            <Code2 className="w-7 h-7 text-cyan-400" />
            <span>Repository & Model Training Source</span>
          </h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-1">
            Browse runnable Python training pipelines, configs, face extraction, and calibration modules created for local execution in VS Code.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
            verisense/
          </span>
        </div>
      </div>

      {/* Explorer Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: File Tree List */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-2 h-[550px] overflow-y-auto">
          <div className="flex items-center gap-2 px-2 pb-2 mb-2 border-b border-slate-800 text-xs font-bold text-slate-400 uppercase tracking-wider">
            <FolderOpen className="w-4 h-4 text-cyan-400" />
            <span>Project Files</span>
          </div>

          {REPO_FILES.map((file) => {
            const isSelected = selectedFile.path === file.path;
            return (
              <button
                key={file.path}
                onClick={() => setSelectedFile(file)}
                className={`w-full text-left p-2.5 rounded-xl transition-all flex items-start gap-2.5 ${
                  isSelected
                    ? 'bg-cyan-500/15 border border-cyan-500/30 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <FileCode className={`w-4 h-4 shrink-0 mt-0.5 ${isSelected ? 'text-cyan-400' : 'text-slate-500'}`} />
                <div className="min-w-0">
                  <div className="text-xs font-mono font-medium truncate">{file.path}</div>
                  <div className="text-[10px] text-slate-500 font-sans mt-0.5">{file.category}</div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Right: Code Viewer & Actions */}
        <div className="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-2xl shadow-xl flex flex-col h-[550px] overflow-hidden">
          {/* Top file meta header */}
          <div className="p-4 border-b border-slate-800 bg-slate-950/70 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-cyan-400 shrink-0" />
                <span className="font-mono text-xs font-bold text-slate-200 truncate">
                  {selectedFile.path}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5 truncate">{selectedFile.description}</p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                onClick={handleDownloadFile}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download</span>
              </button>
            </div>
          </div>

          {/* Syntax Highlighted Code Box */}
          <div className="p-4 flex-1 overflow-auto bg-slate-950 font-mono text-xs text-slate-300 leading-relaxed">
            <pre>
              <code>{selectedFile.snippet}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
