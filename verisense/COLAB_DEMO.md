# Two-day student demo: Colab training

Keep the existing image model and frame-mean video mode as a fallback. A trained
video checkpoint enables the temporal mode only after you copy it into the local
checkpoints folder. The audio branch requires its own evaluated checkpoint.

## 1. Restore the updated code and Drive

Upload `VeriSense_colab.zip` from your Mac's project root, then run:

```python
from google.colab import files, drive
from zipfile import ZipFile
from pathlib import Path

uploaded = files.upload()
with ZipFile(next(iter(uploaded))) as archive:
    archive.extractall("/content")
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
```

```python
%cd /content/verisense
import torch
assert torch.cuda.is_available(), "Select a GPU runtime"
base = Path("/content/drive/MyDrive/VeriSense")
for name in ("train", "val", "test"):
    assert (base / "splits" / f"image_{name}.csv").is_file()
raw = Path("/content/ffpp")
print("Raw videos available:", len(list(raw.rglob("*.mp4"))))
```

If raw videos are missing, redownload the same limited FaceForensics++ subset
using your authorized download script and the EU2 server. This needs no Mac storage.
Do not use the exported face images as video sequences: use the original videos.

## 2. Install training dependencies only if the runtime was reset

```python
%pip install -q "transformers>=4.38,<5" "albumentations==1.4.24" pandas scikit-learn pyyaml tqdm
```

Do not replace Colab's installed PyTorch. Keep the existing working OpenCV 4.x
installation; if it is 5.x, install `opencv-python-headless==4.11.0.86` and restart
the runtime if Colab requests it.

## 3. Cache video features once

The script uses frozen Hugging Face CLIP ViT-L/14, not random features or a second
OpenCLIP download. It samples up to 16 frames in the first 30 seconds and keeps
frames with exactly one detected face. It reuses source assignments from your
existing image splits and saves small feature arrays and CSVs to Drive.

```python
!python scripts/prepare_video_data.py --video-dir /content/ffpp --output-dir /content/drive/MyDrive/VeriSense/video_cache --split-dir /content/drive/MyDrive/VeriSense/splits
```

Inspect the per-class train/val/test counts before proceeding. Skipped videos are
reported; both classes must be present in every split.

## 4. Configure and train the small temporal model

```python
import yaml
config = yaml.safe_load(Path("configs/video.yaml").read_text())
for split in ("train", "val", "test"):
    config["data"][f"{split}_split_csv"] = str(base / "splits" / f"video_{split}.csv")
config["data"].update(num_workers=0, batch_size=8, val_batch_size=16)
run = base / "checkpoints/video_run1"
run.mkdir(parents=True, exist_ok=True)
config["training"].update(checkpoint_dir=str(run), epochs=10, early_stopping_patience=3)
video_config = run / "config.yaml"
video_config.write_text(yaml.safe_dump(config))
```

```python
!python src/video/train.py --config "{video_config}"
```

Review validation results before the one-time held-out test evaluation:

```python
!python src/video/evaluate.py --config "{video_config}" --checkpoint "{run / 'video_best.pt'}"
```

Metrics and per-video predictions are saved next to the checkpoint. Report the
number of test videos and errors; do not advertise frame accuracy as video accuracy.
The frame head has weak video-level supervision, not annotated manipulation timing.

Some FaceForensics++ c23 files use H.264 High 4:4:4. They decode in the backend
but may fail in a browser player. Make separate H.264 4:2:0 demo copies in Colab
before packaging them; retain the original files for benchmark evaluation:

```python
import subprocess
compatible = base / "mentor_video_browser"
compatible.mkdir(exist_ok=True)
for source in (base / "mentor_video_demo").glob("*.mp4"):
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-an", "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", "18", "-movflags", "+faststart", str(compatible / source.name),
    ], check=True)
```

Re-encoding changes pixels and can change scores. Check the converted copies
with the app; do not assume their predictions equal those of the originals.

Select the decision cutoff on validation predictions only, save its value and
`selected_on: validation` in `video_threshold.json`, and freeze it before reporting
test metrics. Export the selected cutoff with the model so the API uses the same
binary decision rule as the evaluation, rather than the image decision band:

```python
import json
selection = json.loads((run / "video_threshold.json").read_text())
assert selection["selected_on"] == "validation"
threshold = float(selection["threshold"])
assert 0 < threshold < 1
checkpoint = torch.load(run / "video_best.pt", map_location="cpu", weights_only=True)
checkpoint.update(decision_threshold=threshold, threshold_source="validation")
export = run / "export/video_best.pt"
export.parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, export)
files.download(str(export))
```

Place the exported file in local `verisense/checkpoints/video_best.pt`, restart Python,
and check `/api/models` for the temporal classifier and its `decision_threshold`.
The cutoff is an operating choice from a small validation set, not probability
calibration. Image and frame-mean baseline decisions retain their existing band.
If no temporal checkpoint
exists, the app explicitly uses the older frame-score mean baseline.

## 5. Audio data and compact export

Use ASVspoof 2019 LA, not the face-swap videos' soundtracks. FaceForensics++ visual
manipulation labels do not establish whether a voice is synthetic.

Author dataset: https://zenodo.org/records/6906306 (LA.zip is approximately 7.6 GB).
Use its official train/dev/eval protocol files; these partitions separate speakers.
The revised `scripts/prepare_audio_data.py` requires `--protocol-file`, reads the
actual bonafide/spoof labels, and defaults to at most 500 examples per class.
The audio config uses a smaller frozen Wav2Vec2 base encoder with trainable pooling
and classifier. Keep the official train/dev/eval partitions, save selected FLACs
to Drive, and save the CSV columns `filepath`, `label`, `speaker_id`, `audio_id`
and `system_id`. For the demo use 500 clips per class for training and 100 per
class for validation and testing. Label 0 is bonafide and label 1 is spoof.

After training, select a balanced-accuracy cutoff from validation scores only,
save `audio_threshold.json`, and evaluate the fixed cutoff on the held-out test
partition. Export the trained non-backbone weights with the cutoff:

```python
import json
import torch
from pathlib import Path
from google.colab import files
run = Path("/content/drive/MyDrive/VeriSense/checkpoints/audio_run1")
selection = json.loads((run / "audio_threshold.json").read_text())
assert selection["selected_on"] == "validation"
checkpoint = torch.load(run / "audio_best.pt", map_location="cpu", weights_only=True)
config = checkpoint["config"]
assert config["model"]["unfreeze_last_n_layers"] == 0
export = run / "export/audio_head.pt"
export.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_name": config["model"]["backbone_name"],
    "hidden_dim": config["model"]["hidden_dim"],
    "dropout": config["model"]["dropout"],
    "sample_rate": 16000, "duration": 4.0,
    "class_names": ["real", "manipulated"],
    "decision_threshold": float(selection["threshold"]),
    "threshold_source": "validation",
    "head_state_dict": {k: v.cpu() for k, v in checkpoint["model_state_dict"].items()
                        if not k.startswith("backbone.")},
}, export)
files.download(str(export))
```

Put the export in local `verisense/checkpoints/audio_head.pt`, update the API
dependencies and restart Python. `/api/models` should report the speech classifier
and its cutoff. Test WAV/FLAC uploads before demonstrating; the API checks only
the central four seconds of clips up to 30 seconds. Fusion and lip-sync remain
unavailable; video analysis does not automatically analyze its soundtrack.

## Evaluation scope

These are student research baselines. FaceForensics++ and ASVspoof 2019 do not
establish general reliability on new image or voice generators. Keep demo examples
labeled and separate from training, report mistakes, and retain a working fallback.

## Paired LAV-DF fusion features

The saved pilot subset is `VeriSense/fusion/lavdf_subset` on Drive: 100 source
families for train, 25 for validation and 25 for test, with one real, video-only,
audio-only and jointly manipulated clip per family (600 clips). These are 25%
real and 75% manipulated overall, not balanced binary classes. Related originals
stay within the official partitions; the annotations do not prove speaker separation.

`src.fusion.build_features` requires the three evaluated compact checkpoints.
It uses one CLIP pass for raw image-head features and normalized temporal features,
uniformly samples up to 16 frames, and averages/max-pools image scores. It decodes
the actual soundtrack with ffmpeg, applies the training audio normalization to
overlapping four-second windows (two-second hop, including the end), and stores
mean/max speech scores. Window placement never depends on the manipulation labels
or timestamps. This is sampled late fusion, not lip-sync or localization training.

Upload the locally generated `VeriSense_fusion_setup.zip` (source plus the three
small trained checkpoints), then run this in a GPU Colab runtime. The ZIP and per-clip
results persist on Drive, so restarting later does not require another dataset download.

```python
from google.colab import files, drive
from pathlib import Path
from zipfile import ZipFile
import shutil

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
setup = Path("/content/drive/MyDrive/VeriSense/setup/VeriSense_fusion_setup.zip")
if not setup.exists():
    uploaded = files.upload()
    source = Path(next(name for name in uploaded if name.endswith(".zip")))
    setup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, setup)
with ZipFile(setup) as archive:
    assert archive.testzip() is None
    archive.extractall("/content")
```

```python
%pip install -q "transformers>=4.38,<5" pandas scipy tqdm
```

```python
import subprocess
import sys
import torch
from pathlib import Path

assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU"
assert shutil.which("ffmpeg"), "ffmpeg is required for MP4 soundtracks"
base = Path("/content/drive/MyDrive/VeriSense")
command = [sys.executable, "-m", "src.fusion.build_features",
    "--manifest-dir", str(base / "fusion/lavdf_subset/manifests"),
    "--checkpoint-dir", "/content/verisense/checkpoints",
    "--output-dir", str(base / "fusion/lavdf_features_v1")]
subprocess.run(command + ["--limit-per-split", "4"], cwd="/content/verisense", check=True)
```

Inspect the 12-clip check first. Then use `subprocess.run(command,
cwd="/content/verisense", check=True)` for the full subset. Successful per-clip JSON
records are reused; changed videos are rescored. Model, manifest or extraction-code
changes require a new output directory. Failed face/audio decoding is reported
explicitly in the feature CSVs, never replaced by random features or 0.5 scores.
Inspect failure counts per category before training to assess selection bias.
The learned fusion checkpoint is still pending; feature extraction alone does not
activate a fusion model in the API.

## Train the small fusion classifier

The completed 600-clip pilot extraction retained 357/400 training, 90/100 validation
and 92/100 test clips. All 61 reported rejections had fewer than four sampled frames
with exactly one detected face. Preserve the extraction rule and record these as
abstentions; a rejected file is not a successful classification.

The trainer uses five score inputs and balanced-class L2 logistic regression.
Standardization is fitted on training data only. Four regularization candidates
are compared using validation AUC, then a validation cutoff maximizes balanced
accuracy. The checkpoint is frozen before test evaluation. Each individual score
is also evaluated on the same usable test clips with its own validation cutoff.
The output reports category accuracy, majority-class accuracy and rejection coverage.
Clips from a source family remain correlated even within a partition; the small
test set does not establish broad reliability or independent identity generalization.

`VeriSense_fusion_training.zip` is a small update containing only the trainer,
inference model and fusion config. Extract it after restoring the setup ZIP; it
does not change the extractor, its caches or the three trained branch checkpoints.
Save this update in `VeriSense/setup` on Drive for future runtime resets.

```python
%pip install -q pandas scikit-learn pyyaml
```

```python
from google.colab import drive, files
from pathlib import Path
from zipfile import ZipFile
import shutil
import subprocess
import sys

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
base = Path("/content/drive/MyDrive/VeriSense")
if not Path("/content/verisense/src/common/metrics.py").exists():
    with ZipFile(base / "setup/VeriSense_fusion_setup.zip") as archive:
        assert archive.testzip() is None
        archive.extractall("/content")

update = base / "setup/VeriSense_fusion_training.zip"
source = update
if not update.exists():
    uploaded = files.upload()
    names = [name for name in uploaded if name.lower().endswith(".zip")]
    assert len(names) == 1, "Upload VeriSense_fusion_training.zip."
    source = Path(names[0])
with ZipFile(source) as archive:
    required = {"verisense/src/fusion/train_fusion.py", "verisense/src/fusion/model.py", "verisense/configs/fusion.yaml"}
    assert required.issubset(archive.namelist()), "Select the fusion training update ZIP."
    assert archive.testzip() is None
    archive.extractall("/content")
update.parent.mkdir(parents=True, exist_ok=True)
if source != update:
    shutil.copy2(source, update)

result = subprocess.run([
    sys.executable, "-m", "src.fusion.train_fusion",
    "--features-dir", str(base / "fusion/lavdf_features_v1"),
    "--manifest-dir", str(base / "fusion/lavdf_subset/manifests"),
    "--output-dir", str(base / "checkpoints/fusion_run1"),
], cwd="/content/verisense", capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
result.check_returncode()
```

This training step runs on CPU and does not download backbones or repeat feature
extraction. It writes `fusion_best.pt`, `fusion_metrics.json`, and validation/test
prediction CSVs to `VeriSense/checkpoints/fusion_run1`. Scores are not calibrated
probabilities. Review the held-out result before connecting this model to the API.

## LAV-DF adaptation after the first fusion experiment

The first score-only fusion run produced 53.26% accuracy, 57.25% balanced accuracy
and 0.6446 AUC on 92 usable test clips, with eight preprocessing rejections. Its
train/validation diagnostics show weak visual score discrimination (AUC about
0.53–0.58 against `modify_video`) and saturated speech scores even on unmodified
audio. This suggests poor transfer of the FF++/ASVspoof heads to LAV-DF; the existing
benchmark results do not establish LAV-DF performance. Keep this as the recorded
first experiment rather than tuning its cutoff on the test set.

The next experiment adapts heads using the same frozen encoders. First cache visual
embeddings using `scripts/prepare_lavdf_visual.py`. It reads only the train/validation
manifests, uniformly samples the same 16 frames, preserves the four-face-frame
minimum, and labels each usable frame using `modify_video` and the annotated
`fake_periods` in seconds. An audio-only fake is visually unmodified. Unedited
portions of a visually manipulated clip remain negative frame examples. Annotation
times label samples; they do not determine where validation frames are sampled.

The cache stores raw projected CLIP features, timestamps, per-frame labels and a
visual clip label in compressed NPZ files. The raw features support image-head
training; L2-normalized copies can later support temporal-head training. Resume
records include video checksums and extraction provenance. This is a bounded
adaptation experiment; improved detection is not guaranteed, and fully generated
modern images remain a separate evaluation problem.

In this visual cache's CSVs, `label` means `modify_video`; `overall_label` preserves
the original audio-or-video manipulation label for later paired fusion work.

Upload `VeriSense_lavdf_visual.zip`, extract it into `/content`, then run on T4:

```python
base = Path("/content/drive/MyDrive/VeriSense")
command = [sys.executable, "-u", "scripts/prepare_lavdf_visual.py",
    "--subset-dir", str(base / "fusion/lavdf_subset"),
    "--checkpoint", "/content/verisense/checkpoints/image_head.pt",
    "--output-dir", str(base / "adaptation/lavdf_visual_v1")]
with subprocess.Popen(command, cwd="/content/verisense", stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Visual feature caching failed; inspect the output above.")
```

Inspect both the positive/negative frame counts and the number of visual-fake
clips with any sampled edited face. Sparse sampling can miss short edits entirely.
The in-domain head trainer is the subsequent step; this cache does not replace
the original checkpoints or activate the unsuccessful score-only fusion model.

### Fit the visual head from cached embeddings

The completed cache contains 4,529 unmodified and 182 edited training frames from
357 usable clips, and 1,093 unmodified and 54 edited validation frames from 90 clips.
Uniform sampling captured an edited face in 120/178 visually fake training clips
and 32/46 visually fake validation clips. The other modified clips remain in the
clip evaluation; their absent sampled edits are a limitation of this sampling budget.

`scripts/train_lavdf_head.py` warm-starts the existing classifier, freezes the
cached CLIP embeddings, and uses inverse-frequency class weights for frame loss.
It selects by validation visual-clip AUC with max pooling over the sampled faces,
reports frame AUC separately, and stops after four epochs without improvement
(15 epochs maximum). The original head is candidate epoch zero, so an unsuccessful
adaptation retains its weights. No test manifest or test features are opened.

Upload `VeriSense_lavdf_head.zip` and extract it after restoring the project. The
update contains the trainer and the shared validation cutoff helper. Run:

```python
base = Path("/content/drive/MyDrive/VeriSense")
command = [sys.executable, "-u", "scripts/train_lavdf_head.py",
    "--cache-dir", str(base / "adaptation/lavdf_visual_v1"),
    "--checkpoint", "/content/verisense/checkpoints/image_head.pt",
    "--output-dir", str(base / "checkpoints/visual_lavdf_run1")]
with subprocess.Popen(command, cwd="/content/verisense", stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Head training failed; inspect the output above.")
```

This step needs pandas, scikit-learn, PyYAML and Colab's existing PyTorch; it does
not instantiate or download CLIP. It writes `image_lavdf_head.pt`,
`visual_head_metrics.json` and `visual_val_predictions.csv` to the new run folder.
The checkpoint's `video_decision_threshold` applies to the maximum sampled-frame
score, not to an individual photo. Original checkpoints remain separate. Any
reported validation improvement is a development result, not a new held-out result
or evidence of general reliability on modern AI-generated images.

### Adapt the temporal classifier using the same cache

The visual-head experiment selected epoch 15: validation frame AUC increased from
0.5738 to 0.8273 and max-pooled visual-clip AUC from 0.5751 to 0.6944. At the chosen
clip cutoff it still produced 23 false positives among 44 visually unmodified clips.
These are validation development results. The next controlled experiment fits the
existing temporal classifier to the same cached sequences and annotated frame targets.

`VideoEmbeddingDataset` now accepts raw projected NPZ features, applies L2 normalization,
and pads/subsamples features and frame targets together. Old NPY caches retain their
weak clip-derived frame targets. Annotated frame loss uses inverse-frequency weights
computed from actual training targets, excluding padding. The temporal model is
warm-started from the evaluated FF++ checkpoint and keeps that checkpoint as candidate
epoch zero. Model selection and the operating cutoff use validation only.

In annotated training, the clip loss uses genuine clips and manipulated clips with
at least one sampled edited face. A visually manipulated clip whose edit was missed
still supplies correct unmodified frame targets, but no contradictory positive clip
loss. Clip class weights are derived from those observable training examples. All
usable validation clips, including ones whose edits were missed, remain in evaluation.

Upload `VeriSense_lavdf_temporal.zip` and extract into `/content` after restoring the
base setup. It updates the small training wrapper, video dataset/trainer, and shared
metrics helper. The wrapper preloads the small cache into RAM and writes a separate
run under `VeriSense/checkpoints/temporal_lavdf_run1`:

```python
base = Path("/content/drive/MyDrive/VeriSense")
command = [sys.executable, "-u", "scripts/train_lavdf_temporal.py",
    "--cache-dir", str(base / "adaptation/lavdf_visual_v1"),
    "--warm-start", "/content/verisense/checkpoints/video_best.pt",
    "--output-dir", str(base / "checkpoints/temporal_lavdf_run1")]
with subprocess.Popen(command, cwd="/content/verisense", stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Temporal adaptation failed; inspect the output above.")
```

Outputs are `video_lavdf_best.pt`, `video_val_metrics.json`, `video_val_predictions.csv`
and the run config. This experiment concerns visual manipulation, so audio-only fakes
retain visual label zero. It reads no test manifest and does not establish improved
audio detection, lip-sync analysis or broad reliability. Original checkpoint files
remain separate from the experimental run.

### Audio adaptation after the temporal experiment

The first temporal adaptation did not beat the original head: its best new-epoch
validation AUC was 0.5267 versus the original 0.5652. Epoch zero was selected.
Its reported 58.89% accuracy comes from a validation-selected cutoff, not improved
ranking or newly learned weights. CPU training is supported; the run completed
normally. Keep the better visual-head experiment for subsequent fusion work.

`scripts/adapt_lavdf_audio.py` next caches speech embeddings from four-second
windows with two-second hop and an end-covering window. The Wav2Vec2 backbone,
learned layer combination and attentive pooling remain frozen. Only the final
classification head is adapted. Window labels use `modify_audio` and overlap
with annotated edit intervals; a video-only fake has audio label zero. Sampling
is annotation-independent and includes all available train/validation audio clips,
including videos rejected by the visual face detector. Compare the baseline and
adapted audio head on this same cohort; its counts may differ from the original
90-clip paired validation table.

Audio preparation and fusion now share the same decoder/windowing helpers in
`src/audio/preprocessing.py`. The existing audio forward path still computes
the same weighted-pooling features followed by the same classifier. The general
head trainer supports both visual frame features and speech window features.

Upload `VeriSense_lavdf_audio.zip`, extract after restoring the base project, and run:

```python
base = Path("/content/drive/MyDrive/VeriSense")
command = [sys.executable, "-u", "scripts/adapt_lavdf_audio.py",
    "--subset-dir", str(base / "fusion/lavdf_subset"),
    "--checkpoint", "/content/verisense/checkpoints/audio_head.pt",
    "--cache-dir", str(base / "adaptation/lavdf_audio_v1"),
    "--output-dir", str(base / "checkpoints/audio_lavdf_run1")]
with subprocess.Popen(command, cwd="/content/verisense", stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Audio adaptation failed; inspect the output above.")
```

T4 speeds up speech feature caching; CPU also works. Each clip's feature result is
saved to Drive and reused after a restart. The trained head is saved separately as
`audio_lavdf_head.pt`. Its `decision_threshold` is selected on validation windows;
`clip_decision_threshold` is selected on max-pooled validation clips. These distinct
scopes must be preserved when integrating standalone audio versus clip fusion.
The export retains all original frozen layer/pooling parameters, so no randomly
initialized pooling weights enter inference. No test split is read by this adaptation.

### Refit fusion after head adaptation

Audio adaptation selected epoch 15: validation clip AUC rose from 0.6652 to 0.9492,
with 92% accuracy on 100 clips. The validation window AUC was 0.9675. These are
development results; correlated windows are not independent test examples.

`scripts/train_lavdf_fusion.py` uses the existing visual/audio embedding caches.
Three folds keep each source family and its manipulation variants together. Each
fold starts from the original FF++/ASVspoof checkpoint and trains only on the other
families for the previously selected epoch count. Fusion learns from these held-out
training predictions, rather than scores from heads fitted on the same clips.
Validation predictions use the selected full-training heads. The original temporal
scores remain unchanged because temporal adaptation did not improve validation.

The update ZIP includes the wrapper, shared head trainer, logistic fusion trainer,
model, metrics and configuration. After extracting it into `/content`, run:

```python
command = [sys.executable, "-u", "scripts/train_lavdf_fusion.py",
    "--base-dir", str(base),
    "--checkpoint-dir", "/content/verisense/checkpoints",
    "--feature-dir", str(base / "fusion/lavdf_oof_v1"),
    "--output-dir", str(base / "checkpoints/fusion_run2")]
subprocess.run(command, cwd="/content/verisense", check=True)
```

CPU works; no backbone download or feature extraction is needed. Completed fold
predictions are saved and reused. Original/adapted weight identities and cached
feature provenance must match. Rejected visual clips remain abstentions. The small
fusion model selects regularization and its cutoff on validation only; this command
opens no test tables. Review validation balanced accuracy and per-category results,
then freeze the configuration before a separate held-out evaluation. The checkpoint
is experimental and is not automatically activated by the API.

### Frozen fusion test

Use the project's pinned `opencv-python-headless==4.11.0.86`; do not install an
unpinned OpenCV release for this experiment. If `cv2.CascadeClassifier` is missing,
remove all four overlapping OpenCV distributions and reinstall only the pinned
headless package. They share the `cv2` namespace:

```python
%pip uninstall -y -q opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
%pip install -q --no-cache-dir --no-deps "opencv-python-headless==4.11.0.86"
```

Check the installation in a fresh subprocess before rerunning evaluation:

```python
subprocess.run([sys.executable, "-c", "import cv2; "
    "assert hasattr(cv2, 'CascadeClassifier'); "
    "assert not cv2.CascadeClassifier(cv2.data.haarcascades + "
    "'haarcascade_frontalface_default.xml').empty(); "
    "print('OpenCV:', cv2.__version__, 'Haar detector ready')"], check=True)
```

The evaluator also starts a fresh Python process, so these commands do not require
deleting the runtime. Do not add MediaPipe just to fix the Haar installation:
evaluation must preserve the face backend recorded with the training cache.

The OOF run selected C=10 and cutoff 0.5 on validation. Among 90/100 scored clips,
accuracy was 85.56%, balanced accuracy 87.44%, and AUC 0.8994. Category accuracy was
100% for audio-only/both, 91.30% for real, and 52.17% for video-only manipulation.
Ten validation clips were rejected rather than counted as correct predictions.

`scripts/evaluate_lavdf_fusion.py` freezes those choices and verifies the exact
three branch checkpoint identities recorded by fusion. It scores only the saved
test videos, using the same sampler, face backend, shared audio window preprocessing,
and max/mean pooling. Family overlap is checked across all manifests, including
rejected clips. There is no fitting or test-based cutoff selection. Baseline cutoffs
are loaded from the saved validation report. Results and explicit failures resume
from per-clip JSON files, guarded by input/code fingerprints.

After extracting `VeriSense_lavdf_fusion_test.zip` over the restored base setup:

```python
command = [sys.executable, "-u", "scripts/evaluate_lavdf_fusion.py",
    "--base-dir", str(base),
    "--checkpoint-dir", "/content/verisense/checkpoints",
    "--feature-dir", str(base / "fusion/lavdf_frozen_test_v2"),
    "--output-dir", str(base / "checkpoints/fusion_run2_test")]
subprocess.run(command, cwd="/content/verisense", check=True)
```

T4 is recommended for this step because the 100 test clips require fresh backbone
inference; CPU also works. The existing 600 saved MP4 files avoid another dataset
download, although a fresh runtime may need the public CLIP/Wav2Vec2 weights again.
Outputs are `fusion_test_metrics.json` and `fusion_test_predictions.csv`. This is
the existing test cohort already inspected for the initial fusion baseline, not an
entirely new untouched benchmark. Keep that context with any reported improvement.
This evaluator preserves `fusion_run2` and does not activate fusion in the API.

### Install the evaluated bundle locally

Export `image_lavdf_head.pt`, `audio_lavdf_head.pt`, the original `video_best.pt`,
`fusion_best.pt`, and validation/test fusion reports in a `lavdf/` folder. Install
that folder under `verisense/checkpoints/`, keeping the original standalone image
and audio files. The frozen LAV-DF test scored 92/100 clips: 78.26% accuracy,
81.16% balanced accuracy and AUC 0.9074. Video-only accuracy was 43.48%.

The backend verifies exact adapted/source checkpoint hashes, face backend and
speech preprocessing before reporting fusion active. Only small adapted heads are
added; the already loaded CLIP and Wav2Vec2 backbones are shared. Install the API
requirements for the portable FFmpeg decoder, restart the API and refresh the UI.
Video uploads then offer combined analysis or the existing visual-only mode.
The fusion route preserves the evaluated five-score aggregation and cutoff 0.5;
missing audio/faces yield an explicit rejection rather than a replacement verdict.
