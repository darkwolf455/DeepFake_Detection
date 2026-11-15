# VeriSense student baseline: measured scope and limitations

## Held-out results

| Branch | Test subset | Accuracy | AUC | Errors |
| --- | --- | --- | --- | --- |
| Image | 200 face crops, FaceForensics++ Deepfakes | 90% | 0.9714 | 12 false positives, 8 misses |
| Video | 32 videos, FaceForensics++ Deepfakes | 84.38% | 0.9688 | 1 false positive, 4 misses |
| Audio | 200 clips, ASVspoof 2019 LA | 97% | 0.9986 | 1 false positive, 5 misses |
| Experimental fusion | 92 scored / 100 LAV-DF clips | 78.26% | 0.9074 | 3 false positives, 17 misses; 8 rejected |

These small benchmark subsets do not measure reliability on arbitrary uploads.
Image crops from one video are correlated; 200 crops are not 200 independent
people or videos. The video and audio cutoffs were selected on validation data,
then fixed for test evaluation. Their scores are not calibrated probabilities.
Fusion balanced accuracy was 81.16%, with a 75% majority-class accuracy baseline.
Video-only manipulation accuracy was 43.48%, versus 91.30% each for audio-only
and combined edits. The test cohort was already inspected for the initial fusion
baseline. Related variants are correlated; source families are separated across
partitions, while speaker/identity separation is unverified. Rejections are
abstentions, not correct predictions.

## What the demo analyzes

- Images: exactly one detected face. The model was trained on face manipulation,
  not a broad collection of modern fully generated images. It has incorrectly
  accepted an AI portrait during manual testing.
- Visual-only videos: up to 16 samples within the first 30 seconds, using frames with one
  detected face. The soundtrack and lip-sync are not analyzed. Frame scores
  have weak video-level supervision, not annotated manipulation timestamps.
- Combined video/audio: requires at least four usable face samples and audible
  speech in a clip up to 30 seconds. Uses the evaluated LAV-DF visual/audio heads,
  original temporal model, and four-second audio windows with two-second hop.
  Fusion combines five real scores; displayed image/audio scores are maxima.
  No verdict is returned for unusable required modalities. Visual-only analysis
  remains an explicit alternative. There is no lip-sync model or fusion timeline.
- Audio: the central four seconds of a mono/stereo clip up to 30 seconds long.
  Short clips repeat to four seconds. Other portions of the recording are not
  checked. The training set does not establish performance on modern voice tools,
  other languages, music, noisy recordings or replayed speech.

Lip-sync analysis and image heatmaps remain unavailable. Compression,
noise or an unseen generator can cause either a false verdict or an inconclusive
image result; the app has no validated automatic degradation detector.

## Interpretation and storage

Present results as model decisions for human review, with known errors and the
evaluation dataset stated. A verdict does not prove authenticity or fabrication.
Uploaded files use temporary storage with automatic cleanup after processing.
The frontend history retains result metadata and is not a media archive.
