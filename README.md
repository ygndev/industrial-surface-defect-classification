# Industrial Surface Defect Classification

A small PyTorch CNN for classifying grayscale steel surface images from NEU-CLS into six defect classes: crazing (Cr), inclusion (In), patches (Pa), pitted surface (PS), rolled-in scale (RS), and scratches (Sc).

## Data and split

The local dataset contains 1,800 BMP images, each 200 × 200 pixels. SHA-256 hashing identifies one exact duplicate pair: `Pa_101.bmp` and `Pa_105.bmp`. `split_data.py` retains the first file in sorted path order and excludes the duplicate from the split manifest; it does not delete raw files.

The resulting 1,799 unique images are stratified into 1,259 training, 270 validation, and 270 test images using seed 42. `data/splits.csv` records relative paths and split assignments. No exact image hash is shared across splits.

Obtain NEU-CLS separately from its dataset provider and check its usage terms. This repository does not distribute the dataset. Preserve the paths in the manifest, for example `data/raw/NEU-CLS/In_287.bmp`. The manifest alone does not supply the images. `split_data.py` expects all 1,800 original files, including the duplicate.

## Model and training

`DefectCNN` has 23,910 trainable parameters: three convolution blocks (16, 32, and 64 channels), each with batch normalization, ReLU, and max pooling, followed by global average pooling and a six-class linear layer.

Images are converted to grayscale and normalized with mean 0.5 and standard deviation 0.5. Training adds horizontal and vertical flips and rotation up to 10 degrees. Training uses cross entropy, AdamW (learning rate 0.001, weight decay 0.0001), batch size 16, and up to 40 epochs. Early stopping uses validation loss with patience 7; the saved checkpoint is selected by lowest validation loss.

## Verified results

The existing checkpoint was evaluated locally on CPU on 2026-11-10 without retraining. Its saved epoch and validation accuracy were inspected, and validation/test predictions were recomputed using the checked-in code and local split manifest.

| Measurement | Value |
| --- | ---: |
| Best checkpoint epoch | 21 |
| Validation accuracy | 96.30% |
| Test accuracy | 95.56% (258/270) |
| Test macro F1 | 0.9552 |
| Misclassified test images | 12 |

The saved test confusion matrix uses rows for actual classes and columns for predicted classes:

| Actual / predicted | Cr | In | Pa | PS | RS | Sc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cr | 45 | 0 | 0 | 0 | 0 | 0 |
| In | 0 | 42 | 0 | 2 | 0 | 1 |
| Pa | 2 | 0 | 43 | 0 | 0 | 0 |
| PS | 0 | 6 | 0 | 38 | 0 | 1 |
| RS | 0 | 0 | 0 | 0 | 45 | 0 |
| Sc | 0 | 0 | 0 | 0 | 0 | 45 |

## Setup and use

Run commands from the project directory. The inspected environment uses Windows, Python 3.11.9, PyTorch 2.5.1+cu121, and torchvision 0.20.1+cu121. `requirements.txt` lists only direct dependencies at the installed versions. PyTorch pins omit the CUDA build suffix so a separate wheel index can select the hardware build.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This preparation did not test a clean installation or other platforms. The requirements are not a complete transitive dependency lock.

After placing the raw images at the paths recorded in `data/splits.csv`:

```powershell
python find_duplicates.py
python check_dataloader.py
python model.py
python train.py
python evaluate.py
python analyze_errors.py
python gradcam.py
```

The committed manifest preserves the evaluated split. To regenerate it from the original dataset, run `python split_data.py`; this overwrites `data/splits.csv`. Training writes or replaces `checkpoints/best_model.pt`. Evaluation and analysis require that checkpoint; it is deliberately excluded from the repository. Evaluation saves the confusion matrix, error analysis saves incorrect predictions, and Grad-CAM saves three sample comparisons under `results/`. Plotting scripts also open figures.

`explore_data.py` checks image counts, sizes, modes, and unreadable files. `visualize_samples.py` displays examples of each class.

## Limitations

These results describe one held-out NEU-CLS split and one trained checkpoint, not independent factory validation. Exact duplicates were removed before splitting, but near-duplicates and shared source pieces have not been audited or separated by group. The test set has already been examined; further model selection should use validation data.

Seed 42 is set for splitting and training, but deterministic GPU algorithms are not enforced. Retraining may produce different results across devices and software environments. The original training history was not saved to CSV, so per-epoch curves and repeated-run variation are unavailable.

Grad-CAM highlights regions associated with a selected class score; it does not establish a defect boundary or prove a causal explanation. Each heatmap is normalized independently.

## Repository contents

Python source files, this README, dependency pins, `.gitignore`, and the relative-path split manifest are included. Raw data, virtual environments, checkpoints, generated images and reports, and local credentials are excluded. No pretrained weights are distributed.
