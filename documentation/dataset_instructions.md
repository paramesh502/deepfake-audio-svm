# Dataset Access Instructions

## 1. ASVspoof 2019 Logical Access (Primary Training Dataset)

### Source
- Official website: https://www.asvspoof.org/index2019.html
- Database B (LA track) contains TTS and voice-conversion based spoofed speech

### Download Instructions
1. Visit https://www.asvspoof.org/index2019.html
2. Register and accept the terms of use
3. Download the LA partition (approximately 2.6 GB)
4. The dataset includes:
   - Training partition (2580 real + 22800 fake utterances)
   - Development partition (2548 real + 22296 fake utterances)
   - Evaluation partition (7355 real + 63882 fake utterances)
5. Protocol files are in: LA/ASVspoof2019_LA_cm_protocols/

### Label Mapping (from protocol files)
- "bonafide" -> 0 (REAL)
- "spoof"     -> 1 (FAKE)

### Protocol File Format
Each line: SPEAKER_ID FILENAME  - SYSTEM_ID  LABEL
Example: LA_0001 LA_T_1000137 - - bonafide

## 2. In-the-Wild Audio Deepfake Dataset (External Evaluation)

### Source
- https://deepfake-demo.aisec.fraunhofer.de/in_the_wild
- Real celebrity speeches + deepfake counterparts
- Useful for external generalization testing

### Label File
- meta.csv contains file paths and labels

## 3. Colab Setup Instructions

### Option A: Google Drive (Recommended)
1. Download datasets to your local machine
2. Upload to Google Drive folder:
   MyDrive/22MAT220_Deepfake_SVM/data/
3. Mount Drive in Colab (Cell 05)
4. Update configs/experiment_config.yaml with your Drive folder path

### Option B: Direct Colab Download
1. Run Cell 07 to trigger automatic download (requires registered credentials)
2. Datasets will be cached in /content/data/

### Option C: Quick Demo Mode
1. Set quick_demo.max_train_samples: 500 in experiment_config.yaml
2. This uses a small subset to test the pipeline without downloading the full dataset

## 4. Data Integrity Checks
The notebook performs:
- Label validation (verify real/fake counts)
- Duplicate file detection
- Sample rate inspection
- Duration distribution analysis
- Corrupted file detection
- Class balance reporting

## 5. License Notes
ASVspoof 2019 is released for research use. See the official terms at asvspoof.org.
In-the-Wild dataset: see Fraunhofer AISEC terms.
Do not redistribute dataset files. Cite properly in reports.
