# Viva Questions and Answers
# 22MAT220 — Deepfake Audio Detection Using SVM

## 1. Mathematical Foundations

Q: What is the geometric interpretation of the SVM margin?
A: The margin is the perpendicular distance between the two parallel hyperplanes
   y=+1 (w^T x + b = 1) and y=-1 (w^T x + b = -1). Its value is 2/||w||.
   SVM maximises this margin, which corresponds to minimising ||w||^2.

Q: Why does minimising ||w||^2 maximise the margin?
A: The distance between the two margin hyperplanes is 2/||w||. Minimising ||w||^2
   is equivalent to maximising 2/||w||, the geometric margin.

Q: What is the hinge loss and why is it used?
A: Hinge loss L(y, f) = max(0, 1 - y*f(x)) is the convex relaxation of the
   0-1 loss. It equals 0 for correctly classified points outside the margin,
   and grows linearly for margin violations. It is convex (enabling gradient-based
   optimisation) and encourages large-margin solutions.

Q: What are support vectors?
A: Support vectors are training samples that lie on or inside the margin
   (margin value <= 1). They are the ONLY points that influence the decision
   boundary. Removing non-support vectors would not change the solution.
   For soft-margin SVMs, samples with alpha_i = C are margin violators.

Q: Why use subgradient descent instead of gradient descent?
A: The hinge loss is not differentiable at y_i f(x_i) = 1. The subgradient
   is any element of the subdifferential set at that point. Subgradient descent
   converges for convex objectives even without differentiability.

Q: Explain the role of C in SVM.
A: C controls the trade-off between margin width and training error.
   - Small C: wide margin, more misclassifications allowed (underfitting risk)
   - Large C: narrow margin, fewer misclassifications (overfitting risk)
   - C -> infinity: hard-margin SVM (requires separable data)

Q: What is the dual formulation and why is it useful?
A: The Lagrangian dual replaces (w, b, xi) with Lagrange multipliers alpha_i.
   The dual has n variables (one per sample) versus d+1 (d = feature dimension)
   in the primal. The dual is useful when n << d, and critically, enables the
   kernel trick: replace x_i^T x_j with K(x_i, x_j).

Q: Explain the kernel trick.
A: The kernel function K(x_i, x_j) = phi(x_i)^T phi(x_j) computes inner
   products in a high-dimensional feature space without explicitly computing phi.
   The RBF kernel K(x_i,x_j) = exp(-gamma||x_i - x_j||^2) corresponds to an
   infinite-dimensional feature map (Gaussian RKHS).

## 2. Audio Processing

Q: Why are MFCCs useful for speech classification?
A: MFCCs capture the spectral envelope of speech using a mel-scale filterbank
   (which mimics human auditory perception) followed by DCT. They are compact,
   decorrelated, and sensitive to speaking style and vocal tract characteristics.
   Synthetic speech often has abnormal MFCC patterns due to vocoder artefacts.

Q: What is the difference between MFCC and LFCC?
A: MFCC uses a mel-scale (logarithmic) filterbank; LFCC uses a linear-frequency
   filterbank. The linear scale preserves high-frequency information that the mel
   scale compresses. For deepfake detection, LFCC often captures vocoder
   artefacts in the high-frequency region that MFCC misses.

Q: Why do we use delta and delta-delta features?
A: Delta features capture temporal dynamics (how features change over time).
   Delta-delta captures acceleration. These are important because natural speech
   has smooth, human-controlled temporal transitions, while synthetic speech may
   have discontinuities.

Q: What is Platt scaling and why is it needed?
A: Raw SVM decision scores are unbounded real numbers with no probabilistic
   interpretation. Platt scaling fits P(y=1|f) = sigmoid(A*f + B) on a held-out
   calibration set. This converts scores to calibrated probabilities. Without
   calibration, reporting percentages would be misleading.

## 3. Experimental Design

Q: Why is a separate calibration set needed?
A: Calibrating on training data would over-fit the sigmoid parameters to the
   training distribution, producing overconfident probabilities. Using a
   held-out set ensures the calibrated probabilities reflect true accuracy.

Q: What is the generalization gap and why does it occur?
A: The generalization gap is the difference in performance between the internal
   test set and an external dataset. It occurs because: (1) the external dataset
   may contain deepfake generators not present in training, (2) different recording
   conditions, codecs, or acoustic environments, (3) overfitting to dataset-specific
   properties (speaker identity, file format, sample rate).

Q: Why test on external data?
A: Internal test sets from the same dataset as training may share distributional
   properties (same speakers, same generators, same recording conditions) that do
   not hold for real-world audio. External evaluation provides honest evidence of
   generalisation.

Q: What is the Equal Error Rate (EER)?
A: EER is the operating point where the False Acceptance Rate (FAR = FP/(FP+TN))
   equals the False Rejection Rate (FRR = FN/(FN+TP)). Lower EER = better
   discrimination. It is the standard evaluation metric for ASVspoof benchmarks.

## 4. Practical Questions

Q: How does the system handle audio it has never seen?
A: The system is content-agnostic: it extracts acoustic features from any input
   audio, regardless of its filename or source. The inference pipeline processes
   any .wav/.mp3/.flac file. Performance on novel generators depends on how
   different their acoustic properties are from training data.

Q: Can the system detect all deepfakes?
A: No. The system was trained on specific generators (ASVspoof 2019 attack types).
   Novel generators using different vocoders, neural architectures, or voice
   conversion methods may produce artefacts not present in training data. This
   is explicitly reported as a limitation.

Q: Why use SVM and not a neural network?
A: This project is for Mathematics for Computing. SVM has clear mathematical
   foundations rooted in convex optimisation, margin theory, and kernel methods
   — directly aligned with the course objectives. SVMs are interpretable,
   do not require large datasets, and their generalisation is theoretically
   grounded (VC dimension theory). Neural networks are discussed as future work.

Q: How does the website work in Colab?
A: Gradio creates a Python-based web server. The launch() call with share=True
   creates a tunnel to a public URL via Gradio-s server. The Colab runtime
   acts as the backend. The session ends when the Colab runtime disconnects.
