# [cite_start]MediScopeDiffusion: A Diffusion Network for 3D Medical Image Interpretation [cite: 534]

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[cite_start]This repository contains the official implementation for the project **MediScopeDiffusion**, a novel 3D latent diffusion framework designed for robust, full-volume classification of medical CT scans[cite: 534, 566]. [cite_start]This work addresses the limitations of standard 2D-based approaches [cite: 564] [cite_start]by leveraging a guided diffusion process to learn rich spatial and structural features for improved diagnostic accuracy[cite: 583].


---

## 📋 Table of Contents

-   [Problem Statement](#-problem-statement)
-   [Our Solution: MediScopeDiffusion](#-our-solution-mediscopediffusion)
    -   [Phase 1: DCG Pre-training](#phase-1-dcg-pre-training)
    -   [Phase 2: Guided Diffusion Training](#phase-2-guided-diffusion-training)
-   [Results & Progress](#-results--progress)
-   [Getting Started](#-getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Dataset](#dataset)
    -   [How to Run the Notebook](#how-to-run-the-notebook)
-   [Project Team](#-project-team)
-   [License](#-license)

---

## 🩺 Problem Statement

[cite_start]Automated 3D medical scan analysis (CT, MRI) is crucial for healthcare, but current deep learning models face significant challenges with robustness[cite: 559, 561].

[cite_start]**Current Challenges**[cite: 560]:
* [cite_start]**Noise & Artifacts:** Scans are often noisy, which degrades model performance[cite: 562].
* [cite_start]**High Inter-class Similarity:** Different pathologies can appear visually similar[cite: 563].
* [cite_start]**2D Limitations:** Analyzing 2D slices individually fails to capture complex 3D spatial relationships[cite: 564].

**Our Research Gap:**
[cite_start]There is **no dedicated 3D latent diffusion framework** that exists specifically for the task of *volumetric medical image classification*[cite: 566].

---

## 💡 Our Solution: MediScopeDiffusion

[cite_start]Instead of using diffusion models for image generation, we adapt their powerful feature-learning capability for a classification task[cite: 612]. [cite_start]The model is forced to learn a deep structural understanding of the data by learning to reverse a noise-addition process[cite: 587, 607].

Our methodology, detailed in the `dffmet3d.ipynb` notebook, is a two-phase process:

### Phase 1: DCG Pre-training
A simplified **Dual-Channel Guidance (DCG)** model (approx. 457k parameters) is first trained to act as a strong, lightweight feature extractor. It combines two priors:

* [cite_start]**Global Prior ($\hat{y}_g$):** Captures the overall, coarse anatomical context from the entire scan[cite: 667].
* [cite_start]**Local Prior ($\hat{y}_l$):** Captures fine-grained details by focusing on salient Regions of Interest (ROIs) identified by a saliency map[cite: 668, 673].

### Phase 2: Guided Diffusion Training
[cite_start]The pre-trained DCG model then provides guidance to the main 3D U-Net diffusion model[cite: 737]. This process is steered by two key components derived from the DCG's outputs:

* [cite_start]**Dense Guidance Map (M):** A learned 3D map that indicates whether the diffusion model should trust the global or local prior at different spatial locations[cite: 701].
* [cite_start]**Image Feature Prior (F):** A fused feature vector that provides rich contextual information to the U-Net by capturing both global and local structures through attention[cite: 704].

[cite_start]The entire diffusion process operates in a compressed **Latent Space** to ensure it remains computationally efficient and can run on standard hardware[cite: 735, 739].

---

## 📊 Results & Progress

[cite_start]As outlined in our project presentation, our progress is as follows[cite: 751]:

* [cite_start]**Architecture Design:** The end-to-end framework has been fully designed, including the 3D Dense Guidance Map (M), Feature Prior (F), and the Heterologous Diffusion Process[cite: 754]. The model architecture has been simplified to ~457k parameters to better suit the dataset size.
* [cite_start]**Dataset Preparation:** A dataset has been prepared using radiological findings from CT scans to classify the presence of viral pneumonia[cite: 755].
* **Implementation:** The complete, corrected pipeline is implemented in the `dffmet3d.ipynb` notebook.
* [cite_start]**Next Steps:** The immediate next steps are to complete the full training run and benchmark the results against state-of-the-art classifiers[cite: 758, 759].

---

## 🚀 Getting Started

You can run this project using the provided Kaggle notebook.

### Prerequisites

* Python 3.10+
* PyTorch
* TensorFlow (for Keras data utilities)
* NiBabel (for loading `.nii` files)
* Scikit-learn
* Seaborn

### Dataset

The dataset used in this project is based on the **3D image classification** in keras documentation. It consists of 100 normal (CT-0) and 100 abnormal (CT-23) CT scans.

| Dataset | Label | Download Link (Please upload and replace) |
| :--- | :---: | :--- |
| Normal (CT-0) | 0 | `[PASTE GITHUB RELEASE LINK FOR CT-0.zip HERE]` |
| Abnormal (CT-23) | 1 | `[PASTE GITHUB RELEASE LINK FOR CT-23.zip HERE]` |

### How to Run the Notebook

1.  **Open `dffmet3d.ipynb`** in a Jupyter or Kaggle environment with GPU acceleration enabled.
2.  **Cell 1 (Setup):** Installs libraries and downloads the dataset from the original GitHub source.
3.  **Cell 2 (Preprocessing):** Loads, normalizes, and resizes all `.nii` scans to the target shape of (128, 128, 64). It also applies balanced augmentations.
4.  **Cell 3 (Model Definition):** Defines the simplified DCG Model architecture (~457k parameters).
5.  **Cell 4 (Diffusion Components):** Defines the U-Net, Priors, and Diffusion Process components.
6.  **Cell 5 (Training):** This is the main training cell. It runs the two-phase pipeline:
    * **Phase 1:** Pre-trains the DCG model for 50 epochs.
    * **Phase 2:** Freezes the DCG and trains the Diffusion model for 100 epochs.
7.  **Cell 6 (Evaluation):** Loads the best-saved models (`best_dcg_model.pth` and `best_diffusion_model2.pth`) and runs a comprehensive evaluation on the validation and test sets, generating performance metrics and plots.

**Note on Notebook Error:** The final cell (`cell 23`) in the provided notebook fails with a `RuntimeError`. This is because the simplified model defined in `cell 10` (457k params) does not match the architecture of the saved model (`best_dcg_model.pth`) it's trying to load, which was likely from an older, larger version of the notebook. To fix this, you must **run the training (Cell 5) to generate new model weights** *before* running the evaluation (Cell 6).

---

## 🧑‍💻 Project Team

### Students
* [cite_start]Fahad Ahmad (BT23CSD035) [cite: 539, 540]
* [cite_start]Ansh Bharadwaj (BT23CSD063) [cite: 541, 542]
* [cite_start]Anuj Soni (BT23CSD065) [cite: 543, 544]
* [cite_start]Yash Kumar Saini (BT23CSD066) [cite: 545, 546]

### Supervisors
* [cite_start]Mr. Pravin S. Bhagat [cite: 536]
* [cite_start]Ms. Nayna Potdukhe [cite: 537]

[cite_start]**Institution:** Indian Institute of Information Technology, Nagpur (IIITN) [cite: 531]

---

## 📄 License

This project is licensed under the MIT License.
