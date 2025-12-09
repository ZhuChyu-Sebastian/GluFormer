# GluFormer

**Official implementation of GluFormer: A Foundation Model for Continuous Glucose Monitoring**

**GluFormer** is a generative foundation model for Continuous Glucose Monitoring (CGM) data. Trained using self-supervised learning on over **10 million glucose measurements** from 10,812 adults, GluFormer learns latent representations of glycemic dynamics that generalize across diverse populations and devices.

This repository contains the official implementation of the paper. The model uses autoregressive prediction to learn representations.

---

## 🌟 Key Capabilities

GluFormer is designed to serve as a generalizable backbone for metabolic health AI. Its core capabilities include:

**Foundation Model Pretraining:**

Trained on a massive dataset of mostly non-diabetic adults.

Generalizes across 19 external cohorts, spanning 5 countries, 8 different CGM devices, and diverse metabolic states (T1D, T2D, GDM, Obesity).

**Risk Stratification & Prediction:**

Diabetes Risk: Outperforms baseline HbA1c in predicting the progression from prediabetes to diabetes.

Long-term Outcomes: Identified individuals at high risk for cardiovascular mortality and incident diabetes in a cohort with an 11-year follow-up.

Clinical Trials: Improves outcome forecasting for dietary and pharmacologic interventions.

**Generative Modeling:**

Autoregressively generates physiologically plausible synthetic CGM trajectories (24h horizons).

Supports imputation of missing data segments.

**Multimodal Integration:**

Includes a multimodal extension that integrates dietary tokens (macronutrients) to predict individual glycemic responses to food.


## 📂 Repository Structure

The codebase is organized into modular components reflecting the pipeline described in the publication:

```text
GluFormer/
├── create_data_as_tokens/   # Scripts to process raw CGM data into tokenized sequences
├── train_model/             # Logic for pre-training the Transformer and fine-tuning
├── model_usage/             # Tools for inference, embedding extraction, and evaluation
├── demo/                    # Examples and visualization
├── LMRL_Poster/             # Supplementary figures and materials from conference presentations
└── README.md

```

-----

## 💻 System Requirements

This codebase was developed and tested using **Python 3.10**.

**Hardware:**
The model was trained using NVIDIA GPUs (**A40 / A100**).

**Dependencies:**
To ensure reproducibility, we recommend using the specific versions used during development:

  * `torch==2.3.1`
  * `torchaudio==2.3.1`
  * `torchvision==0.18.1`
  * `torchelastic==0.2.2`
  * `numpy==1.26.4`
  * `pandas==2.2.2`
  * `scikit-learn==1.5.0`
  * `scipy==1.14.0`
  * `seaborn==0.13.2`
  * `matplotlib==3.9.0` (with `matplotlib-inline==0.1.6`)
  * `wandb==0.17.3`
  * `umap-learn==0.5.5`
  * `tqdm==4.66.4`

-----

## ⚙️ Installation

We recommend creating a dedicated Conda environment to manage dependencies.

**1. Clone the repository:**

```bash
git clone [https://github.com/Guylu/GluFormer.git](https://github.com/Guylu/GluFormer.git)
cd GluFormer
```

**2. Create and activate environment:**

```bash
conda create -n gluformer python=3.10
conda activate gluformer
```

**3. Install dependencies:**
You can install the required packages using pip.
*Note: Installation time is standard.*

```bash
# Core PyTorch installation
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install remaining dependencies
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.0 scipy==1.14.0 \
seaborn==0.13.2 matplotlib==3.9.0 matplotlib-inline==0.1.6 \
wandb==0.17.3 umap-learn==0.5.5 tqdm==4.66.4 torchelastic==0.2.2
```

-----

## 🚀 Usage

The workflow for GluFormer consists of three main steps handled by the directories above:

### 1\. Data Tokenization (`create_data_as_tokens/`)

Raw CGM data (mg/dL or mmol/L) must be converted into discrete tokens before entering the model. This directory contains the logic to map continuous glucose values into the vocabulary used by the Transformer.

### 2\. Model Training (`train_model/`)

This module handles the self-supervised pretraining loop. It uses an autoregressive objective (next-token prediction) to learn temporal glucose dynamics.

### 3\. Inference and Usage (`model_usage/` & `demo/`)

Once trained, the model can be used to generate synthetic trajectories or extract embeddings. 

-----

## 🧪 Downstream Tasks

GluFormer is designed to support various clinical and machine learning tasks. The `model_usage` folder facilitates the following applications described in the paper:

  * **Glucose Forecasting & Generation:**
    The model can autoregressively generate physiologically plausible glucose trajectories based on a context window. This is used to impute missing data or simulate future glucose excursions.

  * **Embedding Extraction:**
    The model's primary power lies in its latent space. By extracting embeddings from the last hidden layer, users can obtain a dense vector representation of a patient's glycemic state. In the paper, these embeddings were used to:

      * Predict HbA1c levels.
      * Stratify patients by risk of developing Type 2 Diabetes.
      * Identify individuals at high risk for cardiovascular mortality.

  * **Multimodal Integration:**
    The architecture supports the integration of dietary tokens alongside glucose tokens, allowing for the assessment of individual glycemic responses to food intake.

-----

## Misc.

Instalation time: standart

Trained useing NVIDIA GPUS (A40 & A100)

Status: Adding code is currently in progress.
 
## 📚 Citation

