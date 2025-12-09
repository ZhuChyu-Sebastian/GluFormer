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



------


## 📊 Model Overview & Results

### 1. The Core Concept: Language Modeling for Metabolism
GluFormer treats continuous glucose monitoring data similarly to how Large Language Models (LLMs) treat text. By training on a massive cohort of over 10,000 individuals, the model utilizes a Transformer architecture to perform "next-token prediction" on glucose measurements. This approach allows it to learn the grammar of glucose dynamics without explicit labeling.

![Main Idea](figures/1.png)

### 2. Generative Architecture
The model functions as an autoregressive generator. It takes a context window of past glucose levels (tokenized) and predicts the future trajectory. This generative capability allows for the creation of physiologically plausible synthetic data and the imputation of missing segments in CGM traces.

![Architecture and Generation](figures/2.png)

### 3. A Meaningful Latent Space
GluFormer is not just a predictor; it is an encoder of metabolic health. The embeddings (latent representations) learned by the model naturally organize themselves by physiological phenotypes. As seen below, the representation space automatically clusters individuals based on postprandial (post-meal) responses and fasting glucose levels without being explicitly supervised to do so.

![Latent Space Phenotypes](figures/3.png)

### 4. Robust Generalization
A key strength of GluFormer is its ability to generalize. The embeddings capture a "manifold of glycemic health" that remains robust across:
* **Diverse Pathologies:** Healthy, Pre-diabetic, T1D, T2D, GDM, and cancer patients.
* **Geography & Hardware:** 19 external cohorts across 5 countries and 8 different CGM devices.

![Generalization Map](figures/4.png)

### 5. State-of-the-Art Performance
When benchmarked against standard clinical metrics, GluFormer demonstrates superior predictive power. It achieves higher ROC AUC scores for diabetes prediction across multiple time horizons (At collection, 2-year, and 4-year), significantly outperforming the standard composite CGM scores and the Glucose Management Indicator (GMI).

![Benchmarking Performance](figures/5.png)

### 6. Forecasting Disease Trajectories
For individuals already at risk (pre-diabetic), GluFormer can distinguish between those likely to deteriorate and those likely to improve. In the figure below, the top quartile (Q4) of GluFormer risk scores correctly identified patients whose HbA1c would rise over a 2-year follow-up, whereas standard HbA1c quartiles failed to show significant differentiation.

![Trajectory Forecasting](figures/6.png)

### 7. Long-Term Diabetes Prediction (12-Year Horizon)
The model's predictive power extends far into the future. In a 12-year longitudinal study, GluFormer successfully stratified patients into distinct risk groups for developing diabetes ($p = 2.3 \times 10^{-6}$). In comparison, the clinical gold standard (HbA1c) showed no significant predictive power for diabetes development over this timeframe.

![Long Term Diabetes Prediction](figures/7.png)

### 8. Mortality Risk Prediction
The same predictive capability holds for cardiovascular outcomes. GluFormer embeddings successfully ranked individuals by risk of cardiovascular-related death over a decade-long period. High-risk groups identified by GluFormer showed a steep cumulative death event curve, while HbA1c-based ranking was non-significant.

![Mortality Prediction](figures/8.png)

### 9. Multimodal Integration (Diet)
GluFormer supports multimodal inputs. By integrating dietary tokens (carbohydrates, proteins, fats) alongside glucose tokens, the model can simulate individual glycemic responses to specific foods. This reduces the Mean Absolute Error (MAE) of predictions and enables "Digital Twin" simulations to test how a patient might react to different meals (e.g., Pizza vs. Salad).

![Multimodal Diet Integration](figures/9.png)

---------

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

