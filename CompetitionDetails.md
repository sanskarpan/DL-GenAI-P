Competition Overview: Messy Mashup
The Messy Mashup competition focuses on robust music genre classification under realistic and noisy mixing conditions. Participants are provided with a curated training dataset consisting of songs from 10 distinct music genres, where each song is instrument-separated into four stems: drums.wav, vocals.wav, bass.wav, and others.wav. In addition, a separate dataset containing random noise sounds is supplied.

The core challenge lies in generalization. Instead of clean, original tracks, the test data is composed of mashups created by mixing instrument stems from different songs belonging to the same genre. To ensure musical coherence during mixing, some instrument tracks may undergo tempo adjustments so that all stems are rhythmically synchronized before being combined. To further increase complexity and simulate real-world audio conditions, random noise samples are added to these mashups at varying intensities and positions.

Description
Participants must design models capable of learning genre-specific musical characteristics that remain invariant to:

Cross-song stem recombination,
Tempo variations introduced during synchronization,
Instrument balance changes,
Additive environmental and synthetic noise. The goal is to predict the correct genre label for each noisy mashup. Success in this task requires effective audio representation learning, noise robustness, and the ability to capture high-level musical structure beyond individual instrument timbres. This competition emphasizes practical audio understanding and mirrors challenges encountered in real-world music information retrieval systems, such as remix analysis, noisy audio classification, and content-based music recommendation. The main challenge here is that the training dataset provided and the testing dataset follow different distributions and the participants must explore multiple data augmentation techniques and audio processing libraries such as librosa to get samples that match the test distribution.

Evaluation
Evaluation Metric
Submissions are evaluated using the Macro F1 Score across the 10 genre classes:

Macro F1 computes the F1 score independently for each genre and then averages them.
This metric treats all genres equally, making it well-suited for evaluating performance under potential class imbalance.
Higher macro F1 scores indicate better overall genre classification performance across all classes.

Submission Format
Participants must submit a CSV file with the following columns:

id: Unique identifier for each test mashup
genre: Predicted genre label
The genre column must contain one of the following values:
["blues", "classical", "country", "disco", "hiphop","jazz", "metal", "pop", "reggae", "rock"]

The file should contain a header and have the following format:
id,genre
0001,blues
0002,classical
0003,country
etc.




Dataset Description
The Messy Mashup dataset is organized into clean instrument stems, noise sources, and noisy mashup test samples. The dataset structure is shown below:

messy_mashup
├── ESC-50-master
├── genres_stems
├── mashups
├── sample_submission.csv
└── test.csv
1. genres_stems/ — Instrument-Separated Training Data
This directory contains the training dataset, organized by genre. There are 10 genre folders, each containing 100 songs, and each song is decomposed into four instrument stems:

drums.wav
vocals.wav
bass.wav
others.wav
The genres included are:

blues, classical, country, disco, hiphop,
jazz, metal, pop, reggae, rock
These clean stems allow models to learn genre-specific musical structure at the instrument level. Participants are encouraged to explore stem-wise feature extraction, cross-instrument fusion, or stem-aware data augmentation strategies.

2. ESC-50-master/ — Noise Dataset
This folder contains the ESC-50 environmental sound dataset, which is provided as an auxiliary resource for noise augmentation and robustness training.

Key contents include:

audio/: 2,000 labeled noise clips spanning 50 environmental sound classes
meta/esc50.csv: Metadata with class labels and categories
LICENSE and README.md: Dataset license and usage details
Noise samples from this dataset are used during mashup creation and may also be used by participants for training-time augmentation.

3. mashups/ — Test Audio Samples
The mashups directory contains 3,020 unlabeled audio files, which form the test set for the competition.

Each mashup is constructed by:

Selecting instrument stems from different songs of the same genre
Applying tempo adjustments where needed to ensure rhythmic synchronization
Mixing the synchronized stems into a single audio track
Adding one or more random noise samples from the ESC-50 dataset at random positions and intensities
The final mashups are intentionally noisy and musically varied, making genre classification significantly more challenging than standard clean-audio benchmarks.

4. test.csv — Test Index File
This file provides the mapping between test sample identifiers and mashup audio files. It includes:

id: Unique identifier for each mashup
Corresponding filenames in the mashups/ directory
Participants should generate predictions for every id listed in this file.

5. sample_submission.csv — Submission Template
A reference submission file demonstrating the required format for leaderboard evaluation.

It contains two columns:

id: Test sample identifier
genre: Predicted genre label
The genre column must contain one of the predefined class names.



# Milestones :
## Milestone 1 
Data Exploration & Baseline Submission

Perform EDA on the dataset (class distribution, audio length, gaps, ESC-50 noise).
Create a rule-based or random baseline submission on Kaggle.
Post questions about dataset, metrics (Macro F1), or unclear parts.

## Milestone 2 
Classical ML Baseline

Clean and preprocess audio stems.
Convert audio to numerical features (MFCCs, Spectrograms).
Train Logistic Regression, Naive Bayes, or boosting models (CatBoost, LightGBM, XGBoost).
Evaluate using Macro F1 and log results in W&B.

## Milestone 3 
Your First Neural Network & CNNs!

Learn PyTorch basics: Tensors, Dataset (custom loader for training), DataLoader.
Convert audio to 2D/1D Mel-Spectrograms.
Build a simple CNN (Convolutional Neural Network)/NN (Neural Network) to process the spectrograms.
Implement training loop, loss, optimizer, and wandb logging.
Train and evaluate your CNN/NN (Neural Network)model.

## Milestone 4 
Fine-Tuning Pre-trained Transformers

Learn transfer learning with Audio Transformers (AST/Hubert).
Use Hugging Face feature extractors and audio models.
Fine-tune on the noisy genre mashup dataset.
Compare with your other models like CNN/CRNN results and log final performance.

## Milestone 5 
Final Submission & Presentation

Make final Kaggle submission.
Create report & Present results and analysis (Macro F1, Error Analysis, Insights).
Deploy model using Streamlit/Gradio.
