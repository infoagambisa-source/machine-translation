# Machine Translation with Seq2Seq (Keras)

This project implements a **sequence-to-sequence (seq2seq) neural network** for machine translation using **Keras and TensorFlow**.  
The model is trained to translate sentences from **English to Korean** using an encoder–decoder architecture with LSTM layers.

This work is based on an off-platform extension of a Codecademy deep learning project and is designed to run locally using **CPU-based TensorFlow**.

---

## 📌 Project Overview

The pipeline consists of three main stages:

1. **Preprocessing**
   - Load parallel sentence data
   - Tokenize text
   - Build vocabularies
   - Convert sentences into one-hot encoded tensors

2. **Training**
   - Encoder–decoder LSTM model
   - Trained using categorical cross-entropy
   - Model saved as `training_model.h5`

3. **Inference / Testing**
   - Load the trained model
   - Reconstruct encoder and decoder for inference
   - Translate unseen sentences from the dataset

---

## 📂 Project Structure

```text
machine-translation-seq2seq/
│
├── preprocessing.py        # Data loading and preprocessing
├── training_model.py       # Seq2Seq training model
├── test_function.py        # Inference and translation
├── kor.txt                 # English–Korean parallel dataset
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## 🧠 Model Architecture

### Encoder
- LSTM layer
- Outputs hidden and cell states

### Decoder
- LSTM layer initialized with encoder states
- Dense softmax output layer for token prediction

### Training
- Optimizer: RMSprop
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/infoagambisa-source/machine-translation.git
cd <repo-name>
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Preprocess the data
```bash
python preprocessing.py
```

This step:
- Loads `kor.txt`
- Builds vocabularies
- Creates encoder and decoder input matrices

---

### Step 2: Train the model
```bash
python training_model.py
```

This step:
- Trains the seq2seq model
- Saves the trained model to `training_model.h5`

⚠️ **Note:** Training can take **20 minutes to several hours** depending on:
- Dataset size  
- Number of epochs  
- CPU performance  

---

### Step 3: Test translations
```bash
python test_function.py
```

This step:
- Loads the trained model
- Translates sample sentences
- Prints input vs. decoded output to the terminal

---

## 📊 Dataset

- **Source:** Parallel English–Korean sentence pairs
- **Format:**
```text
English sentence<TAB>Korean sentence
```

- The number of training lines can be adjusted in `preprocessing.py`

---

## 🚀 Future Improvements

Planned or possible enhancements:

- Separate training and inference logic more cleanly
- Support translation of new, unseen input sentences
- Handle unknown / out-of-vocabulary tokens
- Replace one-hot encoding with word embeddings
- Experiment with:
  - Larger datasets
  - Different latent dimensions
  - Attention mechanisms

---

## 🧪 Notes

- This project currently uses **CPU-based TensorFlow**
- GPU acceleration is recommended for faster training
- Large datasets may exceed memory limits on low-resource machines

---

## 📚 Acknowledgements

- Codecademy — Deep Learning & NLP curriculum  
- TensorFlow / Keras documentation  

---

## 📄 License

This project is for **educational purposes**.
