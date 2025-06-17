# MBTI_AI-ML

Download dataset from [https://www.kaggle.com/datasets/datasnaek/mbti-type]
Download Best Model (Google Drive) [https://drive.google.com/file/d/13raKrQXVd31nrMWy4JP8IE5N4OvJV1As/view?usp=sharing]

# 🧠 MBTI Personality Classifier using BERT

This project fine-tunes a BERT-based model to classify user-generated text into MBTI (Myers–Briggs Type Indicator) personality types. It includes training, evaluation, visualizations, and an interactive Streamlit demo app.

---

## 📁 Project Structure

| File/Folder                      | Description |
|----------------------------------|-------------|
| `MBTI_Classifier_Training.ipynb` | Jupyter notebook used for training the model |
| `app.py`                         | Script for running predictions using a trained model |
| `streamlit_app.py`               | Streamlit app for interactive inference |
| `tokenizer/`                     | Tokenizer files saved from HuggingFace tokenizer |
| `model_config.json`              | Configuration for the BERT model |
| `evaluation_results.json`        | Stores final evaluation metrics (accuracy, F1-score, etc.) |
| `*.png`                          | Various plots (accuracy, loss, confusion matrices, distributions) |
| `README.md`                      | Project documentation (you’re reading it!) |

---

## 🚀 Features

- Fine-tuned BERT for MBTI classification
- HuggingFace tokenizer and model integration
- Evaluation: accuracy, confusion matrix, class distribution
- Streamlit app for interactive text prediction
- Clean visualizations and training logs

---

## ⚙️ Installation

> 💡 Recommended: Use a virtual environment

```bash
pip install -r requirements.txt
If you don’t have a requirements.txt, install dependencies manually:
pip install torch transformers scikit-learn matplotlib pandas streamlit

🧠 Training the Model
To train the model, run the notebook:
jupyter notebook MBTI_Classifier_Training.ipynb

This will:
Tokenize the input text
Train a BERT classifier
Save training logs and model state
Output evaluation metrics and plots

📊 Evaluation Results
accuracy_plot.png & loss_plot.png: Training history
test_confusion_matrix.png & validation_confusion_matrix.png: Evaluation on test/validation data
class_distribution_before.png & class_distribution_after.png: Class balance visualization
evaluation_results.json: Contains test accuracy, precision, recall, F1-score

🌐 Run the Web App
➤ Using Streamlit
To launch the interactive web app locally:
streamlit run streamlit_app.py

➤ Using Standard Python Script
If you prefer a simple terminal-based interaction:
python app.py

📦 Model Files
Trained model weights (.pt) are not included in this repo due to size limits.
You can download the best checkpoint from:

📥 Download best_model.pt (Google Drive)

Place the downloaded file in the root directory or modify the load path in app.py and streamlit_app.py.

🔧 Configuration
Model parameters and tokenizer settings are stored in:

model_config.json

tokenizer/ folder (contains vocab/config files)

These help ensure reproducibility when loading the trained model for inference.

📸 Demo
Input Text	Predicted MBTI
"I love planning social events and helping others grow."	ENFJ
"I enjoy staying in, thinking about abstract concepts and learning."	INTP

🙋‍♂️ Author
Muhammad Umer
📫 [i222578@nu.edu.pk]
