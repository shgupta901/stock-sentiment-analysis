# 📊 Stock Market Sentiment Analysis

This project predicts **stock market sentiment (Up / Down)** from financial news headlines using **Natural Language Processing (NLP)** and **Machine Learning**.  

It combines text processing (TF-IDF, sentiment features) with classifiers (Logistic Regression / SVM), and provides an **interactive Streamlit dashboard**.

---

## 🚀 Features
- Predict sentiment for **single or batch news headlines**  
- Streamlit dashboard with **interactive graphs & probability bars**  
- Uses **TF-IDF + VADER + TextBlob** sentiment features  
- Visualization: prediction probabilities, class distribution, confusion matrix  
- Export predictions as CSV  

---

## 📂 Project Structure
stock-sentiment-project/
├── app.py # Streamlit dashboard
├── stock-market-sentiment-analysis.ipynb # Jupyter notebook
├── stock_sentiment_model2.pkl # Trained ML model (ignored in .gitignore)
├── stock_news.xlsx # Dataset (not pushed, too large)
├── requirements.txt # Dependencies
├── README.md # Project description
└── .gitignore # Ignore large/unnecessary files

---

## ⚙️ Installation & Usage

### 1. Clone the repository

git clone https://github.com/shgupta901/stock-sentiment-analysis.git
cd stock-sentiment-analysis

2. Install dependencies
pip install -r requirements.txt

3. Run the dashboard
streamlit run app.py

Input:  "Tesla rallies after record deliveries"
Output: "Prediction: Positive (Market Up)"
Confidence: 87%

📈 Example

Input:  "Tesla rallies after record deliveries"
Output: "Prediction: Positive (Market Up)"
Confidence: 87%

📊 Results

Validation Accuracy: ~85%

Confusion Matrix shows good separation between Up vs Down headlines

(Add screenshots of your Streamlit dashboard here)

📌 Notes

Dataset: ~26k finance news headlines (not uploaded due to size).

Pretrained model (.pkl) is excluded via .gitignore.

For educational/demo purposes only, not financial advice.


---

## 🔹 Step 2: Add `.gitignore`
Create a `.gitignore` file with:
*.pkl
*.csv
*.xlsx
pycache/
.DS_Store

→ This prevents pushing large datasets/models.

---

## 🔹 Step 3: Add `requirements.txt`
Make a file:
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
textblob
vaderSentiment
joblib


---

## 🔹 Step 4: Commit & Push
Run these commands:
```bash
git add README.md .gitignore requirements.txt
git commit -m "Added README, gitignore, and requirements"
git push

