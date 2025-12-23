# fake-news-detection-
📰 Fake News Detection using Machine Learning
📌 Project Overview

Fake news has become a serious issue in the digital era, spreading misinformation rapidly through online platforms.
This project aims to detect whether a news article is REAL or FAKE using Machine Learning and Natural Language Processing (NLP) techniques.

The model analyzes textual data and predicts the authenticity of news articles with good accuracy. A simple Streamlit web interface is used for user interaction.

🎯 Objectives

Identify fake and real news articles

Apply NLP techniques for text preprocessing

Train and evaluate a machine learning model

Deploy the model using Streamlit

🛠️ Technologies Used

Programming Language: Python

Libraries & Frameworks:

Pandas

NumPy

Scikit-learn


Streamlit

Pickle / Joblib

IDE: VS Code / PyCharm / Jupyter Notebook

📂 Project Structure
Fake_News_Detection/
│
├── dataset/
│   └── news.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── app.py                # Streamlit application
├── train_model.py        # Model training script
├── requirements.txt      # Required libraries
└── README.md             # Project documentation

🔍 Dataset Description

The dataset contains news articles labeled as FAKE or REAL.The model is trained on a limited dataset, and predictions depend on the available data; scaling to larger datasets is required for real-world applications.

Main features:

title

text

label

⚙️ How It Works

Text data is cleaned and preprocessed

TF-IDF Vectorizer converts text into numerical form

Machine learning model is trained on labeled data

The trained model predicts whether news is fake or real

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/Fake_News_Detection.git
cd Fake_News_Detection

2️⃣ Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py

🖥️ Application Output

User enters a news article

Click Predict

Output displays:

✅ Real News

❌ Fake News

📊 Model Performance

Accuracy: ~85–90% (depends on dataset and model)

Evaluation Metrics:

Accuracy Score

Confusion Matrix

🚀 Future Enhancements

Use deep learning models (LSTM, BERT)

Improve UI design

Add multilingual support

