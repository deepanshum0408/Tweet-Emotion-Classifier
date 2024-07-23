# Tweet-Emotion-Classifier

This project focuses on classifying emotions from tweets using various machine learning models, including fine-tuned BERT, custom BERT, Random Forest, SVM, Logistic Regression, and Naive Bayes. The project includes a Streamlit web application for easy interaction and visualization of the classification results.

## Overview

The primary goal of this project is to classify emotions into categories such as sadness, joy, love, anger, and fear. The models have been trained and evaluated to provide accurate predictions based on the input text.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/emotion-classification.git
    cd emotion-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Models and Performance

Below is the table of performance metrics for each model evaluated on the validation dataset:

| Model                                | Train Accuracy | Test Accuracy | F1 Score (Test Data) | Precision (Test) Data) | Recall (Test Data) |
|--------------------------------------|----------------|---------------|----------------------|------------------------|---------------------|
| Custom BERT Model + Fine-tuned       | 0.881          | 0.788         | 0.606148             | 0.820222               | 0.598914            |
| Pretrained BERT Model + Fine-tuned   | 0.89           | 0.79          | 0.93                 | 0.84                   | 0.56                |
| Naive Bayes                          | 0.67           | 0.60          | 0.60                 | 0.76                   | 0.67                |
| Random Forest                        | 0.867          | 0.75          | 0.84                 | 0.86                   | 0.82                |
| SVM                                  | 0.88           | 0.78          | 0.84                 | 0.87                   | 0.82                |
| Logistic Regression                  | 0.85           | 0.76          | 0.79                 | 0.87                   | 0.74                |

## GUI Snapshot
![Screenshot (444)](https://github.com/user-attachments/assets/3ef1a8c9-da91-48ea-8d1a-302f6ef54f98)


![Screenshot (445)](https://github.com/user-attachments/assets/15af7b6d-aec5-4542-b4d5-a01db44714dd)



## Project Information

### Fine-tuned BERT Model

The fine-tuned BERT model has been trained on the dataset to classify emotions. The model and tokenizer are loaded and used for prediction in the Streamlit app.

### Custom BERT Model

The custom BERT model, which has been pre-trained on a different dataset, is also included in the project. This model is used as an alternative for emotion classification.

### Random Forest Model

The Random Forest model uses a TF-IDF vectorizer for feature extraction. This model provides a simple yet effective way of classifying emotions.

### SVM Model

The Support Vector Machine (SVM) model also uses a TF-IDF vectorizer. It is known for its robustness in handling high-dimensional data.

### Logistic Regression Model

The Logistic Regression model, combined with a TF-IDF vectorizer, offers another method for classifying emotions from text.

### Naive Bayes Model

The Naive Bayes model, using a TF-IDF vectorizer, provides a probabilistic approach to emotion classification.

### Streamlit Application

The Streamlit app allows users to input a tweet and select a model for emotion classification. The app displays the predicted emotion and the confidence score.

<h2 id="contact">Contact</h2>
  <p>For any inquiries or feedback, please contact:</p>
  <ul>
    <li><strong>Name:</strong> Deepanshu Miglani</li>
    <li><strong>Education:</strong> B.tech CSE(AIML) , UPES, Dehradun</li>
    <li><strong>Email:</strong> deepanshumiglani0408@gmail.com / Deepanshu.106264@stu.upes.ac.in</li>
    <li><strong>GitHub:</strong> <a href="https://github.com/deepanshum0408">deepanshum0408</a></li>
  </ul>
  
  <h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
  
