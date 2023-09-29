# Personalized Medicine Recommending System

# Drug Review Rating Prediction

### Overview
This project aims to predict the ratings of drug reviews using machine learning. The dataset used for this task is obtained from the UCI Machine Learning Repository, specifically the "drugsCom" dataset, which contains text reviews of various drugs along with corresponding ratings.

## Getting Started

### Prerequisites

To run this project, you will need:

- Python 3.x
- Libraries: pandas, requests, scikit-learn, joblib

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Akhilesh2921/CodeClauseInternship_Personalized_Medicine_Recommending_System.git
   ```

2. Install the required libraries:

   ```bash
   pip install pandas scikit-learn joblib
   ```

3. Run the code as described in the following sections.

## Usage

### Data Retrieval

The project retrieves the dataset from the UCI ML Repository. You can find the dataset URL in the code.

```python
# Specify the UCI ML Repository URL for the dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
```

### Data Processing

The code performs the following data processing steps:

- Downloads the dataset in ZIP format.
- Extracts the contents.
- Loads the data into a Pandas DataFrame.
- Splits the data into training and testing sets.
- Performs TF-IDF vectorization on the text reviews.

### Model Training

A Random Forest Classifier is used for this project. The model is trained on the training data.

```python
# Create a machine learning model (Random Forest Classifier, for example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train_tfidf, y_train)
```

### Model Evaluation

The model is evaluated using accuracy as the metric.

```python
# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model (you can use other metrics as well)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Model Saving

The trained model is saved to a file using joblib.

```python
# Save the trained model to a file using joblib or another preferred method
model_filename = 'personalized_medicine_model.pkl'
joblib.dump(clf, model_filename)
print(f"Model saved as {model_filename}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) for providing the dataset.
- [scikit-learn](https://scikit-learn.org/stable/) for the machine learning tools.

## Author

- [Your Name](https://github.com/Akhilesh2456)

Feel free to reach out for any questions or improvements!
```

Remember to replace "Your Name" with your actual name and update any additional project-specific information as needed.
