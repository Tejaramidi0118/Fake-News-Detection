# Fake News Detection

## Politifact Fact-Checking Project

This project aims to analyze and model fact-checking data from PolitiFact. It includes two machine learning models for classifying statements, along with the dataset used for training and evaluation.

### Project Structure

* **`politifact_factcheck_data.csv`**: This file is a dataset containing information about various statements that have been fact-checked. It includes columns such as the `verdict` (e.g., true, false, mostly-true), `statement_originator`, `statement`, `statement_date`, `statement_source`, `factchecker`, `factcheck_date`, and a link to the `factcheck_analysis_link`. The dataset has 21,152 rows and 8 columns. A significant portion of the data, specifically 11,795 entries, comes from "news," "blog," or "social_media" sources.
* **`Politifact_Model-2Classes.ipynb`**: This Jupyter Notebook contains code for a machine learning model that classifies statements into two categories (binary classification).
* **`Politifact_Model-6Classes.ipynb`**: This Jupyter Notebook contains a more advanced machine learning model for multiclass classification, categorizing statements into six distinct classes.

### Features and Models

The project uses several features for its models, including:
* **Text features**: The models use a TF-IDF vectorizer on the statement text to capture important words and phrases.
* **Sentiment analysis**: A VADER sentiment analyzer is used to extract the sentiment of each statement, which is then used as a feature in the models.
* **Part-of-speech (POS) ratios**: The notebooks calculate the ratios of nouns, verbs, and pronouns in the statements as additional features.

The models explored include:
* Logistic Regression
* Multinomial Naive Bayes
* Support Vector Machines (SVC)
* K-Nearest Neighbors Classifier
* Decision Tree Classifier
* Random Forest Classifier
* AdaBoost Classifier
* XGBoost Classifier

The notebooks perform data preprocessing steps such as filtering data from specific sources (`news`, `blog`, `social_media`), encoding categorical labels, scaling numerical features, and splitting the data into training, validation, and testing sets. They also include performance metrics like `classification_report`, `accuracy_score`, and `confusion_matrix` to evaluate the models' performance.

***
#### How to Use the Notebooks

To run the models and reproduce the results, you will need to:

1.  Clone the repository from GitHub.
2.  Install the necessary libraries listed in the notebook.
3.  Ensure the `politifact_factcheck_data.csv` file is in the same directory as the notebooks.
4.  Run the cells in the Jupyter Notebooks sequentially.

***
#### Repository Statistics

The `politifact_factcheck_data.csv` file contains a total of 21,152 data points. The `statement_source` column has 13 unique categories, with the most common being `news`, `blog`, and `social_media`.

## Liar Dataset Machine Learning Project

This project explores a machine learning approach to fact-checking and classifying statements from the Liar Dataset. The goal is to build models that can predict the truthfulness of a statement based on its content and associated metadata.

### Project Structure

- **`train.tsv`, `test.tsv`, `valid.tsv`**: These files contain the dataset, split into training, testing, and validation sets. [cite_start]Each file is a tab-separated value file with the following columns[cite: 85437]:
    - [cite_start]`label`: The truthfulness label of the statement (e.g., false, barely-true, half-true, mostly-true, true, pants-fire)[cite: 85437].
    - [cite_start]`statement`: The text of the statement itself[cite: 85437].
    - [cite_start]`subject`: The subject of the statement[cite: 85437].
    - [cite_start]`speaker`: The individual who made the statement[cite: 85437].
    - [cite_start]`speaker_job`: The job title or role of the speaker[cite: 85437].
    - [cite_start]`state_info`: The state associated with the speaker[cite: 85437].
    - [cite_start]`party_affiliation`: The political party of the speaker[cite: 85437].
    - [cite_start]`barely_true_counts`, `false_counts`, `half_true_counts`, `mostly_true_counts`, `pants_on_fire_counts`: The number of times the speaker has made statements of that truthfulness type in the past[cite: 85437].
    - [cite_start]`context`: The context in which the statement was made (e.g., interview, rally, debate)[cite: 85437].

- **`Liar_Dataset.ipynb`**: This Jupyter Notebook contains the code for data preprocessing, feature engineering, and model training.

### Methodology

The notebook outlines a comprehensive machine learning pipeline, including:

* [cite_start]**Data Loading & Preprocessing**: The datasets are loaded into pandas DataFrames, and categorical columns like `label` are encoded using `LabelEncoder`[cite: 85437].
* **Feature Engineering**:
    * [cite_start]**Textual Features**: A `TfidfVectorizer` is applied to the `statement` column to convert the text into numerical features, considering both single words and two-word phrases (bigrams)[cite: 85437].
    * [cite_start]**Categorical Features**: One-hot encoding is applied to the `party_affiliation` and `speaker_job` columns[cite: 85437].
    * [cite_start]**Numerical Features**: A min-max scaler is used to process numerical features such as the counts of previous statements[cite: 85437].
* [cite_start]**Model Training & Evaluation**: The notebook explores several classification models to predict the `label` of a statement, including[cite: 85437]:
    * Logistic Regression
    * Multinomial Naive Bayes
    * Support Vector Classifier (SVC)
    * K-Nearest Neighbors Classifier
    * Decision Tree Classifier
    * Random Forest Classifier
    * AdaBoost Classifier
    * XGBoost Classifier
[cite_start]The performance of these models is evaluated using metrics such as `accuracy_score` and `classification_report` on a validation set[cite: 85437].

### How to Run

1.  Clone this repository.
2.  Install the required Python libraries. [cite_start]The notebook imports `pandas`, `numpy`, `scipy.sparse`, and several modules from `sklearn` and `xgboost`[cite: 85437].
3.  Run the cells in the `Liar_Dataset.ipynb` Jupyter Notebook sequentially to reproduce the analysis and model training.
