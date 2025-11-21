# Customer Review Sentiment Analysis Using VADER, TextBlob, Naive Bayes & Logistic Regression

## Project Overview

This project is an **Natural Language Processing (NLP) solution** designed to transform raw, unstructured customer reviews into actionable business intelligence.

Using a combination of **lexicon-based models** (VADER, TextBlob) and a highly accurate **Machine Learning classifier (Logistic Regression)**, the project diagnoses the root causes of customer dissatisfaction across a line of five products.

> **The core takeaway for a business owner:** We moved past simple sentiment scoring to identify precisely which **products, features, and operational issues** ('Customer Service was bad,' 'Product D broke easily') are driving success and failure.

---

## Table of Contents

* [Project Overview](#-project-overview)
* [The Business Problem](#-the-business-problem)
* [Tech Stack](#-tech-stack)
* [Data Preparation & Feature Engineering](#-data-preparation--feature-engineering)
* [Model Comparisons](#-model-comparisons)
* [The Final Model: Logistic Regression](#-the-final-model-logistic-regression)
* [Key Business Insights](#-key-business-insights)
* [Real-World Impact](#-real-world-impact-what-this-project-does-for-a-business)
* [How to Run the Project](#-how-to-run-the-project)

---
## The Business Problem

In today's e-commerce landscape, manually reading thousands of customer reviews is impossible. The core business problem this project solves is:

**How can we accurately and automatically analyze the sentiment of 1,000 reviews across multiple products, and, quickly pinpoint the exact reasons (e.g., durability, customer service, value) that are causing the customer anger?**

This project provides a clear, quantitative, and prioritized list of action items for Product Engineering, Customer Support, and Marketing teams.

---

## Tech Stack

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Primary programming language. |
| **Data Handling** | pandas, numpy, datetime | Data structuring, mock data generation, and mathematical operations. |
| **Text Processing (NLP)** | spaCy, re (Regular Expressions) |  Tokenization, stop word management, Lemmatization, pattern matching for feature engineering. |
| **Sentiment Analysis** | nltk (VADER), TextBlob | Two separate lexicon-based models for validation and baseline comparison. |
| **Machine Learning** | scikit-learn (Logistic Regression, Multinomial Naive Bayes, TF-IDF) | Building the final, highly accurate predictive sentiment classifier. |
| **Topic Modeling** | scikit-learn (LDA, NMF) | Identifying hidden topics/themes (e.g., 'Durability Issues') within the pool of negative reviews. |

---


## Data Preparation & Feature Engineering

The reliability of a sentiment model rests on the quality of its input data. This project involved rigorous text cleaning to ensure the machine learning model learned from the most relevant features.

* **Contraction Expansion & Cleaning:** The original text was cleaned by expanding contractions ("don't" became "do not") to properly capture negation, followed by lowercasing and removal of all punctuation and numbers.
* **Tokenization & Stop Word Removal:** The text was broken into tokens (words). Stop words ("the," "a") were removed, but **negation words ("not," "no") were explicitly kept** so that the meaning of the sentiment does not change.
* **Lemmatization:** Words were reduced to their dictionary root form (e.g., "running," "ran," "runs" became "run"). This consolidated the features, dramatically reducing the unique word count to just **52 highly impactful lemmas**.
* **Feature Conversion:** The cleaned text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which gives higher weight to words that are important and distinctive to a specific review (e.g., 'terrible,' 'excellent').

---

## Model Comparisons

The project used a multi-step validation process, comparing two popular lexicon-based models (VADER and TextBlob) wth Logistic Regression and Multinomial Naive Bayes.

### Lexicon-Based Analysis (VADER & TextBlob)

These baseline models confirmed the general sentiment direction. VADER was applied to the original text to utilize its ability to interpret capitalization and exclamation points as emotional intensity.

| Sentiment Label | VADER Percentage | TextBlob Percentage |
| :--- | :--- | :--- |
| **Positive** | 45.0% | 50.7% |
| **Negative** | 28.8% | 22.1% |
| **Neutral** | 26.2% | 27.2% |

### Final Model Selection

Two supervised ML classifiers were trained on the pre-processed (lemmatized, TF-IDF vectorized) data, using the 1-5 star ratings as the ground truth.

| Model | Accuracy (on Test Data) | F1-Macro Score |
| :--- | :--- | :--- |
| **Logistic Regression (LR)** | **93.5%** | **0.927** |
| Multinomial Naive Bayes (NB) | 84.0% | 0.772 |

> **Conclusion:** The **Logistic Regression** model was the clear winner, achieving an impressive **93.5% accuracy** on unseen data and was selected for final prediction.

---

## The Final Model: Logistic Regression

The Logistic Regression model was used to predict the sentiment of all 1,000 reviews, yielding the most accurate and balanced sentiment distribution.

| Final Model Label | Count | Percentage |
| :--- | :--- | :--- |
| Positive | 390 | **39.0%** |
| Negative | 390 | **39.0%** |
| Neutral | 220 | 22.0% |

**Real-World Impact:** The model identified a perfectly balanced split between positive and negative customers (**39.0% each**). This suggests a deeply polarizing product experience where a large segment loves the product, but an equally large segment is highly dissatisfied. The overall reputation is neutral, which is a serious business problem.

### Model Validation: Heatmap Scorecard

The model proved exceptional at correlating its predictions with the original customer star ratings, especially at the extremes:

* **5-Star Reviews:** **98.5%** correctly classified as 'Positive'.
* **1-Star Reviews:** **97.4%** correctly classified as 'Negative'.
* **3-Star Reviews:** **100.0%** correctly classified as 'Neutral'.

---

##  Key Business Insights

### 1. The Root Cause of Negative Sentiment: Topic Modeling

Topic Modeling (LDA/NMF) was deployed on the 390 negative reviews to cluster the specific, actionable themes causing customer dissatisfaction.

| Actionable Complaint Category | Key Words | Business Focus |
| :--- | :--- | :--- |
| **Product Durability/Breakage** | `break`, `easily`, `terrible` | **Engineering & Materials (Highest Priority)** |
| Customer Service Failure | `service`, `customer`, `bad` | Training & Support |
| Usability & Missing Features | `difficult`, `use`, `feature`, `miss` | Product Design & UI/UX |
| Value/Financial Disappointment | `waste`, `money`, `price` | Pricing & Marketing |

> **Action Item:** The dominant terms like 'broke' and 'easily' prove that **durability and quality control** are the **#1 issue** to address immediately.

### 2. Pinpointing Problem Products

The analysis isolated which product SKU was responsible for the bulk of the negative feedback.

| ProductID | Negative Review Count | Status |
| :--- | :--- | :--- |
| **Product_D** | **98** | **Highest Risk: IMMEDIATE INVESTIGATION** |
| Product_C | 75 | High Risk |
| Product_A | 73 | High Risk (despite being the best performer overall) |

> **Action Item:** The largest pool of anger stems from **Product D**. Resources must be redirected to fix or redesign this specific product.

### 3. Time-Series Analysis: The Early Warning System

Tracking sentiment over time revealed a critical drop in customer satisfaction.

**Critical Event Identified:** In **August 2024 (2024-08)**, Negative sentiment spiked to 50% while Positive sentiment dropped to 23%. This points to a clear, measurable operational or product failure that occurred around that time and must be investigated to prevent future dips.

---

## Real-World Impact: What This Project Does for a Business



### Where to Focus Resources?

* By isolating the **98 Negative reviews tied to Product D**, the project tells the Head of Product exactly which item needs a redesign to fix customer anger.
* By highlighting that **durability** (keywords: 'broke,' 'easily') is the top complaint, it directs the Engineering Team to prioritize material quality over new features.

### How to Talk to Customers?

* The **Positive Word Cloud** provides the Marketing Team with the exact vocabulary customers use to praise the product ('works perfectly,' 'great value').
* The **Negative Topic Model** provides the Customer Service Team with a script for common complaints (e.g., how to address a 'bad experience' or 'difficult to use' product).


By implementing this pipeline, organizations can move from guessing what customers think to **data-driven confidence**, turning customer complaints from frustrating noise into a prioritized list of profit-saving and growth-driving actions.

---
## How to Run the Project

1.  **Clone the repository:**

    ```bash
    git clone [Your-GitHub-Repo-URL]
    ```

2.  **Install dependencies:** The core dependencies can be installed using pip.

    ```bash
    pip install pandas numpy scikit-learn nltk spacy textblob
    python -m spacy download en_core_web_sm
    ```

3.  **Run the analysis:** The entire project, from data generation to final business insights, is contained in a single file.

    ```bash
    jupyter notebook
    Customer_Review_Sentiment_Analysis_VADER_TextBlob_LogRegression.ipynb
    ```

---
## Author

* **Author:** Hannan Baig
* **Education:** MS Computer Science (NUST)
* **Work/Focus:** Thesis on AI-driven Air Pollution Estimation \| Data Science at Kangaroo Ventures
* **Contact:**
    * **Email:** muhammadhannanbaig@gmail.com
    * **GitHub:** [github.com/hannanbaig347](https://github.com/hannanbaig347)
    * **LinkedIn:** [Hannan Baig](https://www.linkedin.com/in/hannan-baig-b10320325/)