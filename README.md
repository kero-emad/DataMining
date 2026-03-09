# Stack Overflow Developer Survey 2024 Analysis & Prediction

## Overview
This project provides a comprehensive analysis of the **Stack Overflow Developer Survey 2024**. It explores various facets of the developer community, including demographics, technology preferences, and professional experiences. Furthermore, it implements machine learning models to cluster developers and predict developer types based on survey responses.

## Key Features
- **Data Preprocessing**: Comprehensive cleaning, handling missing values, and feature encoding to prepare the raw survey data for analysis.
- **Exploratory Data Analysis (EDA)**: Statistical summaries and insights into the developer population.
- **Predictive Modeling**: 
  - **Unsupervised Learning**: KMeans clustering to segment developers into distinct groups.
  - **Supervised Learning**: Logistic Regression model achieving high accuracy (~99%) in classifying developer roles within identified clusters.
- **Data Visualization**: Rich visual representations of survey findings using Seaborn and Matplotlib.

## Project Structure
- `Data_Preprocessing.ipynb`: Contains logic for data cleaning, imputation, and initial transformations.
- `EDA.ipynb`: Focuses on statistical exploration, feature selection, and data splitting for training.
- `Modeling.ipynb`: Implements the machine learning pipeline, including clustering and regression analysis.
- `visualization.ipynb`: Dedicated to generating visual insights and demographic plots.
- `survey_results_public.csv`: Raw survey data.
- `survey_results_public_after_cleaning.csv`: The cleaned dataset used for analysis.

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: For efficient data manipulation and numerical computation.
- **Scikit-learn**: For implementing machine learning models and evaluation metrics.
- **Matplotlib & Seaborn**: For creating high-quality data visualizations.
- **Jupyter Notebook**: For interactive development and documentation.

## Getting Started
1. **Prerequisites**: Ensure you have Python installed along with the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn notebook
   ```
2. **Execution**: Open the Jupyter notebooks in the following order to reproduce the results:
   - Start with `Data_Preprocessing.ipynb` to clean the data.
   - Run `EDA.ipynb` and `visualization.ipynb` for insights.
   - Run `Modeling.ipynb` to train and evaluate the machine learning models.

## Results
The project successfully demonstrates the ability to identify patterns within the developer community. The classification model reached an impressive accuracy, significantly aiding in the automated identification of developer roles based on survey parameters.

## License
This project uses data from the [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/2024/). Please refer to Stack Overflow's data usage policies for licensing details.
