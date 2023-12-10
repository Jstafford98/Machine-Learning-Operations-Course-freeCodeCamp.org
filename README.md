# Introduction to Machine Learning Operations (MLOps) from freeCodeCamp.org

    Utilizing this course to understand where I, as a Data Scientist and Data Engineer, fit into the 
    Machine Learning Operations (MLOps) pipeline. This will help me understand and improve my workflows
    for the discovery of raw data, development of features, model training, and productionization
    of the entire data pipeline. 

    Questions relevant to me:

        Can I get access to the raw data?
        How can I streamline the extraction and transformation of this data?
        How can I store this data? 
        What models are best suited to fit our goals with this data?
        Am I allowed, legally, to use this data?

## Machine Learning Operations (MLOps) is a set of practices that aims to deploy and maintain machine learing models in production reliably and efficiently
    Steps: 
        1. Data Collection: Project scope, data extraction, and feature engineering
        2. Model Training : Data cleaning and model refinement
        3. Deployment 2 Prod : Tracking performance loss (data drift) and retraining

## Model Centric vs Data Centric 
    Model Centric: data is fixed and the code/model is improved iteratively
    Data Centric : the code/model is fixed, and data is improved iteratively

## Value Proposition
    1. Defining the problem, it's importance, and our end-user persona
    2. Geoffrey Moore's Value Proposition Statement Template: 
        For {target customer} who {need}, our
        {product/service} is {product category} that {benefit}.

## Data Sources
    1. identifying potential data sources (internal database, APIs, open data sources, and more)
    2. Consider hidden costs such as data storage and purchasing data

## Prediction Task
    1. Is our ML task supervised or unsupervised? Anomaly detection? 
       Classification, regression, or ranking?
    2. Consider  I/O and model complexity

## Feature Engineering
    1. Working with domain experts to extract features from raw data sources

## Offline Evaluation
    1. Setup metrics to evaluate system performance pre-deployment
    2. Understand model prediction errors and their impacts

## Decisions
    1. How will the end-user interact with our predictions?

## Collecting data
    1. Collecting new data for re-training models and preventing model decay
    2. Data collection cost consideration and the role of humans in data labeling

## Building models
    1. Model re-training frequency and associated hidden costs
    2. Changes to tech stack and scaling

## Live Evaluation and Monitoring
    1. Setup of metrics to track systems performance post-deployment
    2. Correlation of model metrics and business metrics

## Data Engineering Pipeline
    1. Data Ingest : collection of data from various sources
    2. Exploration and Validation : understanding of data content and structure
    3. Data Wrangling : Formatting and cleaning of data
    4. Data Labeling  : Assigning categories to data points
    5. Data Splitting : Division of data into training, validation, and testing sets

## Model Engineering Pipeline
    1. Model Training: application of ML algos on training data
    2. Model Evaluation : validation of model pre-deployment
    3. Model Testing    : accepting results using test data
    4. Model Packaging  : Export of model into business application

## Model Deployment Pipeline
    1. Model Serving : put model into production
    2. Model Performance Modeling : monitor performance on live, unseen data
    3. Model Performance Logging : recording every inference request

# ZenML
    Pipeline-based approach to MLOps, each step depends on the previous one