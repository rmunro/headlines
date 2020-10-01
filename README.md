# Exploratory Data Analysis for News Headlines
Practical example from Human-in-the-Loop Machine Learning book


## Getting started

To run:

`python headlines.py`

This will open a HTML window that will allow you to choose a label to annotate news headlines according 
to whether they belong to that label.

## Data

The data is approximately one million headlines in total, taken from ABC News in Australia. 

They are all in English

## Problem being addressed

Data Analysts want to understand the distribution of information in their news headline data:
- "I want to see how many news headlines are related to specific topics"
- "I want to track the changes in news headlines topics over time"
- "I want to export all the news articles related to a certain topic for further analysis"



## Annotation strategy

The interface allows you to choose an annotation strategy:

- filter by keyword, 
- filter by year the news article was published, and/or 
- sample items whether the headline was most confusing for the current machine learning model 

The models are built using DistilBERT.

The system incrementally updates the current model with each new annotation. 

The system also continually tries to retrain from scratch when there is new data, because 
an incrementally trained model can be non-optimal. 

Evaluation data is annotated concurrently with training data, so the data analyst using 
the system does not need to wait to annotate a large amount of evaluation data before seeing results. 

Models are always compared across the _current_ evaluation data. So, a model might have been more accurate 
on past data, but it will be replaced when it is not the most accurate on the current (larger)
evaluation data set at any given time.


## Potential extensions

There are many different components in this architecture that could be extended or replaced. 
After playing around with the interface and looking at the results, think about what you might replace/change first.
(Numbers refer to book/chapter sections, but you don't need the book to experiment with this code.)

### Annotation Interface

- Batch-annotation (See 11.2.1): Accepting/rejecting multiple annotations at once. The set of messages that are already grouped per-year could be a good place to start. 
- More powerful filtering (See 9.5): The manual filtering is for exact string matching and this could be made more sophisticated to allow regular expression matching or combinations of multiple keywords.

### Annotation Quality Control

- Using the model as an annotator (See 9.3): Cross-validate the training data to find disagreements between the predicted and the actual annotations, as potential annotation errors to show back to the analyst
- Annotation Aggregation (See 8.1-8.3): If multiple people were using this, strategizing about the ground-truth and inter-annotator agreement methods to aggregate that data. You might split the strategy, updating the model incrementally for every annotation in real-time, but only batch-retraining with items that have been annotated multiple times and are confidently labeled. 

### Machine Learning Architecture

- Self-supervised Learning (See 9.4): Use metadata like the year or the subdomain of the URL as labels, and build a model over the entire dataset to predict those labels, which can in turn be used as a representation in this model.
- Tune the model to the unlabeled data: Tune DistillBERT to the entire dataset of headlines, first. This will make the pre-trained model adapted to this specific domain of text and will likely lead to more accurate results more quickly.

### Active Learning

- Ensemble-based Sampling (See 3.4): Maintain multiple models and track the uncertainty of a prediction across all the models, sampling items with the highest average uncertainty and/or the highest variation in predictions. 
- Diversity Sampling (See 4.2-4.4): Explore clustering and model-based outliers to ensure that there aren’t parts of the features space that are being over-sampled or ignored entirely.





