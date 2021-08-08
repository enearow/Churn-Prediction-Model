## Modification Date
08/08/2021

## Description
Adventure Hardware Group (AHG) is a global manufacturing organisation with operations in
America, Europe and Asia. AHG has been experiencing a shift in the market place towards digital
as well as change in customer demographic, possibly leading to decline in its store sales. To make
the strategic shift toward a greater digital share of wallet and overall growth, AHG has engaged
Kernel Decision Science Limited to help it find a solution to this problem. Among its requirements,
AHG need Kernel to develop a Business Intelligence and Insight Visualisation capability to better
understand and monitor key trends over time. The goal is to create commercial action plan and develop a predictive model using Python to automatically notify AHG leaders 90 days before a customer is due to churn, i.e. when a customer’s churn propensity exceeds a specific threshold, say 0.7

## Data Source 

AHG Relational Database

## Solution Approach 

1. Define churn – Churn is defined as any customer who didn’t make any purchase within the last 8 months of Business transaction i.e. any customer whose maximum/last transaction date is less than 31/10/2007. 

2. Carry exploratory analysis to understand the profile of churners using their demographic, transactional and attitudinal data. I used a clustering model(K means) or decision tree to generate a rule set that can help you describe these customers or plot a distribution of the demographic variables vs the Churn variable (target)

3. Develop a predictive model

4. Churn is my target variable for modelling exercise

5. I have a single customer view i.e the final dataset has a single customer data per row no duplicates 

6. I followed the modelling phases including, statistical testing(Chi-square tests, correlations etc) to test for multi collinearity and over fitting, investigate the distributions, Identify outliers, extreme values or spurious values, impute missing values where applicable, Carry out feature selection, partition the data into training, testing or validation or use cross validation technique, develop various models and select champion model

7. Deploy or score model on an unseen dataset/hold out sample, this dataset is not included in the model development phase.

8. Re validated model before rolling out the main marketing campaign (A/B testing is applicable for a response model)


## Conclusion.
We expect a great progress and quick change in market penetration and revenue in online shops.


## References

https://medium.com/rants-on-machine-learning/7-ways-to-improve-your-predictive-models-753705eba3d6
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
https://towardsdatascience.com/end-to-end-python-framework-for-predictive-modeling-b8052bb96a78
