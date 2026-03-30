**Fraud Analysis**

In this analysis, we would perform 3 stage analysis to discuss the best model to identify fraudulent
transactions.

First, we will perform a series of data analyses to identify data patterns and potential features for the
model. So we can have a better understanding for our model performance.

Later, we will perform a lightGBM model for fraud predictions that optimizes the recall rate. Details and
Reasons are documented below.

Then we will move on to stage 2 analysis, where we form a cost function (weighted fraud transaction
to save costs, on top of our current model to reflect on the company’s various businesses
value/preference.

In our last stage, stage 3, we tested on 2 other different models, Random Forest and Xgboost further
proved our option – LightGBM, the most efficient and accurate model.
