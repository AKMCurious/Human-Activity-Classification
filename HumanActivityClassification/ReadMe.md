README
======

Scala classes:
==============

Of the 4 classifiers we have implemented scala classes for Random forest classifier and Logistic Regression.

Random Forest Classifier usage:
-------------------------------

`spark-submit --class RandomForest <path-to-jar> <location-path>`

Arguments:
- location-path: The parent path that contains both the test.csv and train.csv files

Logistic Regression usage:
--------------------------

Training:
`spark-submit --class LRModel <path-to-jar> <location-path>`

The model built will be save to `<location-path>/savedModel/LR`

Testing the saved model:
`spark-submit --class LRModelTest <path-to-jar> <location-path>`

Arguments:
- location-path: The parent path that contains both the test.csv and train.csv files

DataBricks:
===========

All the 4 classification models are implemented on Databricks and can be found in the below links:

Logistic Regression:
--------------------
accuracy: 93% \
k-fold: 10-folds

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/485634021358808/1609677151688511/5984390043029076/latest.html

Naive Bayes:
------------
accuracy: 83%

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1056971658570274/1822866646691806/3096978060883362/latest.html

Random Forest:
--------------
accuracy: 85.15% \
number of trees: 1000 \
PCA: 100 features

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2723948372695676/1542488507779544/8805466117407199/latest.html

Decision Tree:
--------------
accuracy: 77% \
PCA: 200 features

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/485634021358808/686005152690432/5984390043029076/latest.html
