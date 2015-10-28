# COS513-Finance

[Raw data link on GDELT site](data.gdeltproject.org/events/index.html)

[Google Drive Project Folder](https://drive.google.com/a/princeton.edu/folderview?id=0B1BY3pYXuXADUlJTV3llUXJCVE0&usp=sharing_eid&ts=56088709)

Ideas:

* PRIORITY #1: Move data to within an (ignored) subfolder in this git directory, then change scripts to work on relative paths. clone directory on a larger cluster for more paralleled pipeline.
* Scale the date-summary array in glm.py
* Add another feature (number of events per day, N). Also check that lowering scaling by N doesn't make feature floats too small 
* Add another feature (needs to be added in glm.py) - how long ago the news was relative to present day (maybe just do -(num_days_since_unix_epoch))
* Extend W2V corpus to have more words, add it as a feature
* Currently, the clustering columns (topic-columns) are not scaled/normalized (the were before, on a single-day scope, which is inacurate). Consider using the random sample from clustering.py to generate a pre-processin psuedo-normalization which scales the columns according to a sample mean and sample sd (note MLE bias correction there), both prior to generating the kmeans clusters and before doing a classification.
* Smarter cluster sampling - not just 150 lines from each day...
* Try other commodities
* Try SVM classifier
* Other linear classifiers: http://scikit-learn.org/stable/modules/linear_model.html - GLM, RANSAC, Bayesian
* Try linear regression on the return proportions (p[t+1]-p[t])/p[t] in glm.py
* Use the HMM for up/down classifictation, add its output as another feature
* Smarter clustering: GMM, IGMM, HDP, SNGP
* Get more data, for years <= 2013. Need to convert to YYYYMMDD.export.CSV format. On historical data, we'll need to check the DATE_REPORTED column to recover the same info. Make sure preprocessing.py can handle these differences.
* Introduce periodicity features (this is day i of period P - how to learn P?)
