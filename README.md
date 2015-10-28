# COS513-Finance

[Raw data link on GDELT site](data.gdeltproject.org/events/index.html)

[Google Drive Project Folder](https://drive.google.com/a/princeton.edu/folderview?id=0B1BY3pYXuXADUlJTV3llUXJCVE0&usp=sharing_eid&ts=56088709)

Ideas:

* PRIORITY #1: Get training to run on ionic.princeton.edu
* Scale the date-summary array in glm.py
* Add another feature (number of events per day, N). Also check that lowering scaling by N doesn't make feature floats too small 
* Extend W2V corpus to have more words, add it as a feature
* Currently, the clustering columns (topic-columns) are not scaled/normalized (the were before, on a single-day scope, which is inacurate). Consider using the random sample from clustering.py to generate a pre-processin psuedo-normalization which scales the columns according to a sample mean and sample sd (note MLE bias correction there), both prior to generating the kmeans clusters and before doing a classification.
* Smarter cluster sampling - not just 150 lines from each day...
* Try other commodities
* Try SVM classifier
* Other linear classifiers: http://scikit-learn.org/stable/modules/linear_model.html - GLM, RANSAC, Bayesian
* Try linear regression on the return proportions (p[t+1]-p[t])/p[t] in glm.py
* Use the HMM for up/down classifictation, add its output as another feature
* Smarter clustering: GMM, IGMM, HDP, SNGP. Two approaches here:
  * Apply the dynamic clustering algorithm as a nonparametric model to our random sample. This is most similar to our current pipeline. It would produce an "intrinsic" number of clusters, but this would still be a static clustering pipeline in that the model will eventually grow stale.
  * Take on a fully bayesian, fully dynamic approach. Add priors to all hyperparameters (and non-hyper parameters - this includes both the linear model and the clusters), and then start on a 0- or 1-cluster prior. Run through our training data, making daily updates to the model (should result in incremental slope changes).
    * Reasoning: if we have a dynamic number of clusters, then we need to somehow have a variable number of input features to our linear model. To do this, we'll need to have bayesian regression that learns every day (since it would be able to adapt to learn the new slopes as it gains more clusters). This is completely different to train, but is completely dynamic as well. There's no meaningful transition from a static training set to a test set (and with priors over regularization constants there's no need for validation at all). Instead, every day, we just do a bayesian update on the number of clusters, the slopes associated with the features for a given cluster, etc.
* Get more data, for years <= 2013. Need to convert to YYYYMMDD.export.CSV format. On historical data, we'll need to check the DATE_REPORTED column to recover the same info. Make sure preprocessing.py can handle these differences.
* Introduce periodicity features (this is day i of period P - how to learn P?)
