# COS513-Finance

[Raw data link on GDELT site](data.gdeltproject.org/events/index.html)

[Google Drive Project Folder](https://drive.google.com/a/princeton.edu/folderview?id=0B1BY3pYXuXADUlJTV3llUXJCVE0&usp=sharing_eid&ts=56088709)

Ideas:

* Smarter cluster sampling - not just 150 lines from each day...
* Try SVM classifier
* Try linear regression on the return proportions (p[t+1]-p[t])/p[t] in glm.py
* Use the HMM for up/down classifictation, add its output as another feature
* Extend W2V corpus to have more words
* Smarter clustering: GMM, IGMM, HDP, SNGP
* Get more data, for years <= 2013. Need to convert to YYYYMMDD.export.CSV format. On historical data, we'll need to check the DATE_REPORTED column to recover the same info. Make sure preprocessing.py can handle these differences.
* Introduce periodicity features (this is day i of period P - how to learn P?)
