import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
import scikitplot as skplt
import seaborn as sns


#=============================Input Data=======================================
data = pd.read_csv("dataset/drebin-215-dataset-5560malware-9476-benign.csv")
print(data.head())
print(data.info())
print(data.describe())

#============================Pre-processing====================================
print("Total missing values : ",sum(list(data.isna().sum())))

classes,count = np.unique(data['class'],return_counts=True)
#Perform Label Encoding
lbl_enc = LabelEncoder()
print(lbl_enc.fit_transform(classes),classes)
data = data.replace(classes,lbl_enc.fit_transform(classes))

#Dataset contains special characters like ''?' and 'S'. Set them to NaN and use dropna() to remove them
data=data.replace('[?,S]',np.NaN,regex=True)
print("Total missing values : ",sum(list(data.isna().sum())))
data.dropna(inplace=True)
for c in data.columns:
    data[c] = pd.to_numeric(data[c])
data.head()

print("Total Features : ",len(data.columns)-1)

#bening and malcious count plot
plt.bar(classes,count)
plt.title("Class balance")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

#Dataset heatmap
corr= data.corr()
sns.heatmap(corr)
plt.title('Dataset Correlation Plot')
plt.show()
#=============================model selection==================================
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2, shuffle=True)

print("Train features size : ",len(train_x))
print("Train labels size : ",len(train_y))
print("Test features size : ",len(test_x))
print("Test features size : ",len(test_y))


# Scaling the data to make it suitable for the auto-encoder
X_scaled = MinMaxScaler().fit_transform(x)
df1 = pd.DataFrame(X_scaled)

#PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()

#==============================Classification==================================


#pca = PCA(n_components=2)
pca_dims = PCA()
pca_dims.fit(train_x)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=d)
X_reduced = pca.fit_transform(train_x)
X_recovered = pca.inverse_transform(X_reduced)
print("reduced shape: " + str(X_reduced.shape))
print("recovered shape: " + str(X_recovered.shape))

'''RandomForestClassifier'''

print('RandomForestClassifier')
print()
classifier = RandomForestClassifier()
classifier.fit(X_reduced,train_y)
X_test_reduced = pca.transform(test_x)

y_hat_reduced = classifier.predict(X_test_reduced)
rf_prob = classifier.predict_proba(X_test_reduced)[:,1]
print("RF accuracy is: " + str(accuracy_score(test_y, y_hat_reduced))*100)
print()
print('Classification Report')
cr=classification_report(test_y, y_hat_reduced)
print(cr)
rf_cm = confusion_matrix(test_y, y_hat_reduced)
print(rf_cm)
print()
rf_tn = rf_cm[0][0]
rf_fp = rf_cm[0][1]
rf_fn = rf_cm[1][0]
rf_tp = rf_cm[1][1]
Total_TP_FP=rf_cm[0][0]+rf_cm[0][1]
Total_FN_TN=rf_cm[1][0]+rf_cm[1][1]
specificity = rf_tn / (rf_tn+rf_fp)
RF_specificity=format(specificity,'.3f')

print('RF_specificity:',RF_specificity)
print()

rf_mcc=((rf_tp*rf_tn)-(rf_fp*rf_fn))/(np.sqrt((rf_tp+rf_fn)*(rf_tn+rf_fp)*(rf_tp+rf_fp)*(rf_tn+rf_fn)))
print('RF_Matthews Correlation Coefficient:',rf_mcc)
plt.figure()

skplt.estimators.plot_learning_curve(RandomForestClassifier() , train_x, train_y,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="RF Digits Classification Learning Curve");
plt.figure()
sns.heatmap(confusion_matrix(test_y,y_hat_reduced),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print()
rf_fp, rf_tp, thresholds = roc_curve(test_y, rf_prob)
rf_roc_auc = auc(rf_fp, rf_tp)*100
print('RF ROC Accuracy is:',rf_roc_auc,'%')
#LR ROC plot
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characteristic')
plt.plot(rf_fp,rf_tp, color='red',label = 'AUC = %0.2f' % rf_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


'''XGBClassifier'''
print()
print('XGBClassifier')
print()
from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_reduced,train_y)

yhat_reduced = clf.predict(X_test_reduced)
mlp_prob = clf.predict_proba(X_test_reduced)[:,1]
print("XGB accuracy is: " + str(accuracy_score(test_y, yhat_reduced))*100)
print()
print('Classification Report')
xgb_cr=classification_report(test_y, yhat_reduced)
print(xgb_cr)
xgb_cm = confusion_matrix(test_y, yhat_reduced)
print(xgb_cm)
print()
xgb_tn = xgb_cm[0][0]
xgb_fp = xgb_cm[0][1]
xgb_fn = xgb_cm[1][0]
xgb_tp = xgb_cm[1][1]
Total_TP_FP=xgb_cm[0][0]+xgb_cm[0][1]
Total_FN_TN=xgb_cm[1][0]+xgb_cm[1][1]
specificity = xgb_tn / (xgb_tn+xgb_fp)
xgb_specificity=format(specificity,'.3f')

print('XGB_specificity:',xgb_specificity)
print()

xgb_mcc=((xgb_tp*xgb_tn)-(xgb_fp*xgb_fn))/(np.sqrt((xgb_tp+xgb_fn)*(xgb_tn+xgb_fp)*(xgb_tp+xgb_fp)*(xgb_tn+xgb_fn)))
print('XGB_Matthews Correlation Coefficient:',xgb_mcc)
plt.figure()

skplt.estimators.plot_learning_curve(XGBClassifier() , train_x, train_y,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="XGB Digits Classification Learning Curve");
plt.figure()
sns.heatmap(confusion_matrix(test_y,yhat_reduced),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print()
xgb_fp, xgb_tp, thresholds = roc_curve(test_y, mlp_prob)
xgb_roc_auc = auc(xgb_fp, xgb_tp)*100
print('XGB ROC Accuracy is:',xgb_roc_auc,'%')
#LR ROC plot
plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characteristic')
plt.plot(xgb_fp,xgb_tp, color='red',label = 'AUC = %0.2f' % xgb_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()