import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
# ตั้งค่าให้แสดงผลคอลัมน์ทั้งหมด ไม่ต้องย่อ
pd.set_option('display.max_columns', None)

# ตั้งค่าให้แสดงความกว้างของแต่ละคอลัมน์ให้ครบ (กันชื่อคอลัมน์ยาวเกินไปแล้วโดนตัด)
pd.set_option('display.expand_frame_repr', False)
#%%%%
data = pd.read_csv('semiconductor_quality_control.csv')
# แปลงคอลัมน์ Timestamp ให้เป็นรูปแบบวันที่และเวลา
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# ดู 5 แถวแรก
print(data.head())
# ดูสรุปโครงสร้าง จำนวนแถว/คอลัมน์ และการใช้หน่วยความจำ
print(data.info())
#.dt (Datetime accessor) เข้าไปดึงค่า "ชั่วโมง" (0-23)
data['Hour'] = data['Timestamp'].dt.hour
data['DayOfWeek'] = data['Timestamp'].dt.day_name() #monday to sunday
data['Is_NightShift'] = data['Hour'].apply(lambda x: True if (x >= 20 or x < 8) else False) #8pm to 8am = Night shift
data.to_excel("data_summary_report.xlsx")
#%%
wf = pd.read_excel('data_summary_report.xlsx', index_col=0) # บอกให้ Python รู้ว่าคอลัมน์แรกคือ Index ไม่ต้องสร้างคอลัมน์ใหม่
wf = wf.drop(columns=['DayOfWeek', 'Process_ID', 'Wafer_ID', 'Timestamp', 'Is_NightShift','Tool_Type'])
print(wf.info())
#%% Check Data Ratio
# 1. นับจำนวนแถวในแต่ละกลุ่ม (0 และ 1)
defect_counts = data['Defect'].value_counts()
# 2. คำนวณเป็นเปอร์เซ็นต์ (Ratio)
defect_ratio = data['Defect'].value_counts(normalize=True) * 100

df_class_1 = wf[wf['Defect'] == 1]
df_class_0 = wf[wf['Defect'] == 0]

# คำนวณจำนวนที่ต้องสุ่มดึงจากกลุ่มปกติ (ให้เป็น 3 เท่าของกลุ่มของเสีย)
n_class_0_samples = len(df_class_1) * 3

# สุ่มดึงข้อมูลกลุ่มปกติออกมาตามจำนวนที่คำนวณไว้
df_class_0_sampled = df_class_0.sample(n=n_class_0_samples, random_state=123)

# นำข้อมูลของเสียทั้งหมด มารวมกับข้อมูลปกติที่สุ่มมาแล้ว
wf = pd.concat([df_class_1, df_class_0_sampled], axis=0)
print("สัดส่วนใหม่หลังการสุ่ม:")
print(wf['Defect'].value_counts())
print(wf['Defect'].value_counts(normalize=True) * 100)
#%%% Encoding
wf.Defect=pd.Series(wf.Defect.astype("category"))
wf.Join_Status=pd.Series(wf.Join_Status.astype("category"))
#wf.Tool_Type=pd.Series(wf.Tool_Type.astype("category"))

## Dummy variable ช่วยให้อ่านค่าง่ายขึ้น เพราะดูเทียบ base variable
# drop_first=True จะลบออก 1 คอลัมน์เพื่อป้องกันปัญหา Multi-collinearity (มักใช้ใน Logistic Regression)
# dummy = pd.get_dummies(wf[['Tool_Type']], prefix=['Tool_Type'], prefix_sep='-', drop_first=True)
# Dummy ที่ได้ไปรวมกับ DataFrame หลัก (wf)
# wf_dummy = wf.join(dummy)
#ลบคอลัมน์ Tool_Type เดิมที่เป็นตัวอักษรออก
# wf_dummy = wf_dummy.drop(['Tool_Type'], axis=1)

# #Get X and Y
# wf_y = wf_dummy.Defect
# wf_x = wf_dummy
wf_y = wf.Defect
wf_x = wf.drop(columns=['Defect', 'Join_Status'])
#%%% Standardize
# 1. สร้าง Object สำหรับการ Scale
scaler = StandardScaler()

# 2. เลือกคอลัมน์ที่เป็น Sensor Data to create Scaler by calculating from all features = wf_x
scaled_data = scaler.fit_transform(wf_x)

# 3. แปลงกลับเป็น DataFrame เพื่อให้ดูง่ายและมีชื่อคอลัมน์เหมือนเดิม
wf_x_scaled = pd.DataFrame(scaled_data, columns=wf_x.columns)
#%% Split the dataset into training (75%) and testing (25%)
wf_train_x, wf_test_x, wf_train_y, wf_test_y = train_test_split(
    wf_x_scaled,  ##caution: use x features that are standardized
    wf_y, 
    test_size=0.25, 
    random_state=123 # Using a fixed state for reproducible splits
)
#%%
##RandomForest##
## Randominzed Search in RF
#high accu, low pecision
params_RF = {'n_estimators':[100, 200, 300], #The number of trees in the forest.
             'max_depth':[5, 8, 10, 12],  #The maximum depth of the tree.
             'max_features':[2, 4, 6, 8, 'sqrt'], #The number of features to consider when looking for the best split
             'min_samples_split':[7, 10, 20, 50], #The minimum number of samples required to split an internal node
             'min_samples_leaf':[5, 10, 20, 50]} #The minimum number of samples required to be at a leaf node.
#med accu, med precision
params_RF = {'n_estimators':list(range(50,151,20)), #The number of trees in the forest.
             'max_depth':list(range(2,5)),  #The maximum depth of the tree.
             'max_features':list(range(2,5)), #The number of features to consider when looking for the best split
             'min_samples_split':list(range(10,31,5)), #The minimum number of samples required to split an internal node
             'min_samples_leaf':list(range(5,21,5))} #The minimum number of samples required to be at a leaf node.

tunnRF = RandomizedSearchCV(RandomForestClassifier(class_weight='balanced'), 
                            params_RF, cv = 3,scoring='f1', random_state=(516)) #เปลี่ยนจาก Accuracy เป็น f1 เพราะเราเน้นของเสีย
tunnRF.fit(wf_train_x,wf_train_y)
    
#Get Best parameter
print("\n''''''''''''RandomForest Classifier''''''''''''''''")
print('\nBest parameters :',tunnRF.best_params_)
#%%
#Build classifier
rf = RandomForestClassifier(criterion = 'gini',
                            class_weight = 'balanced',
                            n_estimators = tunnRF.best_estimator_.n_estimators,
                            max_depth = tunnRF.best_estimator_.max_depth,
                            max_features = tunnRF.best_estimator_.max_features, 
                            min_samples_split = tunnRF.best_estimator_.min_samples_split,
                            min_samples_leaf = tunnRF.best_estimator_.min_samples_leaf,
                            random_state=516)
rf_wf = rf.fit(wf_train_x,wf_train_y)

#Get importance of feature with sorting
Wafer_imp = pd.DataFrame({'Feature':wf_train_x.columns,
                         'Importance':rf_wf.feature_importances_})
print('\nFeature importance:\n',Wafer_imp.sort_values(by=['Importance'],ascending=False))

# Store feature importances for common KPI selection
all_importances = {}
all_importances['RF'] = rf_wf.feature_importances_
#%%
#Predict the train subset
RFtrain_pred = rf_wf.predict(wf_train_x)
train_conf = pd.crosstab(wf_train_y,RFtrain_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n',train_conf)
print('\nAccuracy:',metrics.accuracy_score(wf_train_y, RFtrain_pred))
print('\nRecall:',metrics.recall_score(wf_train_y, RFtrain_pred, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(wf_train_y, RFtrain_pred, pos_label=1, average='binary'))
print('\nFmeasure:',metrics.f1_score(wf_train_y, RFtrain_pred, pos_label=1, average='binary'))

#Predict the test subset
RFtest_pred = rf_wf.predict(wf_test_x)
test_conf = pd.crosstab(wf_test_y,RFtest_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing:\n',test_conf)
print('\nAccuracy:',metrics.accuracy_score(wf_test_y,RFtest_pred,normalize = True))
print('\nRecall:', metrics.recall_score(wf_test_y, RFtest_pred, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(wf_test_y, RFtest_pred, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(wf_test_y, RFtest_pred, pos_label=1, average='binary'))

#%% not use
# 1. แทนที่จะทาย 0/1 เลย ให้ดึง "ความน่าจะเป็น" (Probability) ออกมาก่อน
RFtest_proba = rf_wf.predict_proba(wf_test_x)[:, 1]
#2. ตั้งเกณฑ์ (Threshold) ใหม่เพื่อให้ Recall สูงขึ้น 
custom_threshold = 0.4 
RFtest_pred = (RFtest_proba >= custom_threshold).astype(int)
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    temp_pred = (RFtest_proba >= threshold).astype(int)
    rec = metrics.recall_score(wf_test_y, temp_pred)
    print(f"Threshold: {threshold:.2f} | Recall: {rec:.4f}")
#%%
##XGBoost Classifier##
# Prepare the data
wf_train_y_num = wf_train_y.astype(int)
wf_test_y_num = wf_test_y.astype(int)
## Randominzed Search in XGB

ratio = len(wf_train_y_num[wf_train_y_num==0]) / len(wf_train_y_num[wf_train_y_num==1])
# parameter grid
xgb_candidates = {'n_estimators':[100, 200, 300], #The number of trees in the forest.
                  'max_depth':[2,3,4],  #The maximum depth of the tree.
                  'learning_rate':[0.1,0.05,0.01], #Step size shrinkage used in update to prevents overfitting.
                  'min_child_weight':[0.5,1,5], #The minimum sum of weights of all observations required in a child (not number of observations which is min_samples_leaf. ).
                  'colsample_bytree':[0.5,0.7,0.9]} #Similar to max_features. Denotes the fraction of columns to be randomly samples for each tree.
tunnXgb = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio), 
                             xgb_candidates, scoring='f1', cv = 3, random_state=(123))
tunnXgb.fit(wf_train_x, wf_train_y_num)

#Get Best parameter
print("\n''''''''''''''''XGBoostClassifier''''''''''''''''")
print('\nBest parameters :',tunnXgb.best_params_)
#%%
#Build classifier
XGB = XGBClassifier(n_estimators=tunnXgb.best_estimator_.n_estimators,
                    learning_rate=tunnXgb.best_estimator_.learning_rate,
                    max_depth=tunnXgb.best_estimator_.max_depth,
                    min_child_weight=tunnXgb.best_estimator_.min_child_weight,
                    colsample_bytree=tunnXgb.best_estimator_.colsample_bytree,
                    scale_pos_weight=ratio,
                    eval_metric='logloss')
xgb_wf = XGB.fit(wf_train_x, wf_train_y_num)

#Get importance of feature with sorting
wf_imp_xgb = pd.DataFrame({'Feature':wf_train_x.columns,'Importance':xgb_wf.feature_importances_})
print('\nFeature importance:\n',wf_imp_xgb.sort_values(by=['Importance'],ascending=False))

# Store feature importances for common KPI selection
all_importances['XGB'] = xgb_wf.feature_importances_
#%%
#Predict the train subset
XGBtrain_pred = xgb_wf.predict(wf_train_x)
#XGBtrain_pred = pd.Series(train_pred).replace(1,'Non-joining').replace(0,'Joining').astype("category")
XGBtrain_conf = pd.crosstab(wf_train_y,XGBtrain_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n', XGBtrain_conf)
print('\nAccuracy:',metrics.accuracy_score(wf_train_y, XGBtrain_pred))
print('\nRecall:',metrics.recall_score(wf_train_y, XGBtrain_pred, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(wf_train_y, XGBtrain_pred, pos_label=1, average='binary'))
print('\nFmesure:',metrics.f1_score(wf_train_y, XGBtrain_pred, pos_label=1, average='binary'))

#Predict the test subset
XGBtest_pred = xgb_wf.predict(wf_test_x)
#XGBtest_pred = pd.Series(test_pred).replace(1,'Yes').replace(0,'No').astype("category")
XGBtest_conf = pd.crosstab(wf_test_y, XGBtest_pred, rownames=['real'], colnames=['pred'])
print('\nConfusion Matrix Testing:\n',XGBtest_conf)
print('\nAccuracy:',metrics.accuracy_score(wf_test_y,XGBtest_pred,normalize = True))
print('\nRecall:', metrics.recall_score(wf_test_y, XGBtest_pred, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(wf_test_y, XGBtest_pred, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(wf_test_y, XGBtest_pred, pos_label=1, average='binary'))
#%%
XGBtest_proba = xgb_wf.predict_proba(wf_test_x)[:, 1]

# สร้างตารางเปรียบเทียบค่า Threshold
threshold_results = []
for th in [0.35, 0.4, 0.45, 0.5]:
    temp_pred = (XGBtest_proba >= th).astype(int)
    rec = metrics.recall_score(wf_test_y_num, temp_pred)
    acc = metrics.accuracy_score(wf_test_y_num, temp_pred)
    f1 = metrics.f1_score(wf_test_y_num, temp_pred)
    threshold_results.append((th, rec, acc, f1))

df_XGB = pd.DataFrame(threshold_results, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])
print(df_XGB)
#%%%
#Adaboost
base_estimator = DecisionTreeClassifier(random_state=1) 

params_AB = {
    'n_estimators': [20, 50, 100],
    'learning_rate': [0.001, 0.01, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3] # base estimator parameter: DT
}

tunnAB = RandomizedSearchCV(AdaBoostClassifier(base_estimator=base_estimator, random_state=1), 
                            params_AB, cv = 5, random_state=2, error_score='raise', scoring='f1') # สุ่มลองสัก 20 รูปแบบ
tunnAB.fit(wf_train_x, wf_train_y_num)

print("\n''''''''''''Adaboost Classifier''''''''''''''''")
print('\nBest parameters :',tunnAB.best_params_)

ab = tunnAB.best_estimator_
ab_wf = ab.fit(wf_train_x, wf_train_y_num)

ab_wf_imp = pd.DataFrame({'Feature':wf_train_x.columns,
                          'Importance':ab_wf.feature_importances_})
print('\nFeature importance (AB):\n',ab_wf_imp.sort_values(by=['Importance'],ascending=False).head(5))

#Store in place
all_importances['AB'] = ab_wf.feature_importances_
#%%%
train_pred_ab = ab_wf.predict(wf_train_x)
train_conf_ab = pd.crosstab(wf_train_y,train_pred_ab,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training (AB):\n',train_conf_ab)
ab_train_accu = metrics.accuracy_score(wf_train_y, train_pred_ab,normalize = True)
ab_train_recall = metrics.recall_score(wf_train_y, train_pred_ab, pos_label=1, average='binary')
ab_train_precision = metrics.precision_score(wf_train_y, train_pred_ab, pos_label=1, average='binary')
ab_train_fmeasure = metrics.f1_score(wf_train_y, train_pred_ab, pos_label=1, average='binary')
print('\nAccuracy:', ab_train_accu)
print('\nRecall:', ab_train_recall)
print('\nPrecision:', ab_train_precision)
print('\nFmeasure:', ab_train_fmeasure)

#Predict the test subset
test_pred_ab = ab_wf.predict(wf_test_x)
test_conf_ab = pd.crosstab(wf_test_y, test_pred_ab,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing (AB):\n',test_conf_ab)
ab_test_accu = metrics.accuracy_score(wf_test_y, test_pred_ab)
ab_test_recall = metrics.recall_score(wf_test_y, test_pred_ab, pos_label=1, average='binary')
ab_test_precision = metrics.precision_score(wf_test_y, test_pred_ab, pos_label=1, average='binary')
ab_test_fmeasure = metrics.f1_score(wf_test_y, test_pred_ab, pos_label=1, average='binary')
print('\nAccuracy:', ab_test_accu )
print('\nRecall:', ab_test_recall)
print('\nPrecision:', ab_test_precision)
print('\nFmeasure:', ab_test_fmeasure)
#%%%
AB_proba = ab_wf.predict_proba(wf_test_x)[:, 1]
# สร้างตารางเปรียบเทียบค่า Threshold
threshold_results = []
for th in [0.35, 0.4, 0.45, 0.48, 0.49, 0.5]:
    temp_pred = (AB_proba >= th).astype(int)
    rec = metrics.recall_score(wf_test_y_num, temp_pred)
    acc = metrics.accuracy_score(wf_test_y_num, temp_pred)
    f1 = metrics.f1_score(wf_test_y_num, temp_pred)
    threshold_results.append((th, rec, acc, f1))

df_AB = pd.DataFrame(threshold_results, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])
print(df_AB)

print('\nConfusion Matrix Testing (AB):\n',test_conf_ab)
best_pred = (AB_proba >= 0.48).astype(int) 
print('\nAccuracy (th 0.48):', metrics.accuracy_score(wf_test_y_num, best_pred))
print('\nRecall (th 0.48):', metrics.recall_score(wf_test_y_num, best_pred))
print('\nPrecision (th 0.48):', metrics.precision_score(wf_test_y_num, best_pred))
print('\nFmeasure (th 0.48):', metrics.f1_score(wf_test_y_num, best_pred))
#%%%
#Select Common KPIs (Cumulative Average Importance > 85%) from RF, XGB, AB

print("\n''''''''''''Common KPI Selection''''''''''''''''")
feature_names = wf_train_x.columns

# Create DataFrame to accumulate total Feature Importance from two models
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': all_importances['RF'],
    'XGB_Importance': all_importances['XGB'],
    'AB_Importance': all_importances['AB']
})

#Average degree of importance
importance_df['Average_Importance'] = importance_df[['AB_Importance', 'XGB_Importance', 'RF_Importance']].mean(axis=1)

#Sort from H to L
importance_df = importance_df.sort_values(by='Average_Importance', ascending=False).reset_index(drop=True)

#Calculate Cumulative Importance
importance_df['Normalized_Avg_Importance'] = importance_df['Average_Importance'] / importance_df['Average_Importance'].sum()
importance_df['Cumulative_Avg_Importance'] = importance_df['Normalized_Avg_Importance'].cumsum()
print('\nAverage Feature importance (Sorted):\n')

#Add Average_Importance column in matrix
print(importance_df[['Feature', 'Average_Importance', 'Normalized_Avg_Importance', 'Cumulative_Avg_Importance']])

#Selected Common KPIs: Cumulative average importance > 85% == capture 85% of impartance ก็คือจุดใต้ 85% ลงมา
kpi_features = importance_df[importance_df['Cumulative_Avg_Importance'] <= 0.85]

print("\n--- Selected Common KPIs (Cumulative Average Importance > 85%) ---")
print(f"Number of Selected Features: {len(kpi_features)}")
print("Selected Features:", list(kpi_features['Feature']))
#%% PCA
wf = pd.read_excel(r'C:\Users\Kwankamol\Downloads\Semiconductor Sensor Data for Predictive Quality\data_summary_report.xlsx', index_col=0) # บอกให้ Python รู้ว่าคอลัมน์แรกคือ Index ไม่ต้องสร้างคอลัมน์ใหม่
wf = wf.drop(columns=['DayOfWeek', 'Process_ID', 'Wafer_ID', 'Timestamp']) #เก็บ'Tool_Type'ไว้ทำ PCA
print(wf.info())
#%% Check Data Ratio
df_class_1 = wf[wf['Defect'] == 1]
df_class_0 = wf[wf['Defect'] == 0]
# คำนวณจำนวนที่ต้องสุ่มดึงจากกลุ่มปกติ (ให้เป็น 3 เท่าของกลุ่มของเสีย)
n_class_0_samples = len(df_class_1) * 3
# สุ่มดึงข้อมูลกลุ่มปกติออกมาตามจำนวนที่คำนวณไว้
df_class_0_sampled = df_class_0.sample(n=n_class_0_samples, random_state=123)
# นำข้อมูลของเสียทั้งหมด มารวมกับข้อมูลปกติที่สุ่มมาแล้ว
wf = pd.concat([df_class_1, df_class_0_sampled], axis=0)
print("สัดส่วนใหม่หลังการสุ่ม:")
print(wf['Defect'].value_counts())
print(wf['Defect'].value_counts(normalize=True) * 100)
#%%
wf.Join_Status=pd.Series(wf.Join_Status.astype("category"))
wf.Tool_Type=pd.Series(wf.Tool_Type.astype("category"))

## Dummy variable ช่วยให้อ่านค่าง่ายขึ้น เพราะดูเทียบ base variable
# drop_first=True จะลบออก 1 คอลัมน์เพื่อป้องกันปัญหา Multi-collinearity (มักใช้ใน Logistic Regression)
dummy = pd.get_dummies(wf[['Tool_Type']], prefix=['Tool_Type'], prefix_sep='-', drop_first=True)
# Dummy ที่ได้ไปรวมกับ DataFrame หลัก (wf)
wf_dummy = wf.join(dummy)
#ลบคอลัมน์ Tool_Type เดิมที่เป็นตัวอักษรออก
wf_dummy = wf_dummy.drop(['Tool_Type'], axis=1)
print(wf_dummy.info())

cols_to_int = ['Is_NightShift', 'Tool_Type-Etching', 'Tool_Type-Lithography']

for col in cols_to_int:
    if col in wf_dummy.columns:
        wf_dummy[col] = wf_dummy[col].astype(int)

# #Get X and Y
wf_y = wf_dummy.Defect
wf_x = wf_dummy.drop(columns=['Defect', 'Join_Status'])
#%%
# standarize before PCA
#%%% Standardize
# 1. สร้าง Object สำหรับการ Scale
scaler = StandardScaler()

# 2. เลือกคอลัมน์ที่เป็น Sensor Data to create Scaler by calculating from all features = wf_x
scaled_data = scaler.fit_transform(wf_x)

# 3. แปลงกลับเป็น DataFrame เพื่อให้ดูง่ายและมีชื่อคอลัมน์เหมือนเดิม
wf_x_scaled = pd.DataFrame(scaled_data, columns=wf_x.columns)
#%%
#principal component analysis
pca = PCA(n_components=14, random_state=123) #14 principal components
pca.fit_transform(wf_x_scaled)

eigen_value = pd.DataFrame(np.round(pca.explained_variance_), columns=['eigen value']) #The greater the variation, the better.
print('eigen value:\n',eigen_value)

wf_var = np.round(pca.explained_variance_ratio_,4)*100 #How much variation did each of the 14 principal components explain?
wf_var = pd.DataFrame(wf_var)
wf_var.columns = ['explained var']
print("\n explained variances(%):\n",wf_var)

wf_cumuvar = np.cumsum(wf_var) #The cumulative variance explained by the 11 principal components
wf_cumuvar.columns = ['cumulative var']
print("\n cumulative explained variances(%):\n",wf_cumuvar)

plt.plot(wf_cumuvar)
plt.xlabel('components')
plt.ylabel('cumulative explained variance')
#11 principal components are needed to explain 85% of the variation.
#%%
#choose components
pca11 = PCA(n_components=11, random_state=123)

#New coordinates for functions
new_x = pd.DataFrame(pca11.fit_transform(wf_x_scaled), columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7', 'pc8', 'pc9', 'pc10', 'pc11'])
print('new x:\n',new_x)

eigen_vector = pd.DataFrame(pca11.components_.T, columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7', 'pc8', 'pc9', 'pc10', 'pc11'], index=list(wf_x.columns))
print('\neigen vector:\n',eigen_vector) 
# ดูว่าเซนเซอร์ตัวไหนมีอิทธิพลต่อ PC1 มากที่สุด
print("--- Top 5 Sensors for PC1 ---")
print(eigen_vector['pc1'].abs().sort_values(ascending=False).head(5))
print("\n--- Top 5 Sensors for PC7 ---")
print(eigen_vector['pc7'].abs().sort_values(ascending=False).head(5))

for i in range(1, 12):
    pc_name = f'pc{i}'
    print(f"--- Top 5 Sensors for {pc_name.upper()} ---")
    # ดึงค่า Eigen Vector ของ PC นั้นๆมาทำ Absolute เพื่อดูขนาดอิทธิพล (ไม่สนทิศทาง) เรียงลำดับจากมากไปน้อย และดึง 5 ตัวแรก
    top_sensors = eigen_vector[pc_name].abs().sort_values(ascending=False).head(5)
    print(top_sensors)
    print("-" * 30)
    
# Heat map
plt.figure(figsize=(12, 8))
# ใช้ .T เพื่อสลับแกนให้ Sensor อยู่ด้านซ้าย และ PC อยู่ด้านบน จะอ่านง่ายขึ้น
sns.heatmap(eigen_vector.T, cmap='RdBu', center=0, annot=True, fmt='.2f')
plt.title('Principal Component Loadings (Sensor vs PC)')
plt.show()
#%% Split the dataset into training (75%) and testing (25%)
wf_y_num = wf_y.astype(int)

new_train_x, new_test_x, new_train_y, new_test_y = train_test_split(
    new_x,  ##caution: use PCA x features that are standardized
    wf_y_num, ##XGB, Ada, SVC use integer only
    test_size=0.25, 
    random_state=123 # Using a fixed state for reproducible splits
)
#%%%
#### RF with PCA ####
# In[0] Grid ไม่ต้องรัน ผลไม่ดี
RF_param_grid = { 'min_samples_split': list((20,30,40)),
                   'max_features':list(range(3,8)),
                   'n_estimators':list((80,120,150)),
                   'max_depth':list((4,6,8))}
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=(516)),param_grid = RF_param_grid, cv=3,scoring='f1')
# GridSearchCV:       ลอง Every Parameter Combination ที่ใส่ลงใน Grid: ปูพรมแต่ช้า
# RandomizedSearchCV: Random Pick บางชุดความสัมพันธ์ขึ้นมาทดสอบตามจำนวนครั้งที่เรากำหนด (n_iter)
grid.fit(new_train_x, new_train_y)
grid.cv_results_, grid.best_params_, grid.best_score_
#Get Best parameter
print(grid.best_params_)

RF=RandomForestClassifier(max_depth=grid.best_params_['max_depth'], 
                          max_features=grid.best_params_['max_features'], 
                          min_samples_split=grid.best_params_['min_samples_split'],
                          n_estimators=grid.best_params_['n_estimators'],
                          class_weight = 'balanced')
RF.fit(new_train_x,new_train_y)

#Get importance of feature with sorting
RF_imp = pd.DataFrame({'Feature':new_train_x.columns,
                         'Importance':RF.feature_importances_})
print('\nFeature importance (PCA):\n',RF_imp.sort_values(by=['Importance'],ascending=False))

# Store feature importances for common KPI selection
all_importances_pca = {}
all_importances_pca['RF'] = RF.feature_importances_
#%% TRAIN PCA
pred_RF = RF.predict(new_train_x)
conf_RF = pd.crosstab(new_train_y.values.flatten(),pred_RF,rownames=['real'],colnames=['pred'])
conf_RF.shape
print('\nConfusion Matrix Training:\n',conf_RF)
print("\n Performance by using PCA+RF\n")
print('\nAccuracy= ', metrics.accuracy_score(new_train_y, pred_RF))
RF_p=metrics.accuracy_score(new_train_y, pred_RF)
print('\nRecall:',metrics.recall_score(new_train_y, pred_RF, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(new_train_y, pred_RF, pos_label=1, average='binary'))
print('\nFmeasure:',metrics.f1_score(new_train_y, pred_RF, pos_label=1, average='binary'))

#TEST PCA
testpred_RF = RF.predict(new_test_x)
testconf_RF = pd.crosstab(new_test_y.values.flatten(),testpred_RF,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing:\n', testconf_RF)
print('\nAccuracy:',metrics.accuracy_score(new_test_y,testpred_RF,normalize = True))
print('\nRecall:', metrics.recall_score(new_test_y, testpred_RF, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(new_test_y, testpred_RF, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(new_test_y, testpred_RF, pos_label=1, average='binary'))
#%% #high accu, low pecision
params_RF = {'n_estimators':[100, 200, 300], #The number of trees in the forest.
             'max_depth':[5, 8, 10, 12],  #The maximum depth of the tree.
             'max_features':[2, 4, 6, 8, 'sqrt'], #The number of features to consider when looking for the best split
             'min_samples_split':[7, 10, 20, 50], #The minimum number of samples required to split an internal node
             'min_samples_leaf':[5, 10, 20, 50]} #The minimum number of samples required to be at a leaf node.
#med accu, med precision >> 31% F measure !!!!
params_RF = {'n_estimators':list(range(50,151,20)), #The number of trees in the forest.
             'max_depth':list(range(2,5)),  #The maximum depth of the tree.
             'max_features':list(range(2,5)), #The number of features to consider when looking for the best split
             'min_samples_split':list(range(10,31,5)), #The minimum number of samples required to split an internal node
             'min_samples_leaf':list(range(5,21,5))} #The minimum number of samples required to be at a leaf node.
tunnRF = RandomizedSearchCV(RandomForestClassifier(class_weight='balanced'), 
                            params_RF, cv = 3, scoring='f1', random_state=123) #เปลี่ยนจาก Accuracy เป็น f1 เพราะเราเน้นของเสีย
tunnRF.fit(new_train_x,new_train_y)
#Get Best parameter
print('\nBest parameters :',tunnRF.best_params_)

#Build classifier
rf = RandomForestClassifier(criterion = 'gini',
                            class_weight = 'balanced',
                            n_estimators = tunnRF.best_estimator_.n_estimators,
                            max_depth = tunnRF.best_estimator_.max_depth,
                            max_features = tunnRF.best_estimator_.max_features, 
                            min_samples_split = tunnRF.best_estimator_.min_samples_split,
                            min_samples_leaf = tunnRF.best_estimator_.min_samples_leaf,
                            random_state=123)
rf_wf = rf.fit(new_train_x,new_train_y)

#Get importance of feature with sorting
Wafer_imp = pd.DataFrame({'Feature':new_train_x.columns,
                         'Importance':rf_wf.feature_importances_})
print('\nFeature importance (PCA):\n',Wafer_imp.sort_values(by=['Importance'],ascending=False))

# Store feature importances for common KPI selection
all_importances_pca = {}
all_importances_pca['RF'] = rf_wf.feature_importances_
#%%
#Predict the train subset
RFtrain_pred = rf_wf.predict(new_train_x)
train_conf = pd.crosstab(new_train_y,RFtrain_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training (PCA):\n',train_conf)
print('\nAccuracy:',metrics.accuracy_score(new_train_y, RFtrain_pred))
print('\nRecall:',metrics.recall_score(new_train_y, RFtrain_pred, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(new_train_y, RFtrain_pred, pos_label=1, average='binary'))
print('\nFmeasure:',metrics.f1_score(new_train_y, RFtrain_pred, pos_label=1, average='binary'))

#Predict the test subset
RFtest_pred = rf_wf.predict(new_test_x)
test_conf = pd.crosstab(new_test_y,RFtest_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing (PCA):\n',test_conf)
print('\nAccuracy:',metrics.accuracy_score(new_test_y,RFtest_pred,normalize = True))
print('\nRecall:', metrics.recall_score(new_test_y, RFtest_pred, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(new_test_y, RFtest_pred, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(new_test_y, RFtest_pred, pos_label=1, average='binary'))
#%% not use
# 1. แทนที่จะทาย 0/1 เลย ให้ดึง "ความน่าจะเป็น" (Probability) ออกมาก่อน
RFtest_proba = rf_wf.predict_proba(new_test_x)[:, 1]
#2. ตั้งเกณฑ์ (Threshold) ใหม่
RFth_results = []
for th in [0.35, 0.4, 0.45, 0.48, 0.49, 0.5]:
        temp_pred = (RFtest_proba >= th).astype(int)
        rec = metrics.recall_score(new_test_y, temp_pred)
        acc = metrics.accuracy_score(new_test_y, temp_pred)
        f1 = metrics.f1_score(new_test_y, temp_pred)
        RFth_results.append((th, rec, acc, f1))
df_RF = pd.DataFrame(RFth_results, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])
print(df_RF)

print('\nConfusion Matrix Testing (RF+PCA):\n',test_conf)
best_pred = (RFtest_proba >= 0.45).astype(int) 
print('\nAccuracy (th 0.45):', metrics.accuracy_score(new_test_y, best_pred))
print('\nRecall (th 0.45):', metrics.recall_score(new_test_y, best_pred))
print('\nPrecision (th 0.45):', metrics.precision_score(new_test_y, best_pred))
print('\nFmeasure (th 0.45):', metrics.f1_score(new_test_y, best_pred))
#%%
## XGB with PCA
ratio = len(new_train_y[new_train_y==0]) / len(new_train_y[new_train_y==1])
# parameter grid
xgb_candidates = {'n_estimators':[100, 200, 300], #The number of trees in the forest.
                  'max_depth':[2,3,4],  #The maximum depth of the tree.
                  'learning_rate':[0.1,0.05,0.01], #Step size shrinkage used in update to prevents overfitting.
                  'min_child_weight':[0.5,1,5], #The minimum sum of weights of all observations required in a child (not number of observations which is min_samples_leaf. ).
                  'colsample_bytree':[0.5,0.7,0.9]} #Similar to max_features. Denotes the fraction of columns to be randomly samples for each tree.
tunnXgb = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio), 
                             xgb_candidates, scoring='f1', cv = 3, random_state=(123))
tunnXgb.fit(new_train_x, new_train_y)

#Get Best parameter
print("\n''''''''''''''''XGBoostClassifier''''''''''''''''")
print('\nBest parameters :',tunnXgb.best_params_)
#Build classifier
XGB = XGBClassifier(n_estimators=tunnXgb.best_estimator_.n_estimators,
                    learning_rate=tunnXgb.best_estimator_.learning_rate,
                    max_depth=tunnXgb.best_estimator_.max_depth,
                    min_child_weight=tunnXgb.best_estimator_.min_child_weight,
                    colsample_bytree=tunnXgb.best_estimator_.colsample_bytree,
                    scale_pos_weight=ratio,
                    eval_metric='logloss',
                    random_state=(123))
xgb_wf = XGB.fit(new_train_x, new_train_y)

#Get importance of feature with sorting
wf_imp_xgb = pd.DataFrame({'Feature':new_train_x.columns,'Importance':xgb_wf.feature_importances_})
print('\nFeature importance:\n',wf_imp_xgb.sort_values(by=['Importance'],ascending=False))

# Store feature importances for common KPI selection
all_importances_pca['XGB'] = xgb_wf.feature_importances_
#%%
#Predict the train subset
XGBtrain_pred = xgb_wf.predict(new_train_x)
#XGBtrain_pred = pd.Series(train_pred).replace(1,'Non-joining').replace(0,'Joining').astype("category")
XGBtrain_conf = pd.crosstab(new_train_y,XGBtrain_pred,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training:\n', XGBtrain_conf)
print('\nAccuracy:',metrics.accuracy_score(new_train_y, XGBtrain_pred))
print('\nRecall:',metrics.recall_score(new_train_y, XGBtrain_pred, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(new_train_y, XGBtrain_pred, pos_label=1, average='binary'))
print('\nFmesure:',metrics.f1_score(new_train_y, XGBtrain_pred, pos_label=1, average='binary'))

#Predict the test subset
XGBtest_pred = xgb_wf.predict(new_test_x)
#XGBtest_pred = pd.Series(test_pred).replace(1,'Yes').replace(0,'No').astype("category")
XGBtest_conf = pd.crosstab(new_test_y, XGBtest_pred, rownames=['real'], colnames=['pred'])
print('\nConfusion Matrix Testing:\n',XGBtest_conf)
print('\nAccuracy:',metrics.accuracy_score(new_test_y,XGBtest_pred,normalize = True))
print('\nRecall:', metrics.recall_score(new_test_y, XGBtest_pred, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(new_test_y, XGBtest_pred, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(new_test_y, XGBtest_pred, pos_label=1, average='binary'))
#%%
XGBtest_proba = xgb_wf.predict_proba(new_test_x)[:, 1]

# สร้างตารางเปรียบเทียบค่า Threshold
XGBthreshold_results = []
for th in [0.35, 0.45, 0.48, 0.49, 0.5]:
    temp_pred2 = (XGBtest_proba >= th).astype(int)
    rec = metrics.recall_score(new_test_y, temp_pred2)
    acc = metrics.accuracy_score(new_test_y, temp_pred2)
    f1 = metrics.f1_score(new_test_y, temp_pred2)
    XGBthreshold_results.append((th, rec, acc, f1))

df_XGB = pd.DataFrame(XGBthreshold_results, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])
print(df_XGB)

print('\nConfusion Matrix Testing (XGB+PCA):\n',XGBtest_conf)
XGBbest_pred = (XGBtest_proba >= 0.49).astype(int) 
print('\nAccuracy (th 0.49):', metrics.accuracy_score(new_test_y, XGBbest_pred))
print('\nRecall (th 0.49):', metrics.recall_score(new_test_y, XGBbest_pred))
print('\nPrecision (th 0.49):', metrics.precision_score(new_test_y, XGBbest_pred))
print('\nFmeasure (th 0.49):', metrics.f1_score(new_test_y, XGBbest_pred))
#%%
#Adaboost+PCA
base_estimator = DecisionTreeClassifier(random_state=1, class_weight = 'balanced') 

params_AB = {
    'n_estimators': [20, 50, 100],
    'learning_rate': [0.001, 0.01, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3] # base estimator parameter: DT
    }
tunnAB = RandomizedSearchCV(AdaBoostClassifier(base_estimator=base_estimator, random_state=1), 
                            params_AB, cv = 5, random_state=2, error_score='raise', scoring='f1')
tunnAB.fit(new_train_x, new_train_y)
print('\nBest parameters :',tunnAB.best_params_)

ab = tunnAB.best_estimator_
ab_wf = ab.fit(new_train_x, new_train_y)

ab_wf_imp = pd.DataFrame({'Feature':new_train_x.columns,
                          'Importance':ab_wf.feature_importances_})
print('\nFeature importance (AB):\n',ab_wf_imp.sort_values(by=['Importance'],ascending=False).head(5))

#Store in place
all_importances_pca['AB'] = ab_wf.feature_importances_
#%%%
train_pred_ab = ab_wf.predict(new_train_x)
train_conf_ab = pd.crosstab(new_train_y,train_pred_ab,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Training (AB):\n',train_conf_ab)
ab_train_accu = metrics.accuracy_score(new_train_y, train_pred_ab,normalize = True)
ab_train_recall = metrics.recall_score(new_train_y, train_pred_ab, pos_label=1, average='binary')
ab_train_precision = metrics.precision_score(new_train_y, train_pred_ab, pos_label=1, average='binary')
ab_train_fmeasure = metrics.f1_score(new_train_y, train_pred_ab, pos_label=1, average='binary')
print('\nAccuracy:', ab_train_accu)
print('\nRecall:', ab_train_recall)
print('\nPrecision:', ab_train_precision)
print('\nFmeasure:', ab_train_fmeasure)

#Predict the test subset
test_pred_ab = ab_wf.predict(new_test_x)
test_conf_ab = pd.crosstab(new_test_y, test_pred_ab,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing (AB):\n',test_conf_ab)
ab_test_accu = metrics.accuracy_score(new_test_y, test_pred_ab)
ab_test_recall = metrics.recall_score(new_test_y, test_pred_ab, pos_label=1, average='binary')
ab_test_precision = metrics.precision_score(new_test_y, test_pred_ab, pos_label=1, average='binary')
ab_test_fmeasure = metrics.f1_score(new_test_y, test_pred_ab, pos_label=1, average='binary')
print('\nAccuracy:', ab_test_accu )
print('\nRecall:', ab_test_recall)
print('\nPrecision:', ab_test_precision)
print('\nFmeasure:', ab_test_fmeasure)
#%%% Not use
AB_proba = ab_wf.predict_proba(wf_test_x)[:, 1]
# สร้างตารางเปรียบเทียบค่า Threshold
threshold_results = []
for th in [0.35, 0.4, 0.45, 0.48, 0.49, 0.5]:
    temp_pred = (AB_proba >= th).astype(int)
    rec = metrics.recall_score(wf_test_y_num, temp_pred)
    acc = metrics.accuracy_score(wf_test_y_num, temp_pred)
    f1 = metrics.f1_score(wf_test_y_num, temp_pred)
    threshold_results.append((th, rec, acc, f1))

df_AB = pd.DataFrame(threshold_results, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])
print(df_AB)

print('\nConfusion Matrix Testing (AB):\n',test_conf_ab)
best_pred = (AB_proba >= 0.49).astype(int) 
print('\nAccuracy (th 0.48):', metrics.accuracy_score(wf_test_y_num, best_pred))
print('\nRecall (th 0.48):', metrics.recall_score(wf_test_y_num, best_pred))
print('\nPrecision (th 0.48):', metrics.precision_score(wf_test_y_num, best_pred))
print('\nFmeasure (th 0.48):', metrics.f1_score(wf_test_y_num, best_pred))
#%%
## SVM+PCA
parameters = {'kernel': ['rbf'], 
              'C': [0.1, 1, 10, 100], 
              'gamma': [1e-4, 1e-3, 0.01, 0.1], # ใช้ค่าที่ละเอียดขึ้น
              'class_weight': ['balanced'] # สำคัญมาก: ช่วยแก้ปัญหาข้อมูลไม่สมดุล
              }
grid = GridSearchCV(SVC(random_state=516),param_grid=parameters, cv=3, scoring='f1')

grid.fit(new_train_x, new_train_y)
grid.cv_results_, grid.best_params_, grid.best_score_
print('\nBest parameters :', grid.best_params_)

svc = grid.best_estimator_
# = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], class_weight='balanced', random_state=516)
svc.fit(new_train_x, new_train_y)
#SVM cant identify KPI !!!
#%% TRAIN PCA
pred_SVC = svc.predict(new_train_x)
conf_SVC = pd.crosstab(new_train_y.values.flatten(),pred_SVC,rownames=['real'],colnames=['pred'])

print('\nConfusion Matrix Training:\n',conf_SVC)
print('\nAccuracy= ', metrics.accuracy_score(new_train_y, pred_SVC))
SVC_p=metrics.accuracy_score(new_train_y, pred_SVC)
print('\nRecall:',metrics.recall_score(new_train_y, pred_SVC, pos_label=1, average='binary'))
print('\nPrecision:',metrics.precision_score(new_train_y, pred_SVC, pos_label=1, average='binary'))
print('\nFmeasure:',metrics.f1_score(new_train_y, pred_SVC, pos_label=1, average='binary'))

#TEST PCA
testpred_svc = svc.predict(new_test_x)
testconf_svc = pd.crosstab(new_test_y.values.flatten(),testpred_svc,rownames=['real'],colnames=['pred'])
print('\nConfusion Matrix Testing:\n', testconf_svc)
print('\nAccuracy:',metrics.accuracy_score(new_test_y,testpred_svc,normalize = True))
print('\nRecall:', metrics.recall_score(new_test_y, testpred_svc, pos_label=1, average='binary'))
print('\nPrecision:', metrics.precision_score(new_test_y, testpred_svc, pos_label=1, average='binary'))
print('\nFmeasure:', metrics.f1_score(new_test_y, testpred_svc, pos_label=1, average='binary'))
#%%%     #Select Common KPIs (Cumulative Average Importance > 85%) from RF, XGB, AB
print("\n''''''''''''Common KPI Selection''''''''''''''''")
pca_feature = new_train_x.columns

# Create DataFrame to accumulate total Feature Importance from two models
pcaimportance_df = pd.DataFrame({
    'Feature': pca_feature,
    'RF_Importance': all_importances_pca['RF'],
    'XGB_Importance': all_importances_pca['XGB'],
    'AB_Importance': all_importances_pca['AB']
})

#Average degree of importance
pcaimportance_df['Average_Importance'] = pcaimportance_df[['AB_Importance', 'XGB_Importance', 'RF_Importance']].mean(axis=1)

#Sort from H to L
pcaimportance_df = pcaimportance_df.sort_values(by='Average_Importance', ascending=False).reset_index(drop=True)

#Calculate Cumulative Importance
pcaimportance_df['Normalized_Avg_Importance'] = pcaimportance_df['Average_Importance'] / pcaimportance_df['Average_Importance'].sum()
pcaimportance_df['Cumulative_Avg_Importance'] = pcaimportance_df['Normalized_Avg_Importance'].cumsum()
print('\nAverage Feature importance (Sorted):\n')

#Add Average_Importance column in matrix
print(pcaimportance_df[['Feature', 'Average_Importance', 'Normalized_Avg_Importance', 'Cumulative_Avg_Importance']])

#Selected Common KPIs: Cumulative average importance > 85% == capture 85% of impartance ก็คือจุดใต้ 85% ลงมา
kpi_features = pcaimportance_df[pcaimportance_df['Cumulative_Avg_Importance'] <= 0.80]

print("\n--- Selected Common KPIs (Cumulative Average Importance > 80%) ---")
print(f"Number of Selected Features: {len(kpi_features)}")
print("Selected Features:", list(kpi_features['Feature']))
#%%  #Distribution Matrix of Y/N of these KPIs
# 1. รายชื่อ Selected Features (จากผล PCA)
selected_kpis = ['pc9', 'pc8', 'pc7', 'pc3', 'pc4', 'pc2']
# 2. เตรียมข้อมูล
temp_df = new_train_x[selected_kpis].copy()
temp_df['Join_Status'] = new_train_y.values

tables = []
print(f"{'--- Distribution Matrix by Quartiles (%) ---':^50}")

# 3. วนลูปเพื่อแบ่งกลุ่มเป็น Quartiles และทำ Crosstab
for col in selected_kpis:
    # แบ่งกลุ่มเป็น 4 ส่วน
    temp_df['Quartile'] = pd.qcut(temp_df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # สร้าง Crosstab และแปลงเป็นร้อยละตามแถว
    ct = pd.crosstab(temp_df['Quartile'], temp_df['Join_Status'])
    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    
    # พิมพ์ชื่อ Feature และขีดเส้นใต้
    print(f"\nFeature: {col}")
    print("-" * 30)
    print(ct_pct)
    print("-" * 30)
    
    # เก็บค่าลง tables สำหรับใช้รวมเป็นตารางเดียวภายหลัง (ถ้าต้องการ)
    ct_pct.index = pd.MultiIndex.from_product([[col], ct_pct.index])
    tables.append(ct_pct)

# 4. รวมเป็นตารางใหญ่ตารางเดียว (Optional - สำหรับนำไปประมวลผลต่อ)
combined_quartiles = pd.concat(tables)
#%%%%
# เตรียมตารางเก็บผลลัพธ์
stat_results = []

for col in selected_kpis:
    # แยกกลุ่มข้อมูลตาม Join_Status
    group0 = temp_df[temp_df['Join_Status'] == 0][col]
    group1 = temp_df[temp_df['Join_Status'] == 1][col]
    
    # 1. F-test เพื่อเช็ค Variance
    f_stat, f_p = stats.levene(group0, group1) # ใช้ Levene's test เสถียรกว่า F-test ปกติ
    
    # 2. T-test (เลือกใช้ equal_var ตามผลจาก Levene's test)
    is_equal_var = True if f_p > 0.05 else False
    t_stat, t_p = stats.ttest_ind(group0, group1, equal_var=is_equal_var)
    
    stat_results.append({
        'Feature': col,
        'F-test p-value': round(f_p, 4),
        'T-test p-value': round(t_p, 4),
        'Is_Significant': 'YES' if t_p < 0.05 else 'NO'
    })

# แสดงผลสรุป
stat_df = pd.DataFrame(stat_results)
print("--- Statistical Test Results (p-value < 0.05 to Confirm KPI) ---")
print(stat_df)
#%%%
models = {
    'Random Forest': rf_wf,
    'XGBoost': xgb_wf,
    'AdaBoost': ab_wf,
    'SVC': svc
}

plt.figure(figsize=(10, 6))

for name, model in models.items():
    # ดึงความน่าจะเป็นของคลาส 1
    if name == 'SVC':
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(new_test_x)[:, 1]
        else:
            y_prob = model.decision_function(new_test_x)
    else:
        y_prob = model.predict_proba(new_test_x)[:, 1]
    
    auc_score = roc_auc_score(new_test_y, y_prob)
    fpr, tpr, _ = roc_curve(new_test_y, y_prob)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Comparison (PCA Features)')
plt.legend()
plt.show()
#%%%
# สร้าง List ของ PC ที่เราต้องการดูค่าสถิติ
target_pcs = ['pc7']

print("--- Real Value Ranges for Quartiles (KPI Analysis) ---")

for pc in target_pcs:
    # คำนวณค่า Quartiles (0%, 25%, 50%, 75%, 100%)
    q_values = new_x[pc].quantile([0, 0.25, 0.5, 0.75, 1.0])
    
    print(f"\n[ Feature: {pc} ]")
    print(f"Q1 Range (Lowest) : {q_values[0]:.4f} to {q_values[0.25]:.4f}")
    print(f"Q2 Range          : {q_values[0.25]:.4f} to {q_values[0.5]:.4f}")
    print(f"Q3 Range          : {q_values[0.5]:.4f} to {q_values[0.75]:.4f}")
    print(f"Q4 Range (Highest): {q_values[0.75]:.4f} to {q_values[1.0]:.4f}")

# เสริม: ฟังก์ชันสำหรับดูความสัมพันธ์ระหว่าง PC กับ Sensors เดิม (Loading Analysis)
# เพื่อให้รู้ว่าค่า PC สูง/ต่ำ หมายถึง Sensor ตัวไหนสูง/ต่ำ
print("\n--- Correlation with Original Sensors (Loading Check) ---")
for pc in target_pcs:
    top_sensor = eigen_vector[pc].sort_values(ascending=False)
    print(f"\nTop Drivers for {pc}:")
    print(f"  Positive (+): {top_sensor.index[0]} ({top_sensor.iloc[0]:.2f})")
    print(f"  Negative (-): {top_sensor.index[-1]} ({top_sensor.iloc[-1]:.2f})")
#%%%
ref_df = pd.DataFrame({
    'Sensor_Name': wf_x.columns,
    'Mean_Original': scaler.mean_,
    'Std_Original': scaler.scale_
})

# 1. กำหนดช่วง Quartile Edges จากรูปภาพ
pc_edges = {
    'pc7': [-3.4987, -0.6794, 0.0003, 0.6715, 3.6572],
    'pc9': [-3.3629, -0.6708, 0.0287, 0.6757, 3.4149]
}

# 2. ดึงค่าจากตารางอ้างอิงและ Eigen Vector
def analyze_top_2_drivers(pc_name, z_values):
    # หา 2 อันดับแรกที่ส่งผลต่อ PC นี้ (ใช้ค่า absolute เพื่อดูอิทธิพลสูงสุด)
    top_2_sensors = eigen_vector[pc_name].abs().sort_values(ascending=False).head(5).index
    
    print(f"\n{'='*50}")
    print(f"  ANALYSIS FOR {pc_name.upper()}")
    print(f"{'='*50}")
    
    labels = ['Min', 'Q1/Q2 Edge', 'Median', 'Q3/Q4 Edge', 'Max']
    
    for sensor in top_2_sensors:
        loading = eigen_vector.loc[sensor, pc_name]
        mean = ref_df.loc[ref_df['Sensor_Name'] == sensor, 'Mean_Original'].values[0]
        std = ref_df.loc[ref_df['Sensor_Name'] == sensor, 'Std_Original'].values[0]
        
        print(f"\n>> Driver: {sensor}")
        print(f"   Loading Value: {loading:.4f}")
        
        # คำนวณค่าจริง: (Z * Std) + Mean
        for label, z in zip(labels, z_values):
            actual_val = (z * std) + mean ## !!!ไม่ได้คูณ loading เพื่อปรับสัดส่วน original feature value ตามค่าที่มันถูกจับใน PC7 สักนิด
            print(f"   {label:12} : {actual_val:.4f}")

# รันการวิเคราะห์
analyze_top_2_drivers('pc7', pc_edges['pc7'])

print("\n--- Top 5 Sensors for PC7 ---")
print(eigen_vector['pc7'].abs().sort_values(ascending=False).head(5))
#%%
#มีทำ StandardScaler() ก่อนทำ PCA ดังนั้นเวลาจะแปลงกลับ ต้องผ่าน 2 ด่าน คือ 
#(1) ย้อนกลับจาก PC Score เป็น Standardized Value
#(2) ย้อนกลับจาก Standardized เป็น Original Value
def get_original_values_from_pc(pc_name, pc_score, eigen_vector, scaler, top_n=5):
    # 1. ดึง Loading ของ PC นั้นๆ
    loadings = eigen_vector[pc_name]
    
    # 2. ดึง Mean และ Std ดั้งเดิมจาก scaler
    means = scaler.mean_
    stds = scaler.scale_
    sensor_names = eigen_vector.index
    
    results = []
    
    for i, sensor in enumerate(sensor_names):
        loading = loadings[sensor]
        
        mean = means[i]
        std = stds[i]
        
        # ✅ หัวใจสำคัญ: แปลงกลับ
        # (pc_score * loading) คือการประมาณค่า Z-score ของ sensor นั้นที่ถูกอธิบายโดย PC นี้
        predicted_original = (pc_score * loading * std) + mean
        
        results.append({
            'Sensor': sensor,
            'Loading': loading,
            'Predicted_Original': predicted_original
        })
    
    # แปลงเป็น DataFrame และเรียงลำดับตามความสำคัญ (Loading Absolute)
    res_df = pd.DataFrame(results)
    res_df['Abs_Loading'] = res_df['Loading'].abs()
    return res_df.sort_values(by='Abs_Loading', ascending=False).head(top_n)

# --- เรียกใช้งาน ---
#'pc7': [-3.4987, -0.6794, 0.0003, 0.6715, 3.6572]
pc7_value = 0.0003
impact_df = get_original_values_from_pc('pc7', pc7_value, eigen_vector, scaler)

print(f"--- Estimated Sensor Values when pc7 = {pc7_value} ---")
print(impact_df[['Sensor', 'Loading', 'Estimated_Original']])
#%%
def calculate_sensor_thresholds(pc_name, pc_range, eigen_vector, scaler, top_n=5):
    loadings = eigen_vector[pc_name]
    means = scaler.mean_
    stds = scaler.scale_
    
    top_sensors_idx = loadings.abs().sort_values(ascending=False).head(top_n).index
    results = []
    
    for sensor in top_sensors_idx:
        # ใช้ .index.get_loc เพื่อหาลำดับที่ของ Sensor ใน Scaler
        idx = eigen_vector.index.get_loc(sensor)
        
        loading = loadings[sensor]
        mean = means[idx]
        std = stds[idx]
        
        # คำนวณค่าขอบเขต (Threshold)
        val_start = (pc_range[0] * loading * std) + mean
        val_end = (pc_range[1] * loading * std) + mean
        
        results.append({
            'Sensor': sensor,
            'Loading': loading,
            'Threshold_Start': val_start,
            'Threshold_End': val_end
        })
    
    return pd.DataFrame(results)

# รันโค้ด
pc7_q2_range = [-0.6794, 0.0003]
threshold_df = calculate_sensor_thresholds('pc7', pc7_q2_range, eigen_vector, scaler)

print(threshold_df[['Sensor', 'Loading', 'Threshold_Start', 'Threshold_End']])