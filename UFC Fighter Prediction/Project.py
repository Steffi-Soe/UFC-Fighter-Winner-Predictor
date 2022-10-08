# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:56:50 2022

@author: steff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Supaya bisa melihat ddata frame yang lebih luas
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None

df = pd.read_csv("data.csv")

# Pembuangan kolum tidak berguna baik untuk aplikasi maupun model
df = df.drop(df.columns[[2, 3, 4, 7]], axis = 1)

# replace string kosong dengan nan
df = df.replace(r'^\s*$', np.nan, regex=True) 

# Isi frame kosong dengan Nan
df = df.fillna(np.nan) 

# Penamaan kembali sehingga gampang diingat
df.rename(columns={
    "B_win_by_KO/TKO": "B_win_by_KO_TKO",
    "R_win_by_KO/TKO": "R_win_by_KO_TKO"},inplace=True)

#drop nan di kolum R_Stance dan B_Stance
df.dropna(subset=["R_Stance", "B_Stance"], inplace=True, axis=0) 

# Pembuangan kolum yang terlalu konstan atau kolum yang variasinya cukup kecil misalnya seperti R_win_by_TKO_Doctor_Stoppage
def remove_constant_data(df):
    categorical = list(df.select_dtypes(include=['object']))
    numeric = df.columns.tolist()
    for c in categorical:
        numeric.remove(c)
    
    constants = []
    for col in numeric:
        if min(df[col]) == max(df[col]):
            constants.append(col)
    if len(constants)>0: 
        return df.drop(columns = constants, axis=1, inplace=True)
    
remove_constant_data(df)

# Membuang kolum streak karena kebanyakan data mendekati konstan
df = df.drop(df.columns[[54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131]], axis = 1)

# Mengubah kolum B_Stance dan R_Stance menjadi 
transformer = [54, 108]
df_transformer = df.iloc[:, transformer]
df_transformer = pd.get_dummies(df_transformer)

# Namain ulang biar konsisten dan g lupa
df_transformer.rename(columns={"B_Stance_Open Stance": "B_Stance_Open_Stance","R_Stance_Open Stance": "R_Stance_Open_Stance"},inplace=True)
df.drop(df.columns[transformer], axis = 1, inplace = True)
df = pd.concat([df,df_transformer], axis = 1)

# Untuk melihat distribusi dari data yang ada untuk menentukan bagaimana mengisi data kosong
import seaborn as sns
# sns.heatmap(df.isna().transpose()) #Jangan lupa diuncomment

# Mengisi nilai kosong pada Height dengan mediannya karena data yang kebanyakan Nan terdapat pada bagian belakang
df["B_Height_cms"] = df["B_Height_cms"].fillna(value=df["B_Height_cms"].median())
df["R_Height_cms"] = df["R_Height_cms"].fillna(value=df["R_Height_cms"].median())

# Untuk Weight digunakan mean karena data stabil dan hanya 2 baris pada B_Weights_lbs yang hilang
df["B_Weight_lbs"] = df["B_Weight_lbs"].fillna(value = df["B_Weight_lbs"].mean())

# Mencari Nilai reach dengan linear regression
# Pengambilan data height dapat menentukan reach
# Karena tinggi secara logika berpengaruh pada panjang tangan maka hanya height yang diambil weight tidak 
# termasuk
measure = [54, 55, 107, 108]
df_pisah = df.iloc[:,measure]
df_pisah.dropna(subset = ['B_Reach_cms', 'R_Reach_cms'], inplace = True, axis = 0)

# Pemisahan antara height dan weight untuk mencari reach
height = list(pd.concat([df_pisah["B_Height_cms"],df_pisah["R_Height_cms"]], axis = 0))
reach = list(pd.concat([df_pisah["B_Reach_cms"],df_pisah["R_Reach_cms"]], axis = 0))

df_pisah = pd.DataFrame(data = {'height':height, 'reach' : reach})
# Pelatihan model Height ke reach
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = df_pisah['height'].values.reshape(-1, 1)
y = df_pisah['reach'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

# Pencarian akurasi dengan r^2 score
from sklearn.metrics import r2_score
acc = r2_score(y_test, y_pred)

# Karena sudah mencapai sekitar 80% maka linear regression dapat dipakai
# Membuat fungsi untuk mengisi nilai kosong
def insert_reach(arr):
    return lr.predict(np.array(arr.tolist()).reshape(-1,1))
df["R_Reach_cms"][pd.isna(df["R_Reach_cms"])] = insert_reach(df[pd.isna(df["R_Reach_cms"])]["R_Height_cms"])
df["B_Reach_cms"][pd.isna(df["B_Reach_cms"])] = insert_reach(df[pd.isna(df["B_Reach_cms"])]["B_Height_cms"])

# Pengisian B_age dan R_age
df["B_age"] = df["B_age"].fillna(value=df["B_age"].mean())
df["R_age"] = df["R_age"].fillna(value=df["R_age"].mean())

# Pemisahan data agar dapat mengisi nilai dari kolum yang kosong
df_pisah = df.iloc[:,4:53]
df_pisah = pd.concat([df_pisah,df.iloc[:,57:106]],axis = 1)

# Pemisahan kelist kemudian di convert ke series dan diremove data nan kemudian akan dimasukkan ke dalam
# akan dicari mean karena data tersebar cukup rapi
for i in range (0, 49):
    avg = list(pd.concat([df_pisah.iloc[:,i], df_pisah.iloc[:,i + 49]], axis = 0))
    avg = pd.Series(avg)
    df[df.columns[i + 4]] = df[df.columns[i + 4]].fillna(value = avg.median())
    df[df.columns[i + 57]] = df[df.columns[i + 57]].fillna(value = avg.median())

# Pada data Winner, terdapat 1 isi yakni draw dimana ini tidak akan dihitung karena pada umumnya UFC
# Hanya memiliki 1 Pemenang sehingga row tersebut akan dibuang
df['Winner'] = df['Winner'].replace("Draw",np.nan)
df.dropna(subset=["Winner"], inplace=True, axis=0) 

df['Winner'] = df['Winner'].replace(["Blue", "Red"], [1, 0]).values

# Untuk masalah scaling data kedepannya maka column title bout
df = df.drop(df.columns[[3]], axis = 1)

# Pembagian variabel independent dan dependent
y = df['Winner']
x = df.drop(df.columns[[0,1,2]], axis = 1)

# Pemisahan Training set dan Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

# Scaling data
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

# Pelatihan Model Logistic Regression, KNN dan SVM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

model1 = LogisticRegression(max_iter = 5000)
model1.fit(x_train, y_train)
y_pred_lr = model1.predict(x_test)

model2 = KNeighborsClassifier(n_neighbors = 11, p = 2, metric = 'euclidean')
model2.fit(x_train, y_train)
y_pred_KNN = model2.predict(x_test)

model3 = SVC(kernel = 'rbf', gamma = 0.7)
model3.fit(x_train, y_train)
y_pred_svm = model3.predict(x_test)

y_pred_lr_train = model1.predict(x_train)
y_pred_knn_train = model2.predict(x_train)
y_pred_svm_train = model3.predict(x_train)

# Pengecekan Akurasi
from sklearn.metrics import accuracy_score, precision_score

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_KNN)
acc_svm = accuracy_score(y_test, y_pred_svm)

prec_lr = precision_score(y_test, y_pred_lr)
prec_knn = precision_score(y_test, y_pred_KNN)
prec_svm = precision_score(y_test, y_pred_svm)

acc_lr_train = accuracy_score(y_train, y_pred_lr_train)
acc_knn_train = accuracy_score(y_train, y_pred_knn_train)
acc_svm_train = accuracy_score(y_train, y_pred_svm_train)

prec_lr_train = precision_score(y_train, y_pred_lr_train)
prec_knn_train = precision_score(y_train, y_pred_knn_train)
prec_svm_train = precision_score(y_train, y_pred_svm_train)

# Karena akurasi yang kurang, maka akan dibuat decision tree classifier dimana decision tree ini akan
# menyatukan hasil yang didapatkan sebelumnya dan menentukan hasil yang ada

# Penggabungan kembali hasil prediksi
result = pd.DataFrame(data = {'Logistic_Regression' : y_pred_lr, 'KNN' : y_pred_KNN, 'SVM' : y_pred_svm, 'True_Value' : y_test})
result_train = pd.DataFrame(data = {'Logistic_Regression' : y_pred_lr_train, 'KNN' : y_pred_knn_train, 'SVM' : y_pred_svm_train, 'True_Value' : y_train})

# Pembagian data result
x = result.drop(result.columns[[3]], axis = 1)
y = result['True_Value']

x1 = result_train.drop(result_train.columns[[3]], axis = 1)
y1 = result_train['True_Value']

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 0)
model.fit(x1, y1)

y_pred = model.predict(x)
y_pred_train = model.predict(x1)

acc = accuracy_score(y, y_pred)
acc_pred = accuracy_score(y1, y_pred_train)

prec = accuracy_score(y, y_pred)
prec_pred = accuracy_score(y1, y_pred_train)

# Pengecekan data
result1 = pd.DataFrame(data = {'y_pred' : y_pred, 'y_true' : y})

df['R_fighter'] = df['R_fighter'].str.lower()
df['B_fighter'] = df['B_fighter'].str.lower()

# Display
from tkinter import *
from tkinter import ttk

def next():
    # Penyimpanan nama petarung dari aplikasi
    red_name = Red.get()
    blue_name = Blue.get()
    
    # Mengambil 5 data terupdate dari masing-masing petarung
    red_temp = df[(df['R_fighter'] == red_name.lower()) | (df['B_fighter'] == red_name.lower())].head(5)
    blue_temp = df[(df['R_fighter'] == blue_name.lower()) | (df['B_fighter'] == blue_name.lower())].head(5)
    
    # Untuk mengecek apakah ada petarung yang tidak terdaftar atau tidak diisi
    if ((red_temp.empty == True) & (blue_temp.empty == True)):
        textarea.insert(END, "Petarung pojok merah dan biru tidak terdapat dalam daftar\n\n")
        return
    elif (red_temp.empty == True):
        textarea.insert(END, "Petarung pojok merah tidak terdapat dalam daftar\n\n")
        return
    elif (blue_temp.empty == True):
        textarea.insert(END, "Petarung pojok biru tidak terdapat dalam daftar\n\n")
        return
    else:
        textarea.insert(END, "Yang berada di pojok kanan!!!\n" + red_name + "\nYang berada di pojok kiri!!!\n" + blue_name + "\n\n")

    # Pengelolaan data petarung
    r_idx = red_temp.shape
    b_idx = blue_temp.shape
    
    red_r = pd.DataFrame()
    red_b = pd.DataFrame()
    blue_r = pd.DataFrame()
    blue_b = pd.DataFrame()
    
    if_red = [0, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 116, 117, 118, 119, 120]
    if_blue = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 , 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 109, 111, 112, 113, 114, 115]
    
    for i in range (0, b_idx[0]):
        if (blue_temp['R_fighter'].iloc[i,] == blue_name.lower()):
            blue_r = pd.concat([blue_r, blue_temp.iloc[i, if_red]], axis = 1)
        else:
            blue_b = pd.concat([blue_b, blue_temp.iloc[i, if_blue]], axis = 1)
    
    for i in range (0,r_idx[0]):
        if(red_temp['R_fighter'].iloc[i,] == red_name.lower()):
            red_r = pd.concat([red_r, red_temp.iloc[i, if_red]], axis = 1)
        else:
            red_b = pd.concat([red_b, red_temp.iloc[i, if_blue]], axis = 1)
    
    blue_columns = ['B_fighter', 'B_avg_KD', 'B_avg_opp_KD', 'B_avg_SIG_STR_pct', 
                    'B_avg_opp_SIG_STR_pct', 'B_avg_TD_pct', 'B_avg_opp_TD_pct', 'B_avg_SUB_ATT', 
                    'B_avg_opp_SUB_ATT', 'B_avg_REV', 'B_avg_opp_REV', 'B_avg_SIG_STR_att', 
                    'B_avg_SIG_STR_landed', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 
                    'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_avg_opp_TOTAL_STR_att', 
                    'B_avg_opp_TOTAL_STR_landed', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_opp_TD_att', 
                    'B_avg_opp_TD_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_opp_HEAD_att', 
                    'B_avg_opp_HEAD_landed', 'B_avg_BODY_att', 'B_avg_BODY_landed', 
                    'B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_LEG_att', 'B_avg_LEG_landed', 
                    'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_DISTANCE_att', 
                    'B_avg_DISTANCE_landed', 'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 
                    'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_opp_CLINCH_att', 
                    'B_avg_opp_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 
                    'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_CTRL_time(seconds)', 
                    'B_avg_opp_CTRL_time(seconds)', 'B_total_time_fought(seconds)', 
                    'B_total_rounds_fought', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs', 'B_age', 
                    'B_Stance_Open_Stance', 'B_Stance_Orthodox', 'B_Stance_Sideways', 
                    'B_Stance_Southpaw', 'B_Stance_Switch']
    
    red_columns = ['R_fighter', 'R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct',
                   'R_avg_TD_pct', 'R_avg_opp_TD_pct', 'R_avg_SUB_ATT', 'R_avg_opp_SUB_ATT', 'R_avg_REV', 
                   'R_avg_opp_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 
                   'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_TOTAL_STR_att',
                   'R_avg_TOTAL_STR_landed', 'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed', 
                   'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_opp_TD_att', 'R_avg_opp_TD_landed', 
                   'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed', 
                   'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 
                   'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 
                   'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_opp_DISTANCE_att',
                   'R_avg_opp_DISTANCE_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 
                   'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_GROUND_att',
                   'R_avg_GROUND_landed', 'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed',
                   'R_avg_CTRL_time(seconds)', 'R_avg_opp_CTRL_time(seconds)', 
                   'R_total_time_fought(seconds)', 'R_total_rounds_fought', 'R_Height_cms', 
                   'R_Reach_cms', 'R_Weight_lbs', 'R_age', 'R_Stance_Open_Stance', 'R_Stance_Orthodox', 
                   'R_Stance_Sideways', 'R_Stance_Southpaw', 'R_Stance_Switch']
    
    if(red_r.empty == False):
        red_r = red_r.transpose()
    if(red_b.empty == False):
        red_b = red_b.transpose()
    if(blue_r.empty == False):
        blue_r = blue_r.transpose()
    if(blue_b.empty == False):
        blue_b = blue_b.transpose()
    
    # Rename hasil pemisahan
    if(red_b.empty == False):
        red_b.columns = red_columns
    if(blue_r.empty == False):
        blue_r.columns = blue_columns
    
    if (red_r.empty == True):
        red = red_b
    elif(red_r.empty == True):
        red = red_r
    else:
        red = pd.concat([red_r,red_b], axis = 0)
        
    if(blue_r.empty == True):
        blue = blue_b
    elif(blue_b.empty == True):
        blue = blue_r
    else:
        blue = pd.concat([blue_b, blue_r], axis = 0)
    
    red = red.drop(red.columns[[0]], axis = 1)
    blue = blue.drop(blue.columns[[0]],axis = 1)
    
    #B Data, B Measure, R Data, R Measure, B Age, R Age, B Stance, R _Stance
    red_after = pd.DataFrame(red.mean())
    red_after = red_after.transpose()
    red_after = red_after.drop(red_after.columns[[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]], axis = 1)
    
    blue_after = pd.DataFrame(blue.mean())
    blue_after = blue_after.transpose()
    blue_after = blue_after.drop(blue_after.columns[[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]], axis = 1)
    
    for i in range(48, 59):
        red_after.loc[0, i] = red.iloc[0, i]
    
    for i in range (48, 59):
        blue_after.loc[0, i] = blue.iloc[0, i]
    
    red_columns = ['R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct',
                   'R_avg_TD_pct', 'R_avg_opp_TD_pct', 'R_avg_SUB_ATT', 'R_avg_opp_SUB_ATT', 'R_avg_REV', 
                   'R_avg_opp_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 
                   'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_TOTAL_STR_att',
                   'R_avg_TOTAL_STR_landed', 'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed', 
                   'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_opp_TD_att', 'R_avg_opp_TD_landed', 
                   'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed', 
                   'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 
                   'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 
                   'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_opp_DISTANCE_att',
                   'R_avg_opp_DISTANCE_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 
                   'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_GROUND_att',
                   'R_avg_GROUND_landed', 'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed',
                   'R_avg_CTRL_time(seconds)', 'R_avg_opp_CTRL_time(seconds)', 
                   'R_total_time_fought(seconds)', 'R_total_rounds_fought', 'R_Height_cms', 
                   'R_Reach_cms', 'R_Weight_lbs', 'R_age', 'R_Stance_Open_Stance', 'R_Stance_Orthodox', 
                   'R_Stance_Sideways', 'R_Stance_Southpaw', 'R_Stance_Switch']
    
    blue_columns = ['B_avg_KD', 'B_avg_opp_KD', 'B_avg_SIG_STR_pct', 
                    'B_avg_opp_SIG_STR_pct', 'B_avg_TD_pct', 'B_avg_opp_TD_pct', 'B_avg_SUB_ATT', 
                    'B_avg_opp_SUB_ATT', 'B_avg_REV', 'B_avg_opp_REV', 'B_avg_SIG_STR_att', 
                    'B_avg_SIG_STR_landed', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 
                    'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_avg_opp_TOTAL_STR_att', 
                    'B_avg_opp_TOTAL_STR_landed', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_opp_TD_att', 
                    'B_avg_opp_TD_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_opp_HEAD_att', 
                    'B_avg_opp_HEAD_landed', 'B_avg_BODY_att', 'B_avg_BODY_landed', 
                    'B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_LEG_att', 'B_avg_LEG_landed', 
                    'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_DISTANCE_att', 
                    'B_avg_DISTANCE_landed', 'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 
                    'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_opp_CLINCH_att', 
                    'B_avg_opp_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 
                    'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_CTRL_time(seconds)', 
                    'B_avg_opp_CTRL_time(seconds)', 'B_total_time_fought(seconds)', 
                    'B_total_rounds_fought', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs', 'B_age', 
                    'B_Stance_Open_Stance', 'B_Stance_Orthodox', 'B_Stance_Sideways', 
                    'B_Stance_Southpaw', 'B_Stance_Switch']
    
    red_after.columns = red_columns
    blue_after.columns = blue_columns
    
    combine = blue_after.iloc[:, :53]
    
    for i in range (0, 53):
        combine.loc[:, i + 52] = red_after.iloc[:, i]
    
    combine.loc[:, 105] = blue_after.iloc[:, 53]
    combine.loc[:, 106] = red_after.iloc[:, 53]
    
    for i in range(54,59):
        combine.loc[:, i + 53] = blue_after.iloc[:, i]
        
    for i in range(54, 59):
        combine.loc[:, i + 58] = red_after.iloc[:, i]
        
    combine_columns = ['B_avg_KD', 'B_avg_opp_KD', 'B_avg_SIG_STR_pct', 
                    'B_avg_opp_SIG_STR_pct', 'B_avg_TD_pct', 'B_avg_opp_TD_pct', 'B_avg_SUB_ATT', 
                    'B_avg_opp_SUB_ATT', 'B_avg_REV', 'B_avg_opp_REV', 'B_avg_SIG_STR_att', 
                    'B_avg_SIG_STR_landed', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 
                    'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_avg_opp_TOTAL_STR_att', 
                    'B_avg_opp_TOTAL_STR_landed', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_opp_TD_att', 
                    'B_avg_opp_TD_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_opp_HEAD_att', 
                    'B_avg_opp_HEAD_landed', 'B_avg_BODY_att', 'B_avg_BODY_landed', 
                    'B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_LEG_att', 'B_avg_LEG_landed', 
                    'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_DISTANCE_att', 
                    'B_avg_DISTANCE_landed', 'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 
                    'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_opp_CLINCH_att', 
                    'B_avg_opp_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 
                    'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_CTRL_time(seconds)', 
                    'B_avg_opp_CTRL_time(seconds)', 'B_total_time_fought(seconds)', 
                    'B_total_rounds_fought', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs',
                    'R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct',
                    'R_avg_TD_pct', 'R_avg_opp_TD_pct', 'R_avg_SUB_ATT', 'R_avg_opp_SUB_ATT', 'R_avg_REV', 
                    'R_avg_opp_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 
                    'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_TOTAL_STR_att',
                    'R_avg_TOTAL_STR_landed', 'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed', 
                    'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_opp_TD_att', 'R_avg_opp_TD_landed', 
                    'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed', 
                    'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 
                    'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 
                    'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_opp_DISTANCE_att',
                    'R_avg_opp_DISTANCE_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 
                    'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_GROUND_att',
                    'R_avg_GROUND_landed', 'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed',
                    'R_avg_CTRL_time(seconds)', 'R_avg_opp_CTRL_time(seconds)', 
                    'R_total_time_fought(seconds)', 'R_total_rounds_fought', 'R_Height_cms', 
                    'R_Reach_cms', 'R_Weight_lbs', 'B_age', 'R_age', 'B_Stance_Open_Stance', 'B_Stance_Orthodox', 
                    'B_Stance_Sideways', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Open_Stance', 
                    'R_Stance_Orthodox', 'R_Stance_Sideways', 'R_Stance_Southpaw', 'R_Stance_Switch']
    
    combine.columns = combine_columns
    combine = mm.transform(combine)
    
    result_pred_lr = model1.predict(combine)
    result_pred_knn = model2.predict(combine)
    result_pred_svm = model3.predict(combine)
    
    result_model = pd.DataFrame(data = {'Logistic_Regression' : result_pred_lr, 'KNN' : result_pred_knn, 'SVM' : result_pred_svm})
    
    result = model.predict(result_model)
    
    if (result[0] == 0):
        textarea.insert(END, "Petarung pojok merah menang!!\nSelamat kepada......." + red_name + "\n\n")
    else:
        textarea.insert(END, "Petarung pojok biru menang!!\nSelamat kepada......." + blue_name + "\n\n")
    textarea.insert(END, "Insert 2 Petarung dengan mengisi kedua kotak yang ada. Petarung pertama/Input pertama berada pada pojok merah dan petarung petarung kedua/Input kedua berada pada pojok biru\n\n")
    textarea.insert(END, "Akurasi dari Aplikasi ini : " + str(acc_pred * 0.8 + acc * 0.2) + "\n\nGunakan aplikasi ini dengan resiko masing-masing\n\n")

root = Tk()

root.geometry('500x570+100+100')
root.title('UFC Prediction')
root.config(bg = '#000000')

chatframe = Frame(root)
chatframe.pack()

scrollbar = Scrollbar(chatframe)
scrollbar.pack(side = RIGHT)

textarea = Text(chatframe, font = ('consolas', '16', 'bold'), height = 12, yscrollcommand = scrollbar.set, wrap = 'word', bg = '#FFFFFF')
textarea.pack()
scrollbar.config(command = textarea.yview())

Red = Entry(root, font = ('consolas', '20'))
Red.pack(pady = 15, fill = X)

vs = Label(root,text = 'VS', font = ('consolas', '20', 'bold'), bg = '#FFFFFF')

Blue = Entry(root, font = ('consolas', '20'))
Blue.pack(pady = 15, fill = X)

button = Button(root, text = "Predict", command = next)
button.pack()

textarea.insert(END, "Insert 2 Petarung dengan mengisi kedua kotak yang ada. Petarung pertama/Input pertama berada pada pojok merah dan petarung petarung kedua/Input kedua berada pada pojok biru\n\n")
textarea.insert(END, "Akurasi dari Aplikasi ini : " + str(acc_pred * 0.8 + acc * 0.2) + "\n\nGunakan aplikasi ini dengan resiko masing-masing\n\n")

root.mainloop()
