# ROCK-VS-METAL-PREDICTION-MACHINE-LEARNING-

Open In Colab
# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from google.colab import drive  # mounting our drive to google colab
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
# accesing data to google collab
file_path = '/content/drive/MyDrive/Datasets/sonardata.csv'
df=pd.read_csv(file_path, header=None)
df.head()   # returns the top 5 rows of the dataset
0	1	2	3	4	5	6	7	8	9	...	51	52	53	54	55	56	57	58	59	60
0	0.0200	0.0371	0.0428	0.0207	0.0954	0.0986	0.1539	0.1601	0.3109	0.2111	...	0.0027	0.0065	0.0159	0.0072	0.0167	0.0180	0.0084	0.0090	0.0032	R
1	0.0453	0.0523	0.0843	0.0689	0.1183	0.2583	0.2156	0.3481	0.3337	0.2872	...	0.0084	0.0089	0.0048	0.0094	0.0191	0.0140	0.0049	0.0052	0.0044	R
2	0.0262	0.0582	0.1099	0.1083	0.0974	0.2280	0.2431	0.3771	0.5598	0.6194	...	0.0232	0.0166	0.0095	0.0180	0.0244	0.0316	0.0164	0.0095	0.0078	R
3	0.0100	0.0171	0.0623	0.0205	0.0205	0.0368	0.1098	0.1276	0.0598	0.1264	...	0.0121	0.0036	0.0150	0.0085	0.0073	0.0050	0.0044	0.0040	0.0117	R
4	0.0762	0.0666	0.0481	0.0394	0.0590	0.0649	0.1209	0.2467	0.3564	0.4459	...	0.0031	0.0054	0.0105	0.0110	0.0015	0.0072	0.0048	0.0107	0.0094	R
5 rows × 61 columns

df.shape      # returns the rows and columnns of the dataset
(208, 61)
df.describe()    # returns the information about numerical columns
0	1	2	3	4	5	6	7	8	9	...	50	51	52	53	54	55	56	57	58	59
count	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	...	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000	208.000000
mean	0.029164	0.038437	0.043832	0.053892	0.075202	0.104570	0.121747	0.134799	0.178003	0.208259	...	0.016069	0.013420	0.010709	0.010941	0.009290	0.008222	0.007820	0.007949	0.007941	0.006507
std	0.022991	0.032960	0.038428	0.046528	0.055552	0.059105	0.061788	0.085152	0.118387	0.134416	...	0.012008	0.009634	0.007060	0.007301	0.007088	0.005736	0.005785	0.006470	0.006181	0.005031
min	0.001500	0.000600	0.001500	0.005800	0.006700	0.010200	0.003300	0.005500	0.007500	0.011300	...	0.000000	0.000800	0.000500	0.001000	0.000600	0.000400	0.000300	0.000300	0.000100	0.000600
25%	0.013350	0.016450	0.018950	0.024375	0.038050	0.067025	0.080900	0.080425	0.097025	0.111275	...	0.008425	0.007275	0.005075	0.005375	0.004150	0.004400	0.003700	0.003600	0.003675	0.003100
50%	0.022800	0.030800	0.034300	0.044050	0.062500	0.092150	0.106950	0.112100	0.152250	0.182400	...	0.013900	0.011400	0.009550	0.009300	0.007500	0.006850	0.005950	0.005800	0.006400	0.005300
75%	0.035550	0.047950	0.057950	0.064500	0.100275	0.134125	0.154000	0.169600	0.233425	0.268700	...	0.020825	0.016725	0.014900	0.014500	0.012100	0.010575	0.010425	0.010350	0.010325	0.008525
max	0.137100	0.233900	0.305900	0.426400	0.401000	0.382300	0.372900	0.459000	0.682800	0.710600	...	0.100400	0.070900	0.039000	0.035200	0.044700	0.039400	0.035500	0.044000	0.036400	0.043900
8 rows × 60 columns

df[60].value_counts()   # checking count of the outputs( rocks and metals(mine))
M    111
R     97
Name: 60, dtype: int64
# separating data and lables
X= df.drop(columns=60, axis=1)
Y=df[60]
X.shape   # independant variable
(208, 60)
Y.shape   # dependant variables
(208,)
# training and test data
X_train, X_test, Y_train, Y_test=train_test_split( X, Y, test_size=0.1, random_state=2)   #splitting the data into train and test
print(X.shape,X_train.shape,X_test.shape)    # checking the size of original and split data
(208, 60) (187, 60) (21, 60)
# model training  using logistic regression
model=LogisticRegression()    # initiating 
model.fit(X_train,Y_train)     # fitting the data
LogisticRegression()
# evaluating our model
#accuracy on training data
Y_train_pred=model.predict(X_train)   # predicting the actual value
train_data_accuracy=accuracy_score(Y_train,Y_train_pred)     # checking the accuracy of the model on training data
print(train_data_accuracy)  #printing the accuracy score for training data
0.8235294117647058
#accuracy on test data
Y_test_pred=model.predict(X_test)   # predicting the actual value
test_data_accuracy=accuracy_score(Y_test,Y_test_pred)     # checking the accuracy of the model on test data
print(test_data_accuracy)  # printing the accuracy score for test data
0.8571428571428571
#making prediction
input_data=(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
# changing the input_data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)   # making the prediction on a given data
print(prediction)
['R']
#checking for the second case( metal)
input_data=(0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
# changing the input_data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)   # making the prediction on a given data
print(prediction)
['M']
