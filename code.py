import pandas as pd
import numpy as np
from sktime.classification.interval_based import TimeSeriesForestClassifier


def getdata(name):
  df = pd.read_csv(name)
  accelX = df['accelX']
  accelY = df['accelY']
  accelZ = df['accelZ']
  gyroX = df['gyroX']
  gyroY = df['gyroY']
  gyroZ = df['gyroZ']

  return accelX,accelY,accelZ,gyroX,gyroY,gyroZ

def makedata(name):
  tmp = getdata(name)
  return pd.concat([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5]],ignore_index=True)


def maketraindata(name,i):
  df = pd.DataFrame(index=list(range(10)),columns=[0,1])
  tmp1 = getdata(name)
  for k in range(10):
    tmp2 = pd.concat([tmp1[0][50*k:50*(k+1)],tmp1[1][50*k:50*(k+1)],tmp1[2][50*k:50*(k+1)],tmp1[3][50*k:50*(k+1)],tmp1[4][50*k:50*(k+1)],tmp1[5][50*k:50*(k+1)]],ignore_index=True)
    df.iat[k,0]=tmp2
    df.iat[k,1]=i
  return df


traindata = pd.concat([maketraindata('a.csv',1),maketraindata('b.csv',2),maketraindata('c.csv',3),maketraindata('d.csv',4),maketraindata('e.csv',5)],ignore_index=True)

X=traindata.iloc[:,[0]]
y=np.array(traindata.iloc[:,1]).astype(np.str_)
classifier = TimeSeriesForestClassifier()
classifier.fit(X, y)

testdf = pd.DataFrame(index=list(range(1,25)),columns=[0])
for i in range(1,25):
  testdf.iat[i-1,0]=makedata("test{}.csv".format(i))

clf = classifier.predict(testdf)
ans = ['a' if item == '1' else 'b' if item == '2' else 'c' if item == '3' else 'd' if item == '4' else 'e' for item in clf]
print(ans)