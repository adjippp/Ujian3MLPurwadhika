import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

model=LinearRegression()
df=pd.read_excel('indo_12_1.xls')
df.columns=['provinsi','1971','1980','1990','1995','2000','2010']
df=df.dropna()
df=df.replace('-',np.NaN)
# df=df.fillna(int(0)).reset_index().drop(columns='index')

plt.style.use('ggplot')
#plot jumlah penduduk Indonesia
def plotJmlIndo(df):
    Indonesia=df['provinsi'].tail(1).values[0]
    #dilakukan transpose tahun menjadi index dan isi provinsi menjadi kolom
    df=df.set_index('provinsi').T
    #index adalah nilai untuk x=(1971 - 2010) dan y adalah value dari kolom variabel Indonesia
    x=df.index.values.astype(int)
    y=df[Indonesia].values.astype(int)
    #ubah x menjadi 2 dimensi agar bisa dilakukan train data
    x=x.reshape(-1,1)
    model.fit(x,y)
    pred=model.predict([[2050]])
    plt.plot(x,y,linestyle='-',color='r',label=Indonesia)
    plt.scatter(x,y,color='r')
    return Indonesia,pred,x,y


#plot jumlah penduduk dari provinsi yang memiliki penduduk terbanyak di tahun 2010
def plotTerbanyak2010(df):
    df=df.drop(df.tail(1).index)
    kotaTerbanyak2010=df['provinsi'][df['2010']==df['2010'].max()].values[0]
    #dilakukan transpose tahun menjadi index dan isi provinsi menjadi kolom
    df=df.set_index('provinsi').T
    #index adalah nilai untuk x=(1971 - 2010) dan y adalah value dari kolom variabel kotaTerbanyak2010
    x=df.index.values.astype(int)
    y=df[kotaTerbanyak2010].values.astype(int)
    #ubah x menjadi 2 dimensi agar bisa dilakukan train data
    x=x.reshape(-1,1)
    model.fit(x,y)
    pred=model.predict([[2050]])
    plt.plot(x,y,linestyle='-',color='g',label=kotaTerbanyak2010)
    plt.scatter(x,y,color='g')
    return kotaTerbanyak2010,pred,x,y

# jumlah penduduk dari provinsi yang memiliki penduduk paling sedikit di tahun 1971
def plotSedikit1971(df):
    df=df.drop(df.tail(1).index)
    kotaSedikit1971=df['provinsi'][df['1971']==df['1971'].min()].values[0]
    #dilakukan transpose tahun menjadi index dan isi provinsi menjadi kolom
    df=df.set_index('provinsi').T
    #index adalah nilai untuk x=(1971 - 2010) dan y adalah value dari kolom variabel kotaSedikit1971
    x=df.index.values.astype(int)
    y=df[kotaSedikit1971].values.astype(int)
    #ubah x menjadi 2 dimensi agar bisa dilakukan train data
    x=x.reshape(-1,1)
    model.fit(x,y)
    pred=model.predict([[2050]])
    plt.plot(x,y,linestyle='-',color='b',label=kotaSedikit1971)
    plt.scatter(x,y,color='b')
    
    return kotaSedikit1971,pred,x,y

def plotBestFitLine(x1,y1,x2,y2,x3,y3):
    model.fit(x1,y1)
    plt.plot(x1,model.predict(x1),linestyle='-',color='y',label='Best Fit Line')
    model.fit(x2,y2)
    plt.plot(x2,model.predict(x2),linestyle='-',color='y')
    model.fit(x3,y3)
    plt.plot(x2,model.predict(x3),linestyle='-',color='y')

# plotJmlIndo(df)
# plotTerbanyak2010(df)
# plotSedikit1971(df)
prov1,pred1,x1,y1=plotTerbanyak2010(df)
prov2,pred2,x2,y2=plotSedikit1971(df)
prov3,pred3,x3,y3=plotJmlIndo(df)
plotBestFitLine(x1,y1,x2,y2,x3,y3)
plt.title('Jumlah Penduduk INDONESIA (1972-2010)')
plt.legend()
plt.xlabel('Tahun')
plt.ylabel('jumlah penduduk(ratus juta jiwa)')
plt.grid(True)
plt.show()

print('Prediksi jumlah penduduk ',prov1,' di tahun 2050:', int(round(pred1[0])))
print('Prediksi jumlah penduduk ',prov2,' di tahun 2050:', int(round(pred2[0])))
print('Prediksi jumlah penduduk ',prov3,' di tahun 2050:', int(round(pred3[0])))