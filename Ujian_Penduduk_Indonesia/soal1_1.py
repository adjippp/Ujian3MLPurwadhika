import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df=pd.read_excel('indo_12_1.xls')
df.columns=['provinsi','1971','1980','1990','1995','2000','2010']
df=df.dropna()
df=df.replace('-',np.NaN)
# df=df.fillna(int(0)).reset_index().drop(columns='index')

plt.style.use('ggplot')
#plot jumlah penduduk Indonesia
def plotJmlIndo(df):
    Indonesia=df['provinsi'].tail(1).values[0]
    df=df.set_index('provinsi').T
    x=df.index.values.astype(int)
    y=df[Indonesia].values.astype(int)
    plt.plot(x,y,linestyle='-',color='r',label=Indonesia)
    plt.scatter(x,y,color='r')
    return Indonesia


#plot jumlah penduduk dari provinsi yang memiliki penduduk terbanyak di tahun 2010
def plotTerbanyak2010(df):
    df=df.drop(df.tail(1).index)
    kotaTerbanyak2010=df['provinsi'][df['2010']==df['2010'].max()].values[0]
    df=df.set_index('provinsi').T
    x=df.index.values.astype(int)
    y=df[kotaTerbanyak2010].values.astype(int)
    plt.plot(x,y,linestyle='-',color='g',label=kotaTerbanyak2010)
    plt.scatter(x,y,color='g')
    return kotaTerbanyak2010

# jumlah penduduk dari provinsi yang memiliki penduduk paling sedikit di tahun 1971
def plotSedikit1971(df):
    df=df.drop(df.tail(1).index)
    kotaSedikit1971=df['provinsi'][df['1971']==df['1971'].min()].values[0]
    df=df.set_index('provinsi').T
    x=df.index.values.astype(int)
    y=df[kotaSedikit1971].values.astype(int)
    plt.plot(x,y,linestyle='-',color='b',label=kotaSedikit1971)
    plt.scatter(x,y,color='b')
    return kotaSedikit1971

plotTerbanyak2010(df)
plotSedikit1971(df)
plotJmlIndo(df)
plt.title('Jumlah Penduduk INDONESIA (1972-2010)')
plt.legend()
plt.xlabel('Tahun')
plt.ylabel('jumlah penduduk(ratus juta jiwa)')
plt.grid(True)
plt.show()