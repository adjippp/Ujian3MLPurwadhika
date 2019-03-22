import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


df=pd.read_csv('data.csv')
# df=df.fillna(np.NaN)
#membuat plot antara Age vs Overall
xAge=df['Age'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
yOverall=df['Overall'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
yPotential = df['Potential'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
indexX=xAge.index.tolist()
xout=df['Age'].loc[~df.index.isin(indexX)]
yOvout=df['Overall'].loc[~df.index.isin(indexX)]
yPotout=df['Potential'].loc[~df.index.isin(indexX)]
plt.figure()
plt.subplot(121)
plt.scatter(xAge,yOverall,label='Target',color='g')
plt.scatter(xout,yOvout,label='Non-Target',color='r')
plt.title('Age vs Overall')
plt.xlabel('Age')
plt.ylabel('Overall')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.scatter(xAge,yPotential,label='Target',color='g')
plt.scatter(xout,yPotout,label='Non-Target',color='r')
plt.title('Age vs Potential')
plt.xlabel('Age')
plt.ylabel('Potential')
plt.legend()
plt.grid(True)
plt.show()