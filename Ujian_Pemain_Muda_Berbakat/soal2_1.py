import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


df=pd.read_csv('data.csv')
df=df.fillna(np.NaN)
#membuat plot antara Age vs Overall
x=df['Age']
y=df['Overall']
y2=df['Potential']
xAge=df['Age'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
yOverall=df['Overall'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
plt.figure()
plt.subplot(121)
plt.scatter(x,y,label='Non-Target',color='r')
plt.scatter(xAge,yOverall,label='Target',color='g')
plt.title('Age vs Overall')
plt.xlabel('Age')
plt.ylabel('Overall')
plt.legend()
plt.grid(True)


yPotential = df['Potential'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]
plt.subplot(122)
plt.scatter(x,y2,label='Non-Target',color='r')
plt.scatter(xAge,yPotential,label='Target',color='g')
plt.title('Age vs Potential')
plt.xlabel('Age')
plt.ylabel('Potential')
plt.legend()
plt.grid(True)
plt.show()