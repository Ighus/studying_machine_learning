#%%
import pandas as pd

#%%

base = pd.read_csv('bases/risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
#%%
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

#%%
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

#%%
resultado = classificador.predict([])