#%%
import pandas as pd

base = pd.read_csv('Decision Trees/bases/risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

labelenconder = LabelEncoder()
previsores[:,0] = labelenconder.fit_transform(previsores[:,0])
previsores[:,1] = labelenconder.fit_transform(previsores[:,1])
previsores[:,2] = labelenconder.fit_transform(previsores[:,2])
previsores[:,3] = labelenconder.fit_transform(previsores[:,3])

from sklearn.tree import DecisionTreeClassifier, export

classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
print(classificador.feature_importances_)
# resultado = classificador.predict
export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garnatias', 
                       'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)
#%%
