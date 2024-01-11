import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

rcp = pd.read_csv('rcp-2017-08.csv',
                  encoding='latin-1', sep=';')

rcp = rcp.drop(rcp.tail(2).index)

coluna_array = rcp[' VALOR MENSAL '].values
rcp_mediana = np.median(coluna_array)
print(rcp_mediana)
print('\n\n\n\n ------------- \n\n\n')

rcp.mean(axis=1)

colunas = ['SERVIÇO', 'CAPAC.', ' VALOR MENSAL ']
dados = rcp[colunas]

dados[' VALOR MENSAL '] = pd.to_numeric(
    dados[' VALOR MENSAL '], errors='coerce')
dados['CAPAC.'] = pd.to_numeric(dados['CAPAC.'], errors='coerce')

dados = dados.dropna()

X = dados[['CAPAC.', ' VALOR MENSAL ']]
y = dados['SERVIÇO']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

coeficientes = pd.DataFrame(
    {'Variável': X.columns, 'Coeficiente': modelo.coef_})
print(coeficientes)

intercepto = modelo.intercept_
print('Intercepto:', intercepto)

y_pred = modelo.predict(X_test)

score = modelo.score(X_test, y_test)
print('R²:', score)
