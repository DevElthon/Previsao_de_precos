import pandas as pd
import plotly
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

#leitura e visualizaçao da tabela
tabela = pd.read_csv("barcos_ref.csv")
print(tabela)
print(tabela.info())

#Correlação das linhas com uma única coluna (preço)
correlacao = tabela.corr()[["Preco"]]
print(correlacao)
sns.heatmap(correlacao, cmap="Blues", annot=True)
plt.show()
#Após visualização e análise, fechar imagem

#Treinamento de dois modelos (regressão linear e árvore de decisão)
y = tabela["Preco"]
x = tabela.drop("Preco", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))
#Modelo de arvore de decisão teve um melhor desempenho para previsão de preços


#Visualização para comparar os dois modelos
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["RegressaoLinear"] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()
#Após visualização e análise, fechar imagem

#Previsão dos preços dos barcos presentes na tabela de novos barcos
tabela_nova = pd.read_csv("novos_barcos.csv")
print(tabela_nova)

previsao = modelo_arvoredecisao.predict(tabela_nova)
print(f"Preco: {previsao}")