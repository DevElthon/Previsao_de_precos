# Previsao_de_precos
Hoje em dia, vários a maior parte dos mercados trabalham com previsão de preços com base em uma série de características do mercado, para isso, são utilizados vários modelos de machine learning para realizar o processo de forma rápida. Pensando nisso, criei esse projeto que utiliza sklearn e alguns modelos de previsão em cima de uma base da dados contendo preços e características de barcos.

Para o desenvolvimento desse projeto, inicialmente foi necessário o tratamento dos dados da tabela para correlacionar as características de cada barco com o preço que é o objetivo final. Após isso, utilizando a biblioteca "Sklearn", foram testados dois modelos de aprendizado, regressão linear e árvore de decisão, onde após realizar os treinamentos e testes, a árvore de decisão foi o melhor modelo alcançando entre 84% e 86% de precisão.

Por fim, utilizando árvore de decisão, foi realizada uma previsão dos preços de alguns barcos presentes na tabela de novos barcos (uma tabela contendo os mesmos tipos de características que a primeira, com excessão dos preços).
