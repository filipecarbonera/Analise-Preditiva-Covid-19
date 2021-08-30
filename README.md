![](https://github.com/filipecarbonera/analise_preditiva_covid_19/blob/main/IMAGENS/Hospital-S%C3%ADrio-Liban%C3%AAs.jpeg)
**COVID-19 ANÁLISE PREDITIVA DA NECESSIDADE DE LEITOS DE UTI POR PACIENTES ADMITIDOS**

Por Filipe Carbonera de Souza

Neste projeto apresentarei meu resultado do desafio final do BootCamp de DataScience Aplicada II da Alura, ao qual iniou-se em maio 2021 e finalizar-se-á com a entrega deste relatório. Neste repositório há uma pasta "DADOS" contendo os dados em excel disponibilizados no Kaggle do Hospital Sírio-Libanês, uma pasta IMAGENS, contendo as imagens utilizadas do repositório e uma pasta NOTEBOOK contendo o notebook com todo o código necessário para a realização deste relatório, além dos arquivos README.md e LICENCE.

**CONTEXTO E OBJETIVO:**

PANDEMIA DO COVID-19

Com a evolução da pandemia do COVI-19, surgiu a preocupação relativa a capacidade do sistema de saúde quanto ao número de leitos de UTI disponíveis em relação ao número de pessoas com a doença. A situação que se apresentou com o tempo foi de que, considerando a taxa de contágio do vírus e o número pacientes que necessitam de UTI, rapidamente o número de leitos não iria comportar a necessidade vigente para os mesmos.

![graf](https://user-images.githubusercontent.com/77364648/129631176-f184532e-0e4f-430e-809f-6ceda1523f4e.gif)

Com isso o Hospital Sírio-Libanês propôes através da plataforma Kaggle o desafio de construirmos um modelo de machine learning para prever se um paciente que é admitido no hospital com Covid precisará de um leito de UTI durante sua permanência no hospital ou não.

Nas palavras do próprio hospital, "Quando conseguimos definir a quantidade de leitos necessários em um determinado hospital,conseguimos evitar rupturas, visto que, caso outra pessoa procure ajuda e, eventualmente, precise de cuidados intensivos, o modelo preditivo já conseguirá detectar essa necessidade e, desta forma, a remoção e transferência deste(a) paciente pode ser organizada antecipadamente."

**OS DADOS:**

Conforme apresentado anteriormente, utilizarei a base de dados disponibilizada pelo Hospital Sírio Libanês no Kaggle. Abaixo irei esclarecer alguns pontos e contextualizar os dados.

O dataset disponibilizado tem 1925 linhas e 231 colunas. A coluna WINDOW é a "janela" de tempo em que os dados do paciente foram coletados (exames e medições). A coluna ICU é a variável alvo do estudo, que diz se o paciente foi internado(1) ou não(0) na UTI naquela janela de tempo.

Ou seja, cada registro(linha) do dataframe não se trata de um paciente, e sim que registros de um paciente em uma janela de tempo. Em cada um desses registros também é possível verificar se naquela janela de tempo o paciente foi ou não internado na UTI.

As demais colunas são apresentadas abaixo:

**COLUNAS DO DATASET**

**Informações demográficas**
- AGE_ABOVE65
- AGE_PERCENTIL
- GENDER

**Doenças pré-existentes**
- DISEASE GROUPING 1	
- DISEASE GROUPING 2	
- DISEASE GROUPING 3	
- DISEASE GROUPING 4	
- DISEASE GROUPING 5	
- DISEASE GROUPING 6	

**Resultados dos exames de sangue**

Para cada métrica do exame de sangue há uma coluna com as seguintes informações:
- MEDIAN (médiana)
- MEAN (média)
- MIN (mínimo)
- MAX (máximo)
- DIFF (diferencial)

Seguem:
- ALBUMIN
- BE_ARTERIAL
- BE_VENOUS
- BIC_ARTERIAL
- BIC_VENOUS
- BILLIRUBIN
- BLAST
- CALCIUM
- CREATININ
- FFA
- GGT
- GLUCOSE
- HEMATOCRITE
- HEMOGLOBIN
- INR
- LACTATE
- LEUKOCYTES
- LINFOCITOS
- NEUTROPHILES
- P02_ARTERIAL
- P02_VENOUS
- PC02_ARTERIAL
- PC02_VENOUS
- PCR
- PH_ARTERIAL
- PH_VENOUS
- PLATELETS
- POTASSIUM
- SAT02_ARTERIAL
- SAT02_VENOUS
- SODIUM
- TGO
- TGP
- TTPA
- UREA
- DIMER

**Sinais vitais**

- BLOODPRESSURE_DIASTOLIC
- BLOODPRESSURE_SISTOLIC
- HEART_RATE
- RESPIRATORY_RATE
- TEMPERATURE
- OXYGEN_SATURATION

É importante salientar que esses dados foram anonimizados seguindo as melhores práticas, além de estar de acordo com a [LGPD](https://www.serpro.gov.br/lgpd/menu/protecao-de-dados/dados-anonimizados-lgpd) (Lei Geral de Proteção de Dados).

---

**VARIÁVEIS PRESENTES NO NOTEBOOK**
- seed: Valor fixo e aleatório para garantir a reprodutibilidade do estudo.
- dados: Dataset original, como disponibilizado pelo Hosítal Sirio Libanes.
- dados_finais: Dataset em meio ao processo de limpeza dos dados.
- dados_limpos: Dataset pronto para rodar os modelos.
- x: Conjunto de dados de entrada (todas as colunas do dataframe com excessão da ICU).
- y: Conjunto de dados de saída (apenas a coluna ICU, variável objetivo do estudo).
- modelo_dummy: Variável que contém o modelo Dummy.
- modelo_lr: Variável que contém o modelo LogisticRegression.
- modelo_dtc: Variável que contém o modelo DecisionTreeClassifier.
- modelo_rfc: Variável que contém o modelo RandomForestClassifier.
- modelo_knc: Variável que contém o modelo KNeighborsClassifier.

---

**ESCOPO:**

O objetivo deste relatório é apresentar uma forma viável de prever se os pacientes que dão entrada no hospital irão precisar de leitos de UTI. Para tanto serão aplicadas técnicas de Machine Learning para explorar, modelar e prever dados a fim de encontrar o modelo que apresente os melhores resultados neste contexto.

Dentre os [diversos médotos de aprendizagem em Machine Learning](https://www.startse.com/noticia/nova-economia/machine-learning-inteligencia-artificial-aprendizado), irei focar meus esforços no Aprendizado Supervisionado, pois o objetivo aqui é prever a necessidade ou não de leitos de UTI para pacientes que derem entrada no hospital, o que aproxíma-se de métodos classificação, que serão analisados mais a fundo ao longo deste relatório.

**Etapas do Estudo:**

**1- Análise do Dataframe:**

Reservei o capítulo Dados para contextualizar o dataset utilizado, além disso, demais informações podem ser consultadas diretamente no [repositório do Sírio-Libanês no Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19).

**2- Ajustando os dados para rodar os modelos:**

Função: *preenche_tabela(dados)*
Criei e utilizei essa função para preencher alguns dados faltantes através dos métodos back e front fill. A utilização desses métodos foi motivada pelo fato de neste database não haver muitos dados faltantes em sequência, além de, naturalmente, no intervalo de duas horas das janelas, a tendência é que quanto maior a frequência de medição, menores são os desvios incrementais nos dados. Com isso, o dataset fica apenas com um paciente com dados incompletos, o que contabiliza menos 5 linhas em relação aos dados iniciais.

Função: *selecionando_pacientes(rows)*

Após o preenchimento dos dados faltantes, me concentrei em filtrar apenas os dados úteis à análise dentro do contexto do objetivo proposto. Idealmente, teriamos que prever a necessidade do leito de UTI pelo paciente tão logo ele desse entrada no hospital, então, utilizarei para análise apenas os registros de pacientes que não  foram para a UTI nas primeiras duas horas após a entrada no hospital, pois não temos dados anteriores a isso para prever essa entrada. Outro ponto importante é que agrupei os registros por paciente (cada linha sendo um paciente) e filtrei apenas os que de fato foram para a UTI em alguma janela de tempo posterior a duas horas.

Além disso, tratei a coluna AGE_PERCENTIL do dataset para que se tornasse numérica e pudesse ser utilizada nos modelos. O modelo One-Hot Encoder foi escolhido pois trabalha apenas com zeros e uns, ficando de acordo com o restante do dataset.

Por fim, informo que as devidas explicações relativas ao tratamento dos dados será realizada juntamente aos respectivos códigos no notebook desse repositório.

**3- Função para rodar modelos e modelo leigo (baseline).**

Primeiramente é utilizado o [Dummy](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) para usar como balisador/valor de corte para os demais modelos, ou seja, qualquer modelo que tenha resultado pior não será considerado.

Realizada a compreensão dos dados e os ajustes do dataset, criei a função *rodar_modelos(modelo, dados, n):* para facilitar a aplicações dos modelos e a visualização das métricas. Para cada modelo testado, o mesmo foi primeiramente instanciado e dentro da função são feitos os seguintes passos:

a) Separação das variáveis de entrada e saída n vezes;

b) Estratificação dos dados n vezes; (método: [RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) )


c) Calculo do [AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) médio e seu intervalo de auc ([artigo sobre a AUC-ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5));

O metodo de validação utilizado foi o [Cross Validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).

**4- Testes de modelos de Machine Learning e avaliando métricas.**

LogisticRegression ([documentação](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)).

DecisionTreeClassifier ([documentação](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)).

KNeighborsClassifier ([documentação](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors)).

RandomForestClassifier ([documentação](https://scikit-learn.org/0.15/modules/generated/sklearn.ensemble.RandomForestClassifier.html)).

Tais metodos foram escolhidos devido sua relevancia e popularidade entre os metodos de classificação existentes. 

**CONCLUSÃO**

Dentre eles, o que apresentou melhor resultado foi o **RandomForestClassifier**, pois o ROC AUC score foi o maior, ou seja, é o modelo que melhor consegue prever as pessoas que irão e não irão para a UTI.

Com este estudo foi possível permear todos os tópicos apresentados no Bootcamp de Data Science Aplicada da Alura, mesmo que nem todos tenham sido aplicados diretamente a este estudo. Para próximos estudos em Data Science e Machine Learning irei me aprofundar mais em cada um dos tópicos abordados e em especial a escolha e definição de hiperparametros para os modelos, além de dedicar mais tempo a plotagem de gráficos para melhoria do storytelling.

**AGRADECIMENTOS**

Agradeço imençamente a toda a equipe da Alura que fez parte dessa trajetória de mais de quatro meses. Os aprendizados, resumidos neste relatório, são de grande importancia e significado para mim e minha carreira.

CONTATOS

[Fique a vontade para me contatar](https://linktr.ee/filipecarbonera)!
