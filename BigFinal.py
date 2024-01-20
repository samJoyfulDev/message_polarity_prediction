from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
import matplotlib.pyplot  as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#ACTIVIDAD 1
#ubicar datos/cargar datos/importamos y revisamos contenidos
train=pd.read_csv("D:\Ingenieria de Sistemas\SISTEMA-2022-2\BigData\EF\Train.csv")
train.head();
#OBJETIVO: PREDECIR SI UNA NOTICIA, HA TRAIDO BUENAS O MALAS NOTICIAS
#corr: correlación hetmap: gráfico de matriz    abs: valor absoluto
#APLICANDO COMPARATIVA DE CORRELACIÓN
#utilizando mapa de calor
sns.heatmap(train.corr().abs(), annot=True)

#comparativa de calor 2
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
train.dataframeName = 'Train.csv'
plotCorrelationMatrix(train, 13)
#PODEMOS NOTAR EN EL GRÁFICO QUE LA PENDIENTES NEGATIVA ,
#VA DE ARRIBA HACIA ABAJO
#VAMOS A ASIGNAR CORRELACIONES ABSOLUTAS
corr= train.corr().abs()
#Según mi criterio voy a obtener la columna
#(ES UNA BUENA NOTICIA) 
#Y la relación con las demás variables
corr_IGN = corr.loc[:,['IsGoodNews']]
#SE VA A CORRELACIONAR LA COLUMNA ISGOODNEWS Y FREQ_OF_WORD_21 
#PORQUE ES EL QUE SE ASEMEJA MAS.

sns.boxplot(x='IsGoodNews', y='Freq_Of_Word_21', data=train)

#POSTERIORMENTE RELACIONAMOS LAS FILAS CON VALOR 0.3 HACIA ARRIBA DE ACUERDO A LA LISTA
train_select= train.loc[:,[
    "Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24","Freq_Of_Word_16",
    "Freq_Of_Word_23","Freq_Of_Word_11","LengthOFFirstParagraph"
    ]]
plt.figure(figsize = (20,10))
sns.heatmap(train_select.corr().abs(),annot=True)

#COMPARACIÓN CON LOS VALORES MAS ALTOS
#CREAMOS UNA GRAFICA COMPARATIVA
#CON LOS VALORES MAS ALTOS Y LA COLUMNA 
train_select2 = train.loc[:,["Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24"]]
sns.pairplot(train_select2)

#ACTIVIDAD 4
#GENERAMOS UN MODELO , IMPORTANDO LIBRERIA Y LOS COMPONENTES X , Y
from sklearn.model_selection import train_test_split

X = train.loc[:,["Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24"]]
Y = train.loc[:,["IsGoodNews"]]
#ESTABLECER EL GRUPO DE DATOS TRAIN Y TEST
#EL TAMAÑO DE ELEMENTOS PARA EL TEST ES DEL 30% CON TEST_SIZE
#EL TAMAÑO DE NÚMEROS RANDOM ES DE 33%
#LO RESTANTE PASA AL ENTRENAMIENTO
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=33)

# ACTIVIDAD 5
#============================
#METODO REGRESIÓN LINEAL
#AHORA CREAMOS UN MODELO LINEAL
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
#USAMOS LOS ELEMENTOS SELECCIONADOS PARA EL ENTRENAMIENTO
lm.fit(X_train, y_train)

#observamos si tenemos interpolado y mostramos coeficientes
print(lm.intercept_)

lm.coef_
print(str(lm.coef_))
#APLICAMOS VALORES DE TEST Y EMPEZAMOS HACER PREDICCIONES
predicciones = lm.predict(X_test)
print(predicciones)

#CONVERTIRMOS LA TABLA A UN DATA FRAME
DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
df_unido = y_test.join(DataFramePredicciones)
print(df_unido)

#GRAFICAMOS EL MODELO DE REGRESION LINEAL
plt.plot(X_test,predicciones, color="blue",marker = 'o')
plt.xlabel('IsGoodNews')
plt.ylabel('Promedio de noticias')
plt.scatter(train['IsGoodNews'],train['Freq_Of_Word_21'], color='pink')
plt.grid()

# ACTIVIDAD 6
#APLICAMOS MATRIZ DE CONFUSIÓN Y METRICAS

def metricas(clases_reales, clases_predichas):
    """ Calcular las métricas utilizando sklearn """
    matriz = confusion_matrix(clases_reales, clases_predichas)
    accuracy = accuracy_score(clases_reales, clases_predichas)
    precision = precision_score(clases_reales, clases_predichas)
    recall = recall_score(clases_reales, clases_predichas)
    f1 = f1_score(clases_reales, clases_predichas)
    return matriz, accuracy, precision, recall, f1

def visualiza_metricas(clases_reales, clases_predichas, titulo):
    """ Visualiza la matriz de confusión y métricas """
    
    #Código para calcular las métricas y matriz de confusión
    
    matriz, accuracy, precision, recall, f1 = \
                    metricas(clases_reales, clases_predichas)
    
    #Código de matplotlib para graficar 
    plt.figure(figsize=(3, 3))
    matriz = pd.DataFrame(matriz, 
                          columns=["0 : ISBADNEWS" , "1 : ISGOODNEWS"])
    plt.matshow(matriz, cmap="Blues", vmin=0, vmax=10, fignum=1)
    plt.title("Reales")
    plt.ylabel("Predichas")
    plt.xticks(range(len(matriz.columns)), matriz.columns, rotation=45)
    plt.yticks(range(len(matriz.columns)), matriz.columns)
    etiquetas = (("Verdaderos\nnegativos", "Falsos\npositivos"),
                 ("Falsos\nnegativos", "Verdaderos\npositivos"))
    for i in range(len(matriz.columns)):
        for j in range(len(matriz.columns)):
            plt.text(i, j + 0.14, str(matriz.iloc[i, j]),
                     fontsize=30, ha="center", va="center")
            plt.text(i, j - 0.25, etiquetas[i][j],
                     fontsize=11.5, ha="center", va="center")           
    plt.text(1.60, -0.30, titulo, fontsize=25, c="red")
    plt.text(2.1, 0.10, "Accuracy: %0.2f" % accuracy, fontsize=20)
    plt.text(2.1, 0.40, "Precision: %0.2f" % precision, fontsize=20)
    plt.text(2.1, 0.70, "Recall: %0.2f" % recall, fontsize=20)
    plt.text(2.1, 1.00, "F1: %0.2f" % f1, fontsize=20)    
    plt.show()
    print("\n" * 10)

print("\n" * 10)
# extraer las etiquetas de clase predichas
pred_y = np.where(predicciones > 0.5, 1,0)
visualiza_metricas(y_test, pred_y, "Total")

#LOS 3 TIPOS DE ERROR
#PRIMER ERROR = ERROR ABSOLUTO MEDIO (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, predicciones)
#SEGUNDO ERROR = ERROR CUADRADO MEDIO (MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predicciones)
#TERCER ERROR = RAIZ CUADRADA DEL ERROR CUADRATICO MEDIO (RMSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predicciones, squared=False)
#IMPRIMO UNA GRAFICA DE LA COLUMNA ISGOODNEWS
sns.displot(train.loc[:,['IsGoodNews']])


#AHORA REALIZARE EL PROCESO CON TEST

test = pd.read_csv("D:\Ingenieria de Sistemas\SISTEMA-2022-2\BigData\EF\Test.csv")
X = test.loc[:,["Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24"]]
predicE = lm.predict(X)
X.isna().sum()


#resultado matricial
#SE OBTIENE LAS 3 COLUMNAS Y UNA COLUMNA 0
predicciones = lm.predict(X)
DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
df_entrega = X.join(DataFramePredicciones)
print(df_entrega)

#===============
#SE PROCEDE A CONVERTIR LAS PREDICCIONES
predicciones = lm.predict(X)
DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
id = test.loc[:,['Freq_Of_Word_1']]
id.reset_index(drop = True, inplace = True)
df_entrega = id.join(DataFramePredicciones)
print(df_entrega)

df_entrega.columns = ['Freq_Of_Word_1','IsGoodNews']
df_entrega.to_csv('entrega.csv',index = False)
#GRAFICOS COMP
sns.displot(df_unido.loc[:,['IsGoodNews']], color="skyblue", label="X", kde=True)
sns.displot(df_entrega.loc[:,['IsGoodNews']], color="red", label="X", kde=True)
#===============


#===============================

#*****************************************************************
#METODO REGRESION LOGISTICA
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
prediceLog = logmodel.predict(X_test)
print(prediceLog)

#ANALIZAMOS LAS METRICAS 
from sklearn.metrics import classification_report
print(classification_report(y_test, prediceLog))

#mostramos tabla de confusión
confusion_matrix(y_test, prediceLog)

#se implementara la curva roc para saber si es una buena o mala noticia
#en base a las predicciones
from sklearn.metrics import roc_curve
y_pred_prob = logmodel.predict_proba(X_test)[:,1]

#SE DEBE ESCOGER una de las columnas
#porque los valores son 0 y 1
#0 es una columna y 1 otra columna
roc_curve(y_test, y_pred_prob)

#obtener falso positivo
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
print(fpr)
print(tpr)
print(threshold)

#se dibuja

plt.plot(fpr, tpr, color='red', label = 'Curva ROC')
plt.plot([0,1], color = 'blue', linestyle = '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

#LOS 3 TIPOS DE ERROR
#PRIMER ERROR = ERROR ABSOLUTO MEDIO (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, prediceLog)
#SEGUNDO ERROR = ERROR CUADRADO MEDIO (MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, prediceLog)
#TERCER ERROR = RAIZ CUADRADA DEL ERROR CUADRATICO MEDIO (RMSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, prediceLog, squared=False)
#*******************************************************************
#===================================================================
#METODO KNN VECINOS
#IMPORTAMOS LA LIBRERIA DE ALGORITMOS 
#SE VA UTILIZAR EL CRITERIO DE 1 SOLO VECINO
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#predicciones
pred = knn.predict(X_test)
print(pred)

#generamos el reporte de métricas para analizar
#resultado

report = classification_report(y_test, pred)
tabla = confusion_matrix(y_test, pred)
print(report)
print(tabla)

#calculamos la puntuación
knn.score(X_test, y_test)
#calculamos el scorea a valores de entrenamiento
knn.score(X_train, y_train)

#establecer un número de vecinos de 30
vecinos =np.arange(1,30)
print(vecinos)

#crear 2 matrices vacias 
train_2 =np.empty(len(vecinos))
test_2 =np.empty(len(vecinos))
print(train_2)
print(test_2)
#generamos bucle para registrar datos en las matrices 
 #se genera un bubcle para registrar
for i, k in enumerate(vecinos):
     km = KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train, y_train)
     test_2[i] = knn.score(X_test, y_test)
     train_2[i] = knn.score(X_train, y_train)
    
print(train_2)
print (test_2)

#crear gráfico

plt.title('NUMERO DE VECINOS PROXIMOS KINN')
plt.plot(vecinos, test_2, label='Exactitud de Test')
plt.plot(vecinos, train_2, label='Exactitud de Train')
plt.legend()
plt.xlabel('Número de vecinos')
plt.ylabel('con exactitid')
plt.show()

#LOS 3 TIPOS DE ERROR
#PRIMER ERROR = ERROR ABSOLUTO MEDIO (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred)
#SEGUNDO ERROR = ERROR CUADRADO MEDIO (MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)
#TERCER ERROR = RAIZ CUADRADA DEL ERROR CUADRATICO MEDIO (RMSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred, squared=False)
#===================================================================
#**********************************************************************
#ARBOLES DE DECISIÓN
#ENTRENAMOS
from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier()
arbol.fit(X_train, y_train)
#importamos
#MOSTRAMOS EL ARBOL
from sklearn import tree
tree.plot_tree(arbol)

#PROCEDO A SELECCIONAR LAS COLUMNAS A MOSTRAR
X_nombre = list(X.columns)
classes = ["Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24"]
fig, axes = plt.subplots(nrows = 1, ncols=1,figsize = (3,3), dpi= 300)
tree.plot_tree(arbol,feature_names= X_nombre, class_names =classes, filled=True)
fig.savefig('imagen.png')
#CREAMOS LAS PREDICCIONES
predArbolesDecision = arbol.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predArbolesDecision))
print(classification_report(y_test, predArbolesDecision))

#LOS 3 TIPOS DE ERROR
#PRIMER ERROR = ERROR ABSOLUTO MEDIO (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, predArbolesDecision)
#SEGUNDO ERROR = ERROR CUADRADO MEDIO (MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predArbolesDecision)
#TERCER ERROR = RAIZ CUADRADA DEL ERROR CUADRATICO MEDIO (RMSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predArbolesDecision, squared=False)
#**********************************************************************

#=====================================================================
# ÁRBOLES ALEATORIOS
#El número 100 es número de bosques aleatorios
#*
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
rfc = RandomForestClassifier(n_estimators=20 ,random_state=33)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))

#PROCEDO A SELECCIONAR LAS COLUMNAS A MOSTRAR
X_nombre = list(X.columns)
classes = ["Freq_Of_Word_21","Freq_Of_Word_7","Freq_Of_Word_24"]
fig, axes = plt.subplots(nrows = 1, ncols=1,figsize = (3,3), dpi= 300)
tree.plot_tree(rfc,feature_names= X_nombre, class_names =classes, filled=True)
fig.savefig('Bosque.png')

#grafico

plt.plot(X_test,rfc_pred, color="blue",marker = 'o')
plt.xlabel('IsGoodNews')
plt.ylabel('Promedio de noticias')
plt.scatter(train['IsGoodNews'],train['Freq_Of_Word_21'], color='pink')
plt.grid()

#LOS 3 TIPOS DE ERROR
#PRIMER ERROR = ERROR ABSOLUTO MEDIO (MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, rfc_pred)
#SEGUNDO ERROR = ERROR CUADRADO MEDIO (MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, rfc_pred)
#TERCER ERROR = RAIZ CUADRADA DEL ERROR CUADRATICO MEDIO (RMSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, rfc_pred, squared=False)
#=====================================================================