#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2024-25
# PARTE I del trabajo práctico: Implementación de árboles de decisión 
#                               y random forests
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Campos Naranjo
# NOMBRE: Adrián
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Fornés Jiménez
# NOMBRE: José
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La
# discusión y el intercambio de información de carácter general con
# los compañeros se permite, pero NO AL NIVEL DE CÓDIGO. Igualmente el
# remitir código de terceros, OBTENIDO A TRAVÉS DE LA RED, o de
# cualquier otro medio, se considerará plagio.

# El objetivo principal del trabajo es reforzar de manera práctica
# los conceptos aprendidos en clase, para alcanzar una mayor
# comprensión de los mismos a través de la implementación que se
# pide. Se permite, si así se desea, el uso de herramientas de
# inteligencia artificial generativa que asistan en el desarrollo
# código, pero esta herramienta ha de usarse sólo como un asistente
# que facilite el trabajo, y en ningún caso se debe entregar un código
# que no se conozca en profundidad y con detalle. 

# Para asegurar que la evaluación del mismo está alineada con el
# objetivo descrito en el párrafo anterior, el trabajo ha de ser
# presentado ante el profesor, explicando con detalle y a nivel de
# código la implementación entregada, y será necesario demostrar total
# comprensión del código entregado. Si el trabajo se hace en grupo,
# ambos miembros del grupo deben poder explicar con detalle de código
# cualquier parte del trabajo.

# Cualquier plagio o entrega de código cuyo funcionamiento no se sea
# capaz de explicar con detalle, significará automáticamente la
# calificación de CERO EN LA ASIGNATURA para TODOS los estudiantes
# involucrados. Independientemente de OTRAS ACCIONES DE CARÁCTER
# DISCIPLINARIO que se pudieran tomar.
# *****************************************************************************************


# MUY IMPORTANTE: 
# ===============    
    
# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. ADEMÁS: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn, salvo donde se dice expresamente.
#   En particular, si se pide implementar algo, se refiere a implementar en python,
#   sin usar Scikit Learn.  
  
# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos 
#   los datasets se suponen dados como arrays de numpy. 

# * Este archivo (con las implementaciones realizadas), ES LO ÚNICO QUE HAY QUE ENTREGAR.

# * AL FINAL DE ESTE ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final a entregar, descomentar esos ejemplos del final y no dejar 
#   ninguna otra ejecución de ejemplos. 



import math
import random
import numpy as np
np.random.seed(42) # Para que los resultados sean reproducibles
random.seed(42)


# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-aia.zip (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN:
    
from carga_datos import *    

# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre pasajeros del Titanic y si sobrevivieron o no. Es una versión 
#   restringida de este conocido dataset, con solo tres caracteristicas:
#   Pclass, IsFemale y Age. Se carga en las variables X_train_titanic, 
#   y_train_titanic, X_test_titanic e y_test_titanic.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos (ver 
#   descripción en votos.py)


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer. 
#   Ver descripcición en sikit learn.

  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    


#  Además, en la carpeta datos/ se tienen los siguientes datasets, que
#  habrán de ser procesado y cargado (es decir, no se caragan directamente con
#  carga_datos.py).   
    
# * Un archivo credito.csv con datos sobre concesión de prestamos en una entidad 
#   bancaria, en función de: tipo de empleo, si ya tiene productos finacieros 
#   contratados, número de propiedades, número de hijos, estado civil y nivel de 
#   ingresos (cargarlo usando pd.read_csv en arrays de numpy X_credito e y_credito,
#   donde X_credito son las seis primeras columnas e y_credito la última).


# * Un archivo adultDataset.csv, con datos de personas para poder predecir si
#   alguien gana más o menos de 50000 dólares anuales, en función de una serie 
#   de características (para más detalles, ver https://archive.ics.uci.edu/dataset/2/adult)  
#   Más adelante se explica cómo cargar y procesar este conjunto de datos. 

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos en archivos de texto. 
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

#  >>>Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],Ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#   (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
#  (array([0, 1]), array([136, 229]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([34, 57]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------

def particion_entr_prueba(X,y,test=0.20):

    # Vamos a asegurarnos de que X e Y son arrays de numpy para trabajar con ellos fácilmente
    X=np.array(X)
    y=np.array(y)

    # Para mantener la correspondencia entre X e y vamos a trabajar con los indices en vez de con los datos directamente
    indices=np.arange(len(y))
    # Sacamos las clases de y haciendo uso de np.unique que recorrerá el array entero y se quedará con una lista con los valores distintos que aparecen en y
    clases=np.unique(y)

    indices_entrenamiento=[]
    indices_prueba=[]

    for clase in clases:
        indices_clase = indices[y==clase] # Base para la estratificación
        np.random.shuffle(indices_clase) # Conseguimos la aleatoriedad
        num_prueba = int(len(indices_clase) * test) # Proporción que se quiere para prueba
        indices_prueba.extend(indices_clase[:num_prueba]) # Los añadimos a indices_prueba
        indices_entrenamiento.extend(indices_clase[num_prueba:]) # El resto a indices_entrenamiento
        # Estratificación conseguida, cada clase aparecerá en la proporción deseada tanto en entrenamiento como en prueba

    # Queda conseguir los conjuntos de entrenamiento y test respectivamente con el indexado de los indices anteriores
    X_train = X[indices_entrenamiento]
    y_train = y[indices_entrenamiento]
    X_test  = X[indices_prueba]
    y_test  = y[indices_prueba]
     
    return X_train, X_test, y_train, y_test
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))


# ===============================================
# EJERCICIO 2: IMPLEMENTACIÓN ÁRBOLES DE DECISIÓN
# ===============================================


# En este ejercicio pedimos implementar en python un algoritmo de aprendizaje para árboles 
# de decisión. Los árboles de decisión que trataremos serán árboles binarios, en los que
# en cada nodo interior se pregunta por el valor de un atributo o característica dada, 
# y si ese valor es mayor o menor que un valor umbral dado. Este es el mismo tipo de árbol 
# de decisión que se  manejan en Scikit Learn. 

# Se puede obtener información de este tipo de árboles en la entrada "Decision Trees"
# del manual de Scikit Learn. También en la práctica del Titanic hecha en clase.

# Se propone la implementación de un clasificador basado en árboles de
# de decisión, entrenado usando el algoritmo CART, similar al que implementa 
# la clase DecisonTree de Scikit Learn, pero con ALGUNAS VARIANTES, que indicaremos más
# adelante.

# Los árboles de decisión están formados por nodos. Usar la siguiente clase para la
# implementación de los nodos:
    
class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None,distr=None,*,clase=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izq = izq
        self.der = der
        self.distr= distr
        self.clase = clase
        
    def es_hoja(self):
        return self.clase is not None

# Pasamos a describir los distintos atributos de esta clase:

# - atributo: el atributo por el que se pregunta en el nodo. Referenciaremos a cada
#   atributo POR EL ÍNDICE DE SU POSICIÓN (el número de columna).
# - umbral: es el valor umbral por el que se pregunta en el nodo. Si la instancia tiene un
#   valor de atributo menor o igual que el umbral, se sigue por el subárbol izquierdo. En
#   caso contrario, por el subárbol derecho.
# - izq: es el nodo raiz del subárbol izquierdo.
# - der: el nodo raiz del subárbol derecho.
# - distr: es un diccionario cuyas claves son las posibles clases, y cuyos valores son
#   cuántos ejemplos del conjunto de entrenamiento correspondientes al nodo hay de cada
#   clase. Cuando decimos "ejemplos correspondientes al nodo" queremos decir aquellos que
#   cumplen todas las condiciones (desde la raiz) que llevan a ese nodo.
# - clase: Si el nodo es una hoja, es la clase que predice. Si no es una hoja, este valor es None.



# Lo que sigue es una descripción del algoritmo que se pide implementar para la
# construcción de un árbol de decisión. En principio describiremos la versión básica y más
# conocida, y posteriormente indicaremos las peculiaridades y variantes que pedimos
# introducir a esta versión básica.

# Supondremos que recibimos un conjunto de entrenamiento X,y y además dos valores max_prof
# y min_ejemplos_nodo_interior, que nos van a servir como condiciones adicionales para
# dejar de expandir un nodo. El algoritmo se define recursivamente y tiene además un
# argumento adicional prof (inicialmente 0), con la profundidad del nodo actual.  

# CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0):

# 1. SI prof es mayor o igual que max_prof, 
#       o el número de ejemplos de X es menor que min_ejemplos_nodo_interior,
#       o en X todos los ejemplos son de la misma clase:
#       ENTONCES:
#          Devolver un nodo hoja con la distribución de clases en X,
#                    y con la clase mayoritaria en X
# 2. EN OTRO CASO:
#        encontrar el MEJOR atributo A y el mejor umbral u para ese atributo
#        y particionar en dos tanto X como y:
#            * X_izq, y_izq los ejemplos cuyo valor de A es menor o igual que u
#            * X_der, y_der los ejemplos cuyo valor de A es mayor que u
#        Llamadas recursivas:
#            A_izq=CONSTRUYE_ARBOL(X_izq,y_izq,min_ejemplos_nodo_interior,max_prof,prof+1)
#            A_der=CONSTRUYE_ARBOL(X_der,y_der,min_ejemplos_nodo_interior,max_prof,prof+1)
#        Devolver un nodo interior con el atributo y umbral seleccionado,
#                 con la distribución de clases de X, y con A_izq y A_der
#                 como hijos izquierdo y derecho respectivamente.


# Lo anterior es la descripción básica. A continuación indicamos una serie de variantes y
# cuestiones adicionales que se le piden a esta implementación concreta:

# - Consideraremos la posibilidad de restringir los atributos a usar en el árbol a un
#   número de atributos dado n_atrs. Ese subconjunto de atributos se seleccionará
#   aleatoriamente al principio de la construcción del aŕbol y será el mismo para todos
#   los nodos.
#   Por ejemplo, si el dataset tiene 15 atributos y le damos n_atrs=9, al comienzo de la
#   construcción del árbol seleccionamos aleatoriamente 9 atributos, y ya en los nodos del
#   árbol solo podrán aparecer alguno de esos 9 atributos. Nótese que si n_atrs es igual
#   al total de atributos, tendríamos la versión estándar del algoritmo.
#   NOTA: téngase en cuenta que a diferencia de lo que ocurre en la versión clásica de
#   Random Forests, no sorteamos los atributos en cada nodo, sino que hay un único sorteo
#   inicial para todo el árbol.

# - A la hora de elegir el mejor atributo y umbral para la partición de los nodos
#   interiores, usar el criterio de mejor GANANCIA DE INFORMACIÓN (en particular, NO USAR GINI).

# - La principal carga computacional de este algoritmo se debe a la cantidad de candidatos a
#   mejor atributo y mejor umbral que hay que evaluar en cada nodo, para decidir cuál es
#   la mejor partición. El hecho de limitar el número de atributos candidatos (como se ha
#   descrito más arriba), va en esa dirección. 
#   Otra manera es limitar también los posibles valores umbrales a considerar
#   para cada atributo. Para ello, en la implementación que se pide actuaremos en dos
#   sentidos:
#      (a) Considerar solo como candidatos a umbral los puntos medios entre cada par de 
#         valores consecutivos del atributo en los que hay cambio de clase, para los
#         ejemplos correspondientes a ese nodo.
#         Por ejemplo, si ordenados los valores del atributo A en orden creciente, hay un
#         ejemplo con valor v1 de A y clase C1 y a continuación otro ejemplo con valor v2
#         en A y clase C2 distinta de C1, entonces (v1+v2)/2 es un posible valor umbral
#         candidato. El resto de valores NO se considera candidato.

#      (b) En cada nodo, para elegir los umbrales candidatos correspondientes a un atibuto,
#         no considerar todos los ejemplos que corresponden a ese nodo, sino 
#         sólo  una proporción de los mismos, seleccionada aleatoriamente. La proporción a
#         considerar se da en un parámetro prop_umbral.
#         Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos correspondientes al
#         nodo es de 200 ejemplos, entonces aplicaremos el proceso de selección de
#         umbrales candidatos descrito en (a) considerando sólo un suconjunto de 140
#         ejemplos seleccionado aleatoriamente de entre esos 200.  



# Con las descripciones anteriores, ya podemos precisar lo que se pide en eset apartado. 
# Se pide implementar una clase ArbolDecision con el siguiente formato:
def ganancia_informacion(y, izquierda, derecha):
    def entropia(y):
        # Calcula la entropía de un vector de clases y
        # clases: array con las clases únicas
        # conteo: número de ejemplos de cada clase
        clases, conteo = np.unique(y, return_counts=True)
        probs = conteo / conteo.sum()
        # Añadimos un pequeño valor para evitar log(0)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    entropia_total = entropia(y)
    # Calculamos la entropía de los subconjuntos izquierdo y derecho
    entropia_izquierda = entropia(y[izquierda])
    entropia_derecha = entropia(y[derecha])

    # Calculamos el peso de cada subconjunto respecto al total
    peso_izq = len(y[izquierda]) / len(y)
    peso_der = len(y[derecha]) / len(y)
    # Calculamos la ganancia de información
    ganancia = entropia_total - (peso_izq * entropia_izquierda + peso_der * entropia_derecha)
    return ganancia

def mejor_umbral(X, y, atributo, prop_umbral=1.0):
    # Número total de ejemplos
    numero_ejemplo = X.shape[0]
    # Seleccionamos aleatoriamente una proporción de los ejemplos
    indices = np.arange(numero_ejemplo)
    np.random.shuffle(indices)
    num_prueba = int(numero_ejemplo * prop_umbral)
    # Subconjunto de ejemplos a considerar para buscar umbrales
    X_sub = X[indices[:num_prueba], atributo]
    y_sub = y[indices[:num_prueba]]
    # Ordenamos los ejemplos por el valor del atributo
    orden = np.argsort(X_sub)
    X_ord = X_sub[orden]
    y_ord = y_sub[orden]
    # Buscamos los puntos medios entre valores consecutivos donde cambia la clase
    umbrales = []
    for i in range(1, len(X_ord)):
        if y_ord[i] != y_ord[i-1]:
            umbral = (X_ord[i] + X_ord[i-1]) / 2
            umbrales.append(umbral)
    mejor_umbral_final = None
    mejor_ganancia = -np.inf  # Inicializamos con un valor muy bajo
    # Evaluamos la ganancia de información para cada umbral candidato
    for umbral in umbrales:
        # Crea una máscara para los ejemplos que cumplen la condición es decire true donde lo cumple y false donde no
        izquierda = X[:, atributo] <= umbral
        derecha = X[:, atributo] > umbral
        ganancia = ganancia_informacion(y, izquierda, derecha)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral_final = umbral
    return mejor_umbral_final, mejor_ganancia

def mejor_atributo_umbral(X, y, prop_umbral=1.0):
    mejor_atributo = None
    mejor_umbral_final = None
    mejor_ganancia = -np.inf  # Inicializamos con un valor muy bajo
    # Iteramos sobre cada atributo
    for atributo in range(X.shape[1]):
        umbral, ganancia = mejor_umbral(X, y, atributo, prop_umbral)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_atributo = atributo
            mejor_umbral_final = umbral
    return mejor_atributo, mejor_umbral_final


def CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0, prop_umbral=1.0):
    # Comprobamos si hemos alcanzado la profundidad máxima o el número mínimo de ejemplos
    es_profundidad_maxima = prof >= max_prof
    hay_pocos_ejemplos = X.shape[0] < min_ejemplos_nodo_interior
    todas_las_clases_iguales = len(np.unique(y)) == 1

    if es_profundidad_maxima or hay_pocos_ejemplos or todas_las_clases_iguales:
        # Creamos un nodo hoja con la distribución de clases
        distr = {}
        for clase in np.unique(y):
            distr[clase] = np.count_nonzero(y == clase)
        clase = max(distr, key=distr.get)
        return Nodo(distr=distr, clase=clase)
    else:
        mejor_atributo, mejor_umbral = mejor_atributo_umbral(X, y, prop_umbral)

        if mejor_atributo is None or mejor_umbral is None:
            distr = {}
            for clase in np.unique(y):
                distr[clase] = np.count_nonzero(y == clase)
            clase = max(distr, key=distr.get)
            return Nodo(distr=distr, clase=clase)
        
        X_izq = X[X[:, mejor_atributo] <= mejor_umbral]
        y_izq = y[X[:, mejor_atributo] <= mejor_umbral]
        X_der = X[X[:, mejor_atributo] > mejor_umbral]
        y_der = y[X[:, mejor_atributo] > mejor_umbral]

        if len(y_izq) == 0 or len(y_der) == 0:
            distr = {}
            for clase in np.unique(y):
                distr[clase] = np.count_nonzero(y == clase)
            clase = max(distr, key=distr.get)
            return Nodo(distr=distr, clase=clase)
        
        # Llamadas recursivas para construir los subárboles
        izq = CONSTRUYE_ARBOL(X_izq, y_izq, min_ejemplos_nodo_interior, max_prof, prof + 1)
        der = CONSTRUYE_ARBOL(X_der, y_der, min_ejemplos_nodo_interior, max_prof, prof + 1)
        # Creamos un nodo interior con el atributo y umbral seleccionados
        distr = {}
        for clase in np.unique(y):
            distr[clase] = np.count_nonzero(y == clase)
        return Nodo(atributo=mejor_atributo, umbral=mejor_umbral, izq=izq, der=der, distr=distr)
    
class ArbolDecision:
    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.arbol = None

        # Aqui fue una "mejora" que tuvimos que hacer porque estabamos teniendo problemas con el rendimiento y finalmente era porque no guardabamos los indices
        # de los atributos seleccionados, y al clasificar teníamos que volver a seleccionar
        # los atributos aleatorios, lo que hacía que el rendimiento fuera muy bajo.
        self.indices_atributos = None
               
    def entrena(self, X, y):
        # Comprobamos que X e y son arrays de numpy
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X e y deben ser arrays de numpy")
        # Comprobamos que X e y tienen el mismo número de ejemplos
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de ejemplos")
        # Comprobamos que hay suficientes ejemplos para entrenar
        if X.shape[0] < self.min_ejemplos_nodo_interior:
            print(X.shape[0], self.min_ejemplos_nodo_interior)
            raise ValueError("No hay suficientes ejemplos para entrenar el modelo en entrena")
        

        # Seleccionamos aleatoriamente n_atrs atributos de X
        if self.n_atrs > X.shape[1]:
            raise ValueError("n_atrs no puede ser mayor que el número de atributos en X")
        self.indices_atributos = np.random.choice(X.shape[1], self.n_atrs, replace=False)
        X_seleccionado = X[:, self.indices_atributos]
        # Llamada recursiva para construir el árbol
        self.arbol = CONSTRUYE_ARBOL(X_seleccionado, y, self.min_ejemplos_nodo_interior, self.max_prof, prof=0, prop_umbral=self.prop_umbral)
        

        

    def clasifica(self, X):
        if self.arbol is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado aún.")
        if not isinstance(X, np.ndarray):
            raise ValueError("X debe ser un array de numpy")
        X_sel = X[:, self.indices_atributos]
        predicciones = []
        for x in X_sel:
            nodo = self.arbol  # Reinicia el nodo a la raíz para cada ejemplo
            while not nodo.es_hoja():
                if x[nodo.atributo] <= nodo.umbral:
                    nodo = nodo.izq
                else:
                    nodo = nodo.der
            predicciones.append(nodo.clase)
        return np.array(predicciones)

    def clasifica_prob(self, x):
        #Para un solo ejemplo x, devuelve un diccionario con la probabilidad de pertenecer a cada clase
        nodo = self.arbol
        if nodo is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado aún.")
        x = np.array(x)
        while not nodo.es_hoja():
            if x[nodo.atributo] <= nodo.umbral:
                nodo = nodo.izq
            else:
                nodo = nodo.der
        return nodo.distr
    

    def imprime_arbol(self,nombre_atrs,nombre_clase) :
        nodo = self.arbol
        if nodo is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado aún.")
        nombre_atrs_sel = [nombre_atrs[i] for i in self.indices_atributos]
        def imprime_nodo(nodo, profundidad=0):
            if nodo.es_hoja():
                # Imprimimos la clase y la distribución de clases del nodo hoja
                print(" " * profundidad * 4 + f"{nombre_clase}: {nodo.clase} -- {nodo.distr}")
            else:
                # Imprimimos la condición del nodo
                print(" " * profundidad * 4 + f"{nombre_atrs_sel[nodo.atributo]} <= {nodo.umbral:.3f}")
                # Llamamos recursivamente a los hijos izquierdo y derecho
                imprime_nodo(nodo.izq, profundidad + 1)
                print(" " * profundidad * 4 + f"{nombre_atrs_sel[nodo.atributo]} > {nodo.umbral:.3f}")
                imprime_nodo(nodo.der, profundidad + 1)
        # Llamamos a la función auxiliar para imprimir el árbol
        imprime_nodo(nodo)
#  El constructor tiene los siguientes argumentos de entrada:

#     + min_ejemplos_nodo_interior: mínimo número de ejemplos del conjunto de 
#       entrenamiento en un nodo del árbol que se aprende, para que se considere 
#       su división.  
#     + max_prof: profundidad máxima del árbol que se aprende.
#     + n_atrs: número de atributos candidatos a considerar en cada partición
#     + prop_umbral: proporción de ejemplos a considerar cuando se buscan los 
#       umbrales candidatos.    
  
#      

# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente.
#     

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Método clasifica_prob: recibe UN EJEMPLO y devuelve un diccionario con la predicción
#   de probabilidad de pertenecer a cada clase. Esa probabilidad se calcula como la
#   proporción de ejemplos de clase en la distribución del nodo hoja que da la
#   predicción.

# * Método imprime_arbol: recibe la lista de nombres de cada atributo (columnas) y el
#   nombre del atributo de clasificación, e imprime el árbol de decisión aprendido 
#   (ver ejemplos más abajo) [SUGERENCIA: hacerlo con una función auxiliar recursiva] 


# Si se llama al método de clasificación, o al de impresión, antes de entrenar el modelo,
# se debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        




# Algunos ejemplos (los resultados pueden variar, debido a la aleatoriedad)
# **************************************************************************

# TITANIC
# -------

# clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
# clf_titanic.entrena(X_train_titanic, y_train_titanic)
# clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Sobrevive")

# Mujer <= 0.000
#      Edad <= 11.000
#           Pclass <= 2.500
#                Sobrevive: 1 -- {1: 10}
#           Pclass > 2.500
#                Sobrevive: 0 -- {0: 13, 1: 8}
#      Edad > 11.000
#           Pclass <= 1.000
#                Sobrevive: 0 -- {0: 62, 1: 30}
#           Pclass > 1.000
#                Sobrevive: 0 -- {0: 270, 1: 32}
# Mujer > 0.000
#      Pclass <= 2.000
#           Edad <= 2.000
#                Sobrevive: 0 -- {0: 1, 1: 1}
#           Edad > 2.000
#                Sobrevive: 1 -- {0: 5, 1: 122}
#      Pclass > 2.000
#           Edad <= 38.500
#                Sobrevive: 1 -- {0: 46, 1: 58}
#           Edad > 38.500
#                Sobrevive: 0 -- {0: 9, 1: 1}

# VOTOS
# -----

# >>> clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
# >>> clf_votos.entrena(Xe_votos, ye_votos)
# >>> nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
# >>> clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")

# Votación 4 <= 0.000
#      Votación 3 <= 0.000
#           Votación 11 <= 0.000
#                Votación 13 <= 0.500
#                     Votación 14 <= -0.500
#                          Partido: democrata -- {'democrata': 2}
#                     Votación 14 > -0.500
#                          Partido: republicano -- {'republicano': 3}
#                Votación 13 > 0.500
#                     Votación 7 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 7 > -1.000
#                          Partido: democrata -- {'democrata': 4}
#           Votación 11 > 0.000
#                Partido: democrata -- {'democrata': 11}
#      Votación 3 > 0.000
#           Partido: democrata -- {'democrata': 149}
# Votación 4 > 0.000
#      Votación 11 <= 0.500
#           Votación 10 <= -1.000
#                Votación 12 <= -1.000
#                     Votación 3 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 3 > -1.000
#                          Partido: republicano -- {'republicano': 2}
#                Votación 12 > -1.000
#                     Votación 3 <= 0.000
#                          Partido: republicano -- {'republicano': 35}
#                     Votación 3 > 0.000
#                          Partido: republicano -- {'democrata': 1, 'republicano': 2}
#           Votación 10 > -1.000
#                Partido: republicano -- {'republicano': 55}
#      Votación 11 > 0.500
#           Votación 7 <= -1.000
#                Votación 3 <= -1.000
#                     Votación 13 <= 0.000
#                          Partido: democrata -- {'democrata': 1}
#                     Votación 13 > 0.000
#                          Partido: republicano -- {'democrata': 2, 'republicano': 9}
#                Votación 3 > -1.000
#                     Partido: democrata -- {'democrata': 6}
#           Votación 7 > -1.000
#                Partido: republicano -- {'republicano': 4}


# IRIS
# ----

# clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# clf_iris.entrena(X_train_iris, y_train_iris)
# clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")



#  Long. Pétalo <= 2.450
#       Clase: 0 -- {0: 33}
#  Long. Pétalo > 2.450
#       Long. Pétalo <= 4.900
#            Anch. Pétalo <= 1.650
#                 Clase: 1 -- {1: 32}
#            Anch. Pétalo > 1.650
#                 Clase: 2 -- {1: 1, 2: 3}
#       Long. Pétalo > 4.900
#            Clase: 2 -- {2: 30}


# CÁNCER DE MAMA
# --------------

# >>> clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer.entrena(Xev_cancer, yev_cancer)

# >>> nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension',
#        'radius error', 'texture error', 'perimeter error', 'area error',
#        'smoothness error', 'compactness error', 'concavity error',
#        'concave points error', 'symmetry error',
#        'fractal dimension error', 'worst radius', 'worst texture',
#        'worst perimeter', 'worst area', 'worst smoothness',
#        'worst compactness', 'worst concavity', 'worst concave points',
#        'worst symmetry', 'worst fractal dimension']

# >>> clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")


#  mean concave points <= 0.051
#       mean area <= 696.050
#            area error <= 34.405
#                 mean area <= 505.550
#                      Es benigno: 1 -- {1: 172}
#                 mean area > 505.550
#                      worst texture <= 30.145
#                           mean concave points <= 0.050
#                                Es benigno: 1 -- {1: 63}
#                           mean concave points > 0.050
#                                Es benigno: 0 -- {0: 1, 1: 1}
#                      worst texture > 30.145
#                           mean texture <= 24.840
#                                compactness error <= 0.013
#                                     Es benigno: 0 -- {0: 3}
#                                compactness error > 0.013
#                                     Es benigno: 1 -- {1: 2}
#                           mean texture > 24.840
#                                Es benigno: 1 -- {1: 11}
#            area error > 34.405
#                 mean concave points <= 0.032
#                      Es benigno: 1 -- {1: 7}
#                 mean concave points > 0.032
#                      mean perimeter <= 89.175
#                           Es benigno: 0 -- {0: 3}
#                      mean perimeter > 89.175
#                           mean texture <= 20.115
#                                Es benigno: 1 -- {1: 3}
#                           mean texture > 20.115
#                                Es benigno: 0 -- {0: 1}
#       mean area > 696.050
#            mean texture <= 16.190
#                 Es benigno: 1 -- {1: 4}
#            mean texture > 16.190
#                 worst fractal dimension <= 0.066
#                      Es benigno: 1 -- {1: 2}
#                 worst fractal dimension > 0.066
#                      Es benigno: 0 -- {0: 6}
#  mean concave points > 0.051
#       mean area <= 790.850
#            worst texture <= 25.655
#                 mean concave points <= 0.079
#                      mean concave points <= 0.052
#                           Es benigno: 0 -- {0: 1}
#                      mean concave points > 0.052
#                           Es benigno: 1 -- {1: 20}
#                 mean concave points > 0.079
#                      Es benigno: 0 -- {0: 6}
#            worst texture > 25.655
#                 perimeter error <= 1.558
#                      Es benigno: 0 -- {0: 1, 1: 1}
#                 perimeter error > 1.558
#                      Es benigno: 0 -- {0: 37}
#       mean area > 790.850
#            Es benigno: 0 -- {0: 111}



# EJEMPLOS DE RENDIMIENTOS OBTENIDOS CON LOS CLASIFICADORES:
# ----------------------------------------------------------

# Usamos la siguiente función para medir el rendimiento (proporción de aciertos) 
# de un clasificador sobre un conjunto de ejemplos:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/X.shape[0]
    

# Ejemplos (obviamente, el resultado puede variar):


# print(rendimiento(clf_titanic,X_train_titanic,y_train_titanic))
# 0.8158682634730539
# print(rendimiento(clf_titanic,X_test_titanic,y_test_titanic))
# 0.7982062780269058

# >>> rendimiento(clf_votos,Xe_votos,ye_votos)
# 0.9827586206896551
# >>> rendimiento(clf_votos,Xp_votos,yp_votos)
# 0.9310344827586207

# >>> rendimiento(clf_iris,X_train_iris,y_train_iris)
#  0.98989898989899
# >>> rendimiento(clf_iris,X_test_iris,y_test_iris)
# 0.9607843137254902

# >>> rendimiento(clf_cancer,Xev_cancer,yev_cancer)
# 0.9956140350877193
# >>> rendimiento(clf_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806































# =============================================
# EJERCICIO 3: IMPLEMENTACIÓN DE RANDOM FORESTS
# =============================================

# Usando la clase ArbolDecision, implementar un clasificador Random Forest. 

# Un clasificador Random Forest aplica dos técnicas que reducen el sobreajuste que 
# pudiéramos tener con un único árbol de decisión:

# - En lugar de aprender un árbol. se aprenden varios árboles y a la hora de clasificar
#   nuevos ejemplos, se devuelve la clasificación mayoritaria.
# - Cada uno de esos árboles no se aprende con el conjunto de entrenamiento original, sino
#   con una muestra de ejemplos, obtenido seleccionado los ejemplos aleatoriamente del 
#   conjunto total, CON REEMPLAZO. Además, durante el aprendizaje y en cada nodo, no se usan todos
#   los atributos sino un sunconjunto de ellos obtenidos aleatoriamente (el mismo para todo el árbol). 

# NOTA IMPORTANTE: En la versión estándar del algoritmo Random Forest, el subconjunto de
# atributos a considerar se sortea EN CADA NODO de los árboles que se aprenden. Sin
# embargo, en nuestro caso, como vamos a usar la clase ArbolDecision del ejercicio
# anterior, se va usar el mismo subconjunto de atributos EN CADA ÁRBOL APRENDIDO.

# Concretando, se pide implementar una clase RandomForest con la siguiente estructura:


class RandomForest:
    def __init__(self, n_arboles=5,prop_muestras=1.0,
                       min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.arboles = []

    def entrena(self, X, y):
        for _ in range(self.n_arboles):
            # Seleccionamos aleatoriamente una muestra de ejemplos con reemplazo
            indices = np.random.choice(X.shape[0], int(X.shape[0] * self.prop_muestras), replace=True)
            X_muestra = X[indices]
            y_muestra = y[indices]
            # Creamos un árbol de decisión con los parámetros dados
            arbol = ArbolDecision(min_ejemplos_nodo_interior=self.min_ejemplos_nodo_interior,
                                  max_prof=self.max_prof, n_atrs=self.n_atrs, prop_umbral=self.prop_umbral)
            arbol.entrena(X_muestra, y_muestra)
            self.arboles.append(arbol)

    def clasifica(self, X):
        y = []
        for x in X:
            predicciones = []
            for arbol in self.arboles:
                prediccion = arbol.clasifica(np.array([x]))
                predicciones.append(prediccion[0])  # Añadimos la predicción del árbol en el que estamos a la lista total de predicciones de nuestro ejemplo
            values, counts = np.unique(predicciones, return_counts=True)
            clase_mayoritaria = values[np.argmax(counts)]      
            y.append(clase_mayoritaria)  # Añadimos la clase mayoritaria a la lista de predicciones
        return np.array(y)

    
# Los argumentos del constructor son:

# - n_arboles: el número de árboles que se van a obtener para el clasificador.
# - n_muestras: el número de ejemplos a muestrear para el aprendizaje de cada árbol.
# - El resto de argumentos son los mismos que en el ejercicio anterior, y se usan en el
#   aprendizaje de cada árbol.


# Ejemplos:
# *********

# VOTOS:
# ------

# clf_votos_rf=RandomForest(n_arboles=10,min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=6,prop_umbral=0.8)
# clf_votos_rf.entrena(Xe_votos, ye_votos)
# print(rendimiento(clf_votos_rf,Xe_votos,ye_votos))
# # 0.9517241379310345
# print(rendimiento(clf_votos_rf,Xp_votos,yp_votos))
# # 0.9586206896551724


# >>> clf_cancer_rf = RandomForest(n_arboles=15,min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer_rf.entrena(Xev_cancer, yev_cancer)
# >>> rendimiento(clf_cancer_rf,Xev_cancer,yev_cancer)
# 1.0
# >>> rendimiento(clf_cancer_rf,Xp_cancer,yp_cancer)
# 0.9911504424778761


#------------------------------------------------------------------------------
























# =========================================
# EJERCICIO 4: AJUSTANDO LOS CLASIFICADORES
# =========================================

# En este ejercicio vamos a tratar de obtener buenos clasificadores para los 
# los siguientes conjuntos de datos: IMDB, credito, AdultDataset y dígitos.

# ---------------------------
# 4.1 PREPARANDO LOS DATASETS     
# ---------------------------

# Excepto a IMDB, que ya se carga cuando se ejecuta carga_datos.py, el resto 
# tendremos que hacer antes algún preprocesado:
    
# - En X_credito, los atributos son categóricos, así que hay que transformarlos 
#   en numéricos para que se puedan usar con nuestros árboles de decisión. 
#   En el caso de árboles de decisión no es necesario hacer "one hot encoding",
#   sino que basta con codificar los valores de los atributos con números naturales
#   Para ello, SE PIDE USAR el OrdinalEncoder de sklearn.preprocessing (ver manual). 
#   Será necesario también separar en conjunto de prueba y de entrenamiento y
#   validación. 

# - El dataset AdultDataset nos viene es un archivo csv. Cargarlo con 
#   read_csv de pandas, separarlo en entrenamiento y prueba 
#   y aplicarle igualmente OrdinalEncoder, pero sólo a las características desde la 
#   quinta en adelante (ya que las cuatro primeras columnas ya son numéricas). 

# - El dataset de dígitos los podemos obtener a partir de los datos que están en 
#   la carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#   píxeles, y cada pixel vendrá representado por un caracter "espacio en
#   blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#   (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#   (es decir, no distinguiremos entre el borde y el interior). En cada
#   conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#   clasificaciones de cada imagen (es decir, el número que representan) vienen
#   en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#   funciones python que lean esos ficheros y obtengan los datos en el mismo
#   formato numpy en el que los necesita el clasificador. 
#   Los datos están ya separados en entrenamiento, validación y prueba. 

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

#   Se pide incluir aquí las definiciones y órdenes necesarias para definir
#   las siguientes variables, con los datasets anteriores como arrays de numpy.


# * X_train_credito, y_train_credito, X_test_credito, y_test_credito
#   conteniendo el dataset de crédito con los atributos numñericos:X
encoder = OrdinalEncoder()
X_preprocess = encoder.fit_transform(X_credito)
from sklearn.preprocessing import OrdinalEncoder

# Codificar atributos categóricos
encoder = OrdinalEncoder()
X_credito_numerico = encoder.fit_transform(X_credito)

# Separar en entrenamiento+validación y test
X_train_val, X_test, y_train_val, y_test = particion_entr_prueba(X_credito_numerico, y_credito, test=0.2)

# Separar entrenamiento+validación en entrenamiento y validación
X_train_credito, X_val, y_train_credito, y_val = particion_entr_prueba(X_train_val, y_train_val, test=0.2)



X_test_credito = X_test
y_test_credito = y_test
# * X_train_adult, y_train_adult, X_test_adult, y_test_adult
#   conteniendo el AdultDataset con los atributos numéricos:

#   Cargar el dataset AdultDataset.csv con pandas, y separar en entrenamiento y prueba.

adult_df = pd.read_csv('datos/AdultDataset.csv')
# Separar en entrenamiento y prueba (Al tener todo junto es decir los datos y la clasificacion debemos de separarlo con 
# adult_df.iloc[:, :-1].values y adult_df.iloc[:, -1].values)
X_train_adult, X_test_adult, y_train_adult, y_test_adult = particion_entr_prueba(adult_df.iloc[:, :-1].values, adult_df.iloc[:, -1].values, test=0.2)
# Aplicar OrdinalEncoder a las características desde la quinta en adelante
encoder_adult = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_adult[:, 4:] = encoder_adult.fit_transform(X_train_adult[:, 4:])
X_test_adult[:, 4:] = encoder_adult.transform(X_test_adult[:, 4:])

                                                           







# * X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
#   conteniendo el dataset de los dígitos escritos a mano:
with open('./datos/digitdata/testimages', 'r') as f:
    num_lineas = sum(1 for _ in f)
n_images_test = num_lineas // 28
# print("Número de imágenes test:", n_images_test)

with open('./datos/digitdata/trainingimages', 'r') as f:
    num_lineas = sum(1 for _ in f)
n_imagenes_training = num_lineas // 28
# print("Número de imágenes training:", n_imagenes_training)

with open('./datos/digitdata/validationimages', 'r') as f:
    num_lineas = sum(1 for _ in f)
n_imagenes_validation = num_lineas // 28
# print("Número de imágenes validacion:", n_imagenes_validation)

def carga_imagenes_digitos(ruta_archivo, n_imagenes):
    imagenes = []
    with open(ruta_archivo, 'r') as f:
        for _ in range(n_imagenes):
            imagen = []
            for _ in range(28):
                linea = f.readline()
                # Si la línea está vacía, hemos llegado al final del archivo
                if not linea:
                    break
                # Convertimos cada carácter a 0 (blanco) o 1 (negro)
                # El rstrip lo que quitamos es el caracter de salto de linea del final de cada linea para que no se nos convierta en un 1
                fila = [0 if c == ' ' else 1 for c in linea.rstrip('\n')]
                imagen.extend(fila)
                
            if len(imagen) == 784: # 784 = 28x28 que es cuano saltamos a la siguiente imagen
                imagenes.append(imagen)
    return np.array(imagenes)

def carga_etiquetas_digitos(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        etiquetas = [int(linea.strip()) for linea in f]
    return np.array(etiquetas)

X_test_dg = carga_imagenes_digitos('./datos/digitdata/testimages', n_images_test)
X_train_dg = carga_imagenes_digitos('./datos/digitdata/trainingimages', n_imagenes_training)
X_valid_dg = carga_imagenes_digitos('./datos/digitdata/validationimages', n_imagenes_validation)

# Variables para y de dígitos
y_train_dg = carga_etiquetas_digitos('./datos/digitdata/traininglabels')
y_valid_dg = carga_etiquetas_digitos('./datos/digitdata/validationlabels')
y_test_dg = carga_etiquetas_digitos('./datos/digitdata/testlabels')

# print("\nShapes de los conjuntos de dígitos:")
# print("X_train_dg:", X_train_dg.shape)
# print("X_valid_dg:", X_valid_dg.shape)
# print("X_test_dg:", X_test_dg.shape)
# print("y_train_dg:", y_train_dg.shape)
# print("y_valid_dg:", y_valid_dg.shape)
# print("y_test_dg:", y_test_dg.shape)


# -----------------------------
# 4.2 AJUSTE DE HIPERPARÁMETROS     
# -----------------------------

# En nuestra implementación de RandomForest tenemos los siguientes 
# hiperparámetros: 

# n_arboles
# prop_muestras
# min_ejemplos_nodo_interior
# max_prof
# n_atrs
# prop_umbral

# Se trata ahora de encontrar, en cada dataset, una buena combinación de valores para esos 
# hiperparámetros, tratando de obtener un buen rendimiento de los clasificadores. Hacerlo
# usando un conjunto de validación: según se ha visto en la teoría, esto consiste en particionar  
# en entrenamiento, validación y prueba, entrenando por cada combinación de hiperparámetros 
# con el conjunto de entrenamiento y evaluando el rendimiento en validación. El entrenamiento final 
# con la mejor combinación ha de hacerse en la unión de entrenamiento y validación.
    

# NO ES NECESARIO ser demasiado exhaustivo, basta con probar algunas combinaciones, 
# pero sí es importante describir el proceso realizado y las mejores combinaciones 
# encontradas en cada caso. 
# DEJAR ESTE APARTADO COMENTADO, para que no se ejecuten las pruebas realizadas cuando se cargue
# el archivo. 

# -----------------EXPLICACION DEL PROCESO-----------------
# Para cada dataset, se ha realizado una búsqueda de hiperparámetros utilizando varios bucles anidados
# que iteran sobre diferentes valores de los hiperparámetros. En cada iteración, se entrena un modelo
# con nuestra implementacion de Random Forest (con el conjunto de entrenamiento) con la combinación actual de hiperparámetros y se evalúa su rendimiento en un conjunto
# de validación. Se guarda la mejor combinación de hiperparámetros y su rendimiento. Al final, se entrena
# el modelo final con la mejor combinación de hiperparámetros utilizando el conjunto de entrenamiento y validación.
# ---------------------------------------------------------
# AJUSTE IMDB
n_arboles = [10, 15]
prop_muestras = [0.7]
min_ejemplos_nodo_interior = [3,4]
max_prof = [6, 7]
n_atrs = [20, 50]
prop_umbral = [0.60]
hiperparametros_resultado = {}
X_train_val_imdb = X_train_imdb.copy()
y_train_val_imdb = y_train_imdb.copy()
X_trainfinal_imdb, X_val_imdb, y_trainfinal_imdb, y_val_imdb = particion_entr_prueba(X_train_val_imdb, y_train_val_imdb, test=0.4)

# for arbol in n_arboles:
#     for muestra in prop_muestras:
#         for min_ejemplo in min_ejemplos_nodo_interior:
#             for prof in max_prof:
#                 for atr in n_atrs:
#                     for umbral in prop_umbral: 
#                         mejor_rendimiento = 0
#                         mejor_combinacion = None
#                         RF_IMDB = RandomForest(n_arboles=arbol, prop_muestras=muestra,
#                                               min_ejemplos_nodo_interior=min_ejemplo,
#                                               max_prof=prof, n_atrs=atr, prop_umbral=umbral)
                        
#                         RF_IMDB.entrena(X_trainfinal_imdb, y_trainfinal_imdb)
#                         rendimiento_val = rendimiento(RF_IMDB, X_val_imdb, y_val_imdb)
#                         if rendimiento_val > mejor_rendimiento:
#                             mejor_rendimiento = rendimiento_val
#                             mejor_combinacion = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         # Guardamos el rendimiento y la combinación de hiperparámetros
#                         dicc = {}
#                         dicc[rendimiento_val] = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         hiperparametros_resultado["IMDB"] = dicc

# print("Mejor combinación de hiperparámetros para IMDB:", mejor_combinacion, mejor_rendimiento)


# AJUSTE ADULTS
X_train_val_adult = X_train_adult.copy()
y_train_val_adult = y_train_adult.copy()
X_trainfinal_adult, X_val_adult, y_trainfinal_adult, y_val_adult = particion_entr_prueba(X_train_val_adult, y_train_val_adult, test=0.4)

n_arboles = [10, 15]
prop_muestras = [0.7]
min_ejemplos_nodo_interior = [4]
max_prof = [10]
n_atrs = [7, 8, 10]
prop_umbral = [0.60]

# for arbol in n_arboles:
#     for muestra in prop_muestras:
#         for min_ejemplo in min_ejemplos_nodo_interior:
#             for prof in max_prof:
#                 for atr in n_atrs:
#                     for umbral in prop_umbral: 
#                         mejor_rendimiento = 0
#                         mejor_combinacion = None
#                         RF_ADULT = RandomForest(n_arboles=arbol, prop_muestras=muestra,
#                                                min_ejemplos_nodo_interior=min_ejemplo,
#                                                max_prof=prof, n_atrs=atr, prop_umbral=umbral)
#                         RF_ADULT.entrena(X_trainfinal_adult, y_trainfinal_adult)
#                         rendimiento_val = rendimiento(RF_ADULT, X_val_adult, y_val_adult)
#                         if rendimiento_val > mejor_rendimiento:
#                             mejor_rendimiento = rendimiento_val
#                             mejor_combinacion = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         dicc = {}
#                         dicc[rendimiento_val] = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         hiperparametros_resultado["ADULT"] = dicc

# print("Mejor combinación de hiperparámetros para ADULT:", mejor_combinacion, mejor_rendimiento)

# AJUSTE DIGITS

# Unimos entrenamiento y validación para el ajuste
X_train_val_dg = X_train_dg.copy()
y_train_val_dg = y_train_dg.copy()
X_trainfinal_dg, X_val_dg, y_trainfinal_dg, y_val_dg = particion_entr_prueba(X_train_val_dg, y_train_val_dg, test=0.4)
X_entr_dg = np.copy(X_trainfinal_dg)
y_entr_dg = np.copy(y_trainfinal_dg)
n_arboles = [10, 15]
prop_muestras = [0.7]
min_ejemplos_nodo_interior = [4]
max_prof = [7, 8, 10]
n_atrs = [40, 50, 70]
prop_umbral = [0.60]

# for arbol in n_arboles:
#     for muestra in prop_muestras:
#         for min_ejemplo in min_ejemplos_nodo_interior:
#             for prof in max_prof:
#                 for atr in n_atrs:
#                     for umbral in prop_umbral: 
#                         mejor_rendimiento = 0
#                         mejor_combinacion = None
#                         RF_DG = RandomForest(n_arboles=arbol, prop_muestras=muestra,
#                                              min_ejemplos_nodo_interior=min_ejemplo,
#                                              max_prof=prof, n_atrs=atr, prop_umbral=umbral)
#                         RF_DG.entrena(X_trainfinal_dg, y_trainfinal_dg)
#                         rendimiento_val = rendimiento(RF_DG, X_val_dg, y_val_dg)
#                         if rendimiento_val > mejor_rendimiento:
#                             mejor_rendimiento = rendimiento_val
#                             mejor_combinacion = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         dicc = {}
#                         dicc[rendimiento_val] = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         hiperparametros_resultado["DIGITS"] = dicc

# print("Mejor combinación de hiperparámetros para DIGITS:", mejor_combinacion, mejor_rendimiento)


# AJUSTE CREDITO

n_arboles = [10, 15]
prop_muestras = [0.7]
min_ejemplos_nodo_interior = [4]
max_prof = [7, 8, 10]
n_atrs = [3, 4, 6]
prop_umbral = [0.60]

# for arbol in n_arboles:
#     for muestra in prop_muestras:
#         for min_ejemplo in min_ejemplos_nodo_interior:
#             for prof in max_prof:
#                 for atr in n_atrs:
#                     for umbral in prop_umbral: 
#                         mejor_rendimiento = 0
#                         mejor_combinacion = None
#                         RF_CREDITO = RandomForest(n_arboles=arbol, prop_muestras=muestra,
#                                                  min_ejemplos_nodo_interior=min_ejemplo,
#                                                  max_prof=prof, n_atrs=atr, prop_umbral=umbral)
#                         RF_CREDITO.entrena(X_train_credito, y_train_credito)
#                         rendimiento_val = rendimiento(RF_CREDITO, X_val, y_val)
#                         if rendimiento_val > mejor_rendimiento:
#                             mejor_rendimiento = rendimiento_val
#                             mejor_combinacion = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         dicc = {}
#                         dicc[rendimiento_val] = (arbol, muestra, min_ejemplo, prof, atr, umbral)
#                         hiperparametros_resultado["CREDITO"] = dicc

# print("Mejor combinación de hiperparámetros para CREDITO:", mejor_combinacion, mejor_rendimiento)

# print("Diccionario de hiperparámetros resultado:", hiperparametros_resultado)

# Esto no estaba en ningun lado del codigo pero nos era necesario para poder hacer los tests.
X_train_iris, X_test_iris, y_train_iris, y_test_iris = particion_entr_prueba(X_iris, y_iris, test=0.2)

#--------------IMPORTANTE-----------
# En estos dos casos las mejores combinaciones han sido esas pero sin embargo, tarda mucho tiempo en ejecutarse, por lo que se ha dejado comentado, y en el test
# se ha usado una combinación más sencilla para que no tarde tanto en ejecutarse pero igualmente obtiene buenos resultados.
# Mejor combinación de hiperparámetros para ADULT: (15, 0.7, 4, 10, 10, 0.6) 0.8482579902101929
# Mejor combinación de hiperparámetros para DIGITS: (15, 0.7, 4, 10, 70, 0.6) 0.9108662994491737
#-----------------------------------

# Diccionario de hiperparámetros resultado: {'IMDB': {np.float64(0.5844806007509387): (15, 0.7, 4, 7, 50, 0.6)}, 'ADULT': {np.float64(0.8482579902101929): (15, 0.7, 4, 10, 10, 0.6)}, 'DIGITS': {np.float64(0.9108662994491737): (15, 0.7, 4, 10, 70, 0.6)}, 'CREDITO': {np.float64(0.9223300970873787): (15, 0.7, 4, 10, 6, 0.6)}}


# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES NECESARIAS
# Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENEN A CONTINUACIÓN.

# EN EL APARTADO FINAL DE "RENDIMIENTOS FINALES RANDOM FOREST", USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE DEL EJERCICIO 4

# ESTE ARCHIVO trabajo_aia_23_24_parte_I.py SERÁ CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos_trabajo_aia.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS, 
# PARA EViTAR LOS ERRORES EN LA CARGA Y EJECUCIÓN.   


# *********** DESCOMENTAR A PARTIR DE AQUÍ

print("************ PRUEBAS EJERCICIO 1:")
print("**********************************\n")
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
print("\n")


Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
print("\n")


Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
print("\n\n\n")





print("************ PRUEBAS EJERCICIO 2:")
print("**********************************\n")

clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
clf_titanic.entrena(X_train_titanic, y_train_titanic)
clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Sobrevive")
rend_train_titanic = rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
rend_test_titanic = rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
print(f"****** Rendimiento DT titanic train: {rend_train_titanic}")
print(f"****** Rendimiento DT titanic test: {rend_test_titanic}\n\n\n\n ")




clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
clf_votos.entrena(Xe_votos, ye_votos)
nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")
rend_train_votos = rendimiento(clf_votos,Xe_votos,ye_votos)
rend_test_votos = rendimiento(clf_votos,Xp_votos,yp_votos)
print(f"****** Rendimiento DT votos en train: {rend_train_votos}")
print(f"****** Rendimiento DT votos en test:  {rend_test_votos}\n\n\n\n")


clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
clf_iris.entrena(X_train_iris, y_train_iris)
clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")
rend_train_iris = rendimiento(clf_iris,X_train_iris,y_train_iris)
rend_test_iris = rendimiento(clf_iris,X_test_iris,y_test_iris)
print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")





clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
clf_cancer.entrena(Xev_cancer, yev_cancer)
nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension']
clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")
rend_train_cancer = rendimiento(clf_cancer,Xev_cancer,yev_cancer)
rend_test_cancer = rendimiento(clf_cancer,Xp_cancer,yp_cancer)
print(f"***** Rendimiento DT cancer en train: {rend_train_cancer}")
print(f"***** Rendimiento DT cancer en test: {rend_test_cancer}\n\n\n")



print("************ RENDIMIENTOS FINALES RANDOM FOREST")
print("************************************************\n")


# ATENCIÓN: EN CADA CASO, INCORPORAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS 
# QUE SE HA OBTENIDO EN EL PROCESO DE AJUSTE



# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
# RF_IMDB=RandomForest(n_arboles=15 , prop_muestras=0.6 , min_ejemplos_nodo_interior= 3, max_prof=6 , n_atrs= 50, prop_umbral= 0.6) # ATENCIÓN: incorporar aquí los mejores valoeres de los parámetros tras el ajuste
# RF_IMDB.entrena(X_train_imdb,y_train_imdb) 
# print("Rendimiento RF entrenamiento sobre imdb: ",rendimiento(RF_IMDB,X_train_imdb,y_train_imdb))
# print("Rendimiento RF test sobre imdb: ",rendimiento(RF_IMDB,X_test_imdb,y_test_imdb))
# print("\n")



# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")

# RF_CREDITO=RandomForest(n_arboles= 15, prop_muestras= 0.7, min_ejemplos_nodo_interior= 4, max_prof=10 , n_atrs=6 , prop_umbral=0.6 ) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_CREDITO.entrena(X_train_credito,y_train_credito) 
# print("Rendimiento RF entrenamiento sobre crédito: ",rendimiento(RF_CREDITO,X_train_credito,y_train_credito))
# print("Rendimiento RF  test sobre crédito: ",rendimiento(RF_CREDITO,X_test_credito,y_test_credito))
# print("\n")


# print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")

# RF_ADULT=RandomForest(n_arboles= 2, prop_muestras= 0.2, min_ejemplos_nodo_interior= 4, max_prof=3 , n_atrs=3 , prop_umbral=0.2 ) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_ADULT.entrena(X_train_adult,y_train_adult) 
# print("Rendimiento RF  entrenamiento sobre adult: ",rendimiento(RF_ADULT,X_train_adult,y_train_adult))
# print("Rendimiento RF  test sobre adult: ",rendimiento(RF_ADULT,X_test_adult,y_test_adult))
# print("\n")


print("==== MEJOR RENDIMIENTO RF SOBRE DIGITOS:")
RF_DG=RandomForest(n_arboles= 15, prop_muestras= 0.7, min_ejemplos_nodo_interior= 4, max_prof=10 , n_atrs=70 , prop_umbral=0.6 ) # ATENCIÓN: incorporar aquí los mejores valors de losparámetros tras el ajuste
RF_DG.entrena(X_entr_dg,y_entr_dg)
print("Rendimiento RF entrenamiento sobre dígitos: ",rendimiento(RF_DG,X_entr_dg,y_entr_dg))
print("Rendimiento RF validación sobre dígitos: ",rendimiento(RF_DG,X_val_dg,y_val_dg))
print("Rendimiento RF test sobre dígitos: ",rendimiento(RF_DG,X_test_dg,y_test_dg))








