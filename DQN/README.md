# Tarea: DQN

### Alumno: José de Jesús Hernández Higuera

### Matrícula: 224470489
---
Indicaciones:

1) Implementar el código del capítulo 10
2) Entrenar y evaluar el modelo
3) Modificar el código para que el agente tenga como observación 3 puntos consecutivos de los datos.
4) Entrenar y evaluar el modelo

En esta carpeta del repositorio (Reinforcemente_Learning/DQN/) se encuentran los códigos, carpetas y datos (en formato .csv) relacionados al capítulo 10 del libro *Deep Reinforcment Learning Hands-On* de Maxim Lapan. Adicionalmente, se tienen un par de imágenes correspondientes a las gráficas de las actividades relacionadas en las indicaciones. A continuación se describe el funcionamiento de los códigos, la estructura de las carpetas y el procedimiento seguido para esta tarea.

## Estructura de la carpeta
En esta sección se detallan algunos de los códigos principales y las subcarpetas dentro de `DQN/`:

- `train_model.py`: Entrena el agente usando datos históricos contenidos en la carpeta data/.
- `run_model.py`: Evalúa el agente entrenado y genera gráficas de recompensa.
- `data/`: Archivos CSV con datos históricos de precios de 2015 y 2016.
- `lib/`: Contiene los códigos que desarrollan diferentes funciones dentro de los dos archivos anteriores:
    - `data.py`: Maneja y procesa los datos de los archivos en `data/`.
    - `environ.py`: Define la clase correspondiente al ambiente, `StocksEnv`, las acciones, los estados y los estados adaptados para una red convolucional.
    - `models.py`: Establece el tipo de red neuronal a utilizar.
- `saves/`: Contiene dos carpetas con los modelos entrenados en formato .dat que corresponden a los registros tomados cada 10,000 pasos. La carpeta llamada `simple_original/` corresponde a los datos del modelo tal y como está en el libro, mientras que `simple_bars3/` almacena aquellos en donde se hizo la modificación para tomar como observación 3 puntos consecutivos en los datos.
- `rewards-bars3.png`: gráfica resultante de la evaluación del modelo modificado.
- `rewards-original.png`: gráfica resultante del modelo original.

**NOTA**: en la carpeta original existe un archivo llamado `train_model_conv.py`, el cual realiza el entrenamiento del modelo utilizando una red convolucional. Sin embargo, no fue utilizado en este trabajo.

## Procedimiento
Se descargó el repositorio original del capítulo 10 del libro. Esta carpeta fue movida a la carpeta personal de tareas de la materia de Aprendizaje por Refuerzo y se le cambió el nombre a DQN para identificarlo como la tarea relacionada con este tema. Se realizó una breve exploración de los archivos para conocer los códigos y descomprimir la carpeta con los datos, la cual contenía dos archivos de tipo .csv y que llevan por nombre YNDX_150101_151231.csv y YNDX_160101_161231.csv, haciendo referencia a los datos del 2015 y 2016, respectivamente.

Adicionalmente, se realizó un ambiente virtual con la versión 3.11.7 de Python para poder trabajar con esta actividad. Asimismo, se instalaron todas las dependencias necesarias para la ejecución de los códigos. Las versiones de las librerías utilizadas son las siguientes:
- Numpy 1.24.4
- Torch 2.8.0+cu128
- gym 0.26.2
- gymnasium 0.29.1
- ptan 0.8.1

Finalmente, la implementación de CUDA se hizo para poder utilizar una tarjeta gráfica NVIDIA GeForce RTX 3060 dentro del entrenamiento.

Una vez teniendo los archivos completos y descomprimidos, se procedió con las indicaciones del libro, en el capítulo correspondiente, para la ejecución apropiada de los códigos. En primer lugar, se ejecutó desde la terminal el código llamado `train_model.py` para entrenar el agente. Por default, la ejecución ya ocupa algunos argumentos como el conjunto de datos a utilizar, los datos para la validación y el año correspondiente a los datos que se ocupan, de modo que no es necesario definirlos. Lo único necesario fue darle el nombre que tomará para construir los archivos relacionados con el experimento. Dado que lo primero que se hace es entrenar el modelo tal y como está, se eligió el nombre *original*. El entrenamiento del modelo se hizo por medio del siguiente comando:

```Bash
python train_model.py -r original --cuda
```

Se dejó correr durante, al menos, 50,000 episodios, lo cual generó cerca de 300,000 pasos. Durante este proceso, el código generó la carpeta `saves/simple-original/`, donde se fueron guardando los datos relacionados al entrenamiento del agente. A cada archivo se le da un nombre siguiendo la estructura model-NOMBRE-PASOS, donde el nombre, en este caso es *original* y PASOS es el número de pasos que ha dado el código al guardar el archivo. Se guarda un archivo por cada 10,000 pasos. 

La evaluación del código requiere de más argumentos a la hora de la ejecución:
- -d: es el archivo de los datos que se usarán.
- -m: es el archivo del modelo que se va a utilizar.
- -b: el número de barras que debe tomar el agente.
- -n: nombre que usará el programa para generar la gráfica.

Para ejecutar adecuadamente este archivo, se debe cuidar que los datos no sean los mismo que los de entrenamiento, que el modelo sea el último registrado (en este trbajo es aquel donde se llegó a los 300,000 pasos), que el número de barras coincida con las que están determinadas dentro de los archivos `train_model.py` y `environ.py` (em este caso son 10) y que el nombre sea el mismo que el que se utilizó en la ejecución anterior. Con todos estos elementos, el comando de ejecución del archivo fue el siguiente:

```Bash
python run_model.py -d YNDX_160101_161231.csv -m model-original-300000.dat -n original -b 10
```

La ejecución dió como resultado la gráfica de la imagen `rewards-original.png`.

Para el tercer y cuarto paso de las indicaciones se hizo una modificación en los archivos del ambiente, `environ.py`, y del entrenamiento del modelo, `train_model.py`. Para tomar únicamente 3 barras consecutivas del gráfico (o 3 puntos consecutivos de los datos), se debe modificar el parámetro `bars_count = 3`. En el código original se dejó comentado el valor original (10) y en la línea de abajo se colocó la asignación descrita. Con esta modificación, el proceso posterior fue prácitamente el mismo para el entrenamiento del modelo, la única diferencia fue el nombre, el cual fue cambiado a *bars3* para hacer referencia a las 3 barras que se toman como observación. El comando utilizado entonces fue:

```Bash
python train_model.py -r bars3 --cuda
```

Se dejó entrenando el modelo hasta alcanzar los 300,000 pasos, después se detuvo de manera manual. Posteriormente, se procedió a la ejecución del archivo de evaluación. El archivo del modelo que se utilizó fue el model-bars3-300000.dat y, por supuesto, el número de barras se deben configurar como 3. El comando de ejecución quedó de la siguiente forma:

```Bash
python run_model.py -d YNDX_160101_161231.csv -m model-bars3-300000.dat -n original -b 13
```

Finalmente, la gráfica resultante se guardó como un archivo titulado `rewards-bars3.png`

## Resultados

Se observa que, en ambos casos, la recompensa acumulada tiende a disminuir, lo que indica que el agente no logra mantener decisiones rentables de forma consistente. De hecho, la recompensa cae de manera más notable cuando se toman 3 barras consecutivas que cuando toma 10, lo cuál sugiere que el agente aprende mejor cuando tiene más datos del ambiente. Probablemente, una prueba con más barras o con el modelo convolucional podría arrojar un resultado más claro respecto a esta hipótesis. 

