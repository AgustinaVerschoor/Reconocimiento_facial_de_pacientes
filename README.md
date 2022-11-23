<h1 align="center"> Identificacion de usuarios mediante Reconocimiento Facial </h1>
<h1 align="center"> <img width="194" alt="image" src="https://user-images.githubusercontent.com/72108904/203089320-a75427f2-866b-4701-ae2d-25b2023565cc.png"> </h1>

<h2> Participantes </h2>

- Agustina Verschoor
- Justina Galarce 
- Sofia Tartara Moroni


<h2> Docentes </h2>

- Alejandro Silvestri
- Fabrizio Di Santo 

<h2> Índice <h2/>

* [Introduccion](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-introduccion-)
* [Objetivos](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-objetivos-)
* [Descripción del proyecto](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-descripcion-del-proyecto-)
* [Instalacion](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-instalacion-wrench-)
* [Funcionamiento](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-funcionamiento-hammer-)
* [Tecnologías utilizadas](https://github.com/AgustinaVerschoor/Reconocimiento_facial_de_pacientes/blob/master/README.md#-tecnologias-utilizadas-computer-)
  
<h2> Introduccion </h2>

Hoy en día, dentro del sistema de salud, uno de los problemas más frecuentes es la identificación de los pacientes que se encuentran dentro de la institución. Suele haber múltiples chequeos de identidad de los mismos, para evitar confusiones. Esta reiterada acción de control de identidad, requiere de más trabajo y más tiempo por parte de los profesionales de salud. El problema identificado, está en que no se puede identificar al paciente por número de habitación ni de camilla, ya que en esos contextos aparecen los errores, pues estos no permanecen en las mismas durante todo el recorrido dentro del sistema. Lo más seguro es que la identificación del paciente, esté siempre con el mismo, independientemente de la camilla u habitación, por lo tanto la solución propuesta en este proyecto, es el reconocimiento facial para la identificación de pacientes dentro de un hospital.


<h2> Objetivos </h2>

- Agilizar el proceso de adquisición de datos del usuario

- Optimizar el reconocimiento del usuario y automatizar la verificación de su identidad

- Mejorar la seguridad en el reconocimiento de la identidad del usuario


<h2> Descripcion del proyecto </h2> 

El reconocimiento facial funciona mediante la identificación y medición de los rasgos faciales en una imagen o video. Es una forma de identificación biométrica, que recoge un conjunto de datos biométricos únicos de cada persona, asociados a su rostro y expresión facial para identificar, verificar y/o autenticar a una persona. 
El proyecto consiste en la carga de datos biométricos de los pacientes a la hora del ingreso a la institución, que son cargados a la hora de la admisión de los mismos, donde se tomará un video de unos pocos segundos, en el cual se tomarán múltiples frames de la cara del paciente, y se guardarán en una carpeta con su nombre y dni. Esto se guardará en una base de datos que contiene a todos los pacientes admitidos en la institución. 
Luego de este paso, se entrenará al modelo de reconocimiento facial, lo cual tardará unos segundos en ejecutarse. 
Por último, cuando sea requerido, se aplicará el reconocimiento facial, donde el programa mediante una cámara web, reconoce los rostros de las personas en el frame, y en caso de que estén cargadas en la base de datos, aparecerá su nombre escrito en la pantalla junto a un recuadro verde que rodeará su rostro, mientras que si la persona frente a la camara no esta registrada, se verá recuadrada en rojo con la palabra “unknown” es decir desconocido. 
Así es como el programa carga los datos, los entrena y luego se aplica. 


<h2> Instalacion :wrench: </h2>

Para la correcta instalación del programa, debemos tener en cuenta la siguiente serie de pasos: 

1. Contar con una versión de python en el ordenador 
2. Contar con un ID compatible con python 
3. Contar con un interpreter adecuado a la versión de python

Recomendado: 
  - Python versión 3.9 
  - ID: pycharm
  - Interpreter: 

4. Creación de tres clases .py en un mismo directorio:
   - Schema.py
   - Trainer.py
   - Recognizer.py
5. Creación de un directorio destinado a la base de datos


<h2> Funcionamiento :hammer: </h2>

<h3> Clases involucradas </h3> 

- `Schema.py`: cargo de datos 
- `Trainer.py`: entrenamiento del modelo
- `Recognizer.py`: aplicacion del modelo entrenado

<h4> Schema: </h4>  
La clase Schema es la encargada del registro de datos biométricos de las diferentes personas. 

El primer paso es la importación de las siguientes librerías: 

```
import cv2 as cv 
import os 
import imutils 
```

Luego se deberá modificar el ‘dataPath’, ya que cambia según el ordenador donde se corra el programa: 

```
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i
```
Se selecciona el path original, se borra y luego de crear un directorio nuevo para la base de datos, se copia el Path absoluto y se lo pone entre comillas en reemplazo del original. </h2> 

<img width="383" alt="image" src="https://user-images.githubusercontent.com/72108904/203409919-cfb9c039-9314-48bd-ab45-4e16398b39e4.png"> </h2>



Es muy importante escribir exactamente el nombre del modelo a utilizar, ya que si hay un error en ello, no se creará el modelo.
```
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
Chequear: ‘haarcascade_frontalface_default.xml’ 

<h4> Trainer: </h4> 
Se deben importar las siguientes librerías: 

```
import cv2 as cv
import os
import numpy as np
```

Se cambia el dataPath por el del correspondiente ordenador: 

```
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i
```
<h4> Recognizer: </h4> 
Se deben importar las siguientes librerías: 

```
import cv2 as cv
import os
```
Se cambia el dataPath por el del correspondiente ordenador: 

```
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i
```

Se chequea nuevamente que el nombre del modelo sea correcto: 

```
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

<h3> Pasos de funcionamiento </h3> 

<h4> 1. Nuevo registro: </h4>  

   i. Cada vez que se quiera registrar a una persona nueva en la base de datos, se deberá asignar un nombre a la carpeta, en la clase Schema.py idealmente con el nombre y apellido de la persona a registrar: 
<h4 align="center"><img width="202" alt="image" src="https://user-images.githubusercontent.com/72108904/203565546-031ae588-1122-4376-9592-19945c0536f3.png"></h4> 

<h4> 2. Cargar en la base de datos: </h4>

   i. Luego de asignar el nombre a la carpeta, se debe correr la clase (Schema.py), lo que prenderá la cámara web de la computadora, y comenzara a tomar frames de la captura, las cuales serán guardadas en orden de captura en la carpeta creada.
IMP!: Se debe apuntar la cámara a la persona cargada en la carpeta, y no debe aparecer ningún otro rostro en el marco de la cámara web.
<h4> 3. Entrenar el modelo: </h4>

   i. Una vez modificada la base de datos, se debe correr la clase Trainer.py, lo que en caso de no existir, creará un modelo.xml nuevo, y en caso de existir, escribirá la nueva información sobre el mismo. 

   ii. Deberá aparecer de la siguiente manera:
<h4 align="center"><img width="162" alt="image" src="https://user-images.githubusercontent.com/72108904/203565773-dd8dd4d5-8806-4208-bf94-e204b324de8f.png"> </h4>

<h4> 4. Reconocimiento facial: </h4>
   i. Una vez cargada la nueva persona, y entrenado el modelo .xml, se deberá correr la clase Recognizer.py, para el reconocimiento facial. Se prendera la cámara nuevamente, y se reconoceran los rostros capturados en ella, y se los recuadrada. Si la persona frente a la cámara está registrada en la base de datos, entonces su rostro estará rodeado por un rectángulo verde y aparecerá una etiqueta con su nombre. Si la persona no esta registrada, entonces el rectángulo será rojo y la etiqueta dirá: “Unknown” (desconocido). 

<h2> Tecnologias utilizadas :computer: </h2>

- Pyhton 3.9
- Pycharm
- Open Cv
- OS
- modelo Haarcascade





