(los más populares en deep learning)

**Idea central

"Hagamos que el modelo NO SEPA de qué dominio viene el dato"

"Si un modelo no puede distinguir de qué dominio viene un dato, entonces ha aprendido una representación común"

LOS METODOS ADVERSARIALES intentar borrar la información de dominio, pero conservar la información de la tarea.

Es un equilibrio delicado.

Inspirados en GANs.

**¿Qué hacen?

Entrenan 3 cosas:

1. Extractor de features : Resumen la información del dato
2. Clasificador de tarea:  Usa los features para predecir Y
3. Discriminador de dominio: Intenta adivinar: "¿Esto viene de source o target?"

**El extractor: 
* Quiere ayudar al clasificador
* Pero quiere engañar al discriminador , aprende features que confunden al discriminador
![[Pasted image 20260125174045.png]]

Ejemplo mental
* El discriminador dice: "Esto es simulado"
* El extractor responde: "no, parecen iguales"


El truco está en:

Invertir el gradiente del discriminador hacia el extractor.


**Métodos famosos

* DANN: Entrena todo junto  ( Clasificador minimiza error, discriminador distinguie dominios, extractor confunde dominios)
* ADDA: Entrena source primero , luego adapta target
* CDAN: etc....



**PROBLEMAS 

**Invarianza excesiva (muy común)

Para confundir al discriminador:
* El extractor elimina señales reales que si pueden importar para Y

Resultado

* Buen alineamiento
* Mala predicción

**Confudir dominios != generalizar

Engañar al discriminador:

* No garantiza que el modelo funcione en target
* Solo garantiza que no distingue dominio


**Dependencia fuerte de hiperparametros 


**Opacidad total

* No se sabe que información elimino
* Que se preservo
* si se destruyo estructura fisica

![[Pasted image 20260125174710.png]]
