el problema de fondo: ¿Por qué falla el ML en el mundo real?

En machine learning clásico hacemos una suposición fuerte:

* Los datos de entrenamiento y los datos de prueba vienen de la misma distribución

Formalmente:

   P_train (X,Y) = P_test(X,Y)


Pero en la practica... casi nunca pasa.


Ejemplos reales

Entrenas con:

* imágenes simuladas , pruebas con imágenes experimentales
* datos médicos de un hospital , despliegas en otro hospital
* fotos diurnas , inferencia nocturna
* clientes históricos , clientes nuevos post-cambio regulatorio

Ese quiebre se llama :

  * **Domain Shift / Dataset shift**
Y aquí nace Domain Adaptation (DA).

2) Domain adaptation es el conjunto de métodos que permiten que un modelo entrenado en un dominio fuente funcione bien en un dominio objetivo, aunque las distribuciones de datos sean distintas.

![[Pasted image 20260125163647.png]]
El objetivo de **DA**  es aprender un modelo que minimice el error en el dominio objetivo, usando información del dominio fuente.

3) ¿Por qué es necesario la adaptación de dominio?
- Etiquetar datos es caro 
- Datos reales != datos reales (Ruido,sensores,condiciones no controladas)
- Modelos frágiles ( Las CNN aprenden correlaciones espurias del dominio fuente)



Sin adaptación

- overfitting al dominio fuente
- Caida brutal de performance en deployment
- Modelos no confiables

**Tipos de Domain Adaptation ( Clasificación clave )**

1. Según disponibilidad de etiquetas 

![[Pasted image 20260125164126.png]]
2. Según qué cambia entre dominios

a) Covariate Shift 

![[Pasted image 20260125164236.png]]
b) Label shift
![[Pasted image 20260125164300.png]]
![[Pasted image 20260125164316.png]]

**¿Qué hace realmente Domain Adaptation?**

En esencia, DA intenta reducir una distancia entre dominios.

Intuición clave 

"Si hago que las representaciones internas del modelo sean similares para source y target, el clasificador generaliza."

Matemáticamente:

![[Pasted image 20260125164532.png]]
NOTA : PROFUNDIZAR EN ESTAS DISTANCIAS


**"DOMAIN ADAPTATION : Busca aprender representaciones que ignoren diferencias irrelevantes entre dominios y conserven la estructura necesaria para resolver la tarea."**






