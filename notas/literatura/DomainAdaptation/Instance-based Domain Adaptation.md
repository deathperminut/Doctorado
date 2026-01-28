
**Idea central

"No todos los datos del source sirven igual para el target"
Algunos ejemplos del source se parecen más al target


**¿Qué hacen?

Asignar un peso a cada dato del source:

* Datos más "parecidos"  más peso
* Datos muy distintos  menos peso

Ejemplo sencillo

Supón:

* Source: Clientes de todo el país
* Target: Clientes Urbanos

Entonces:

* Clientes rurales -- Peso bajo
* Clientes urbanos -- Peso Alto

Entonces, en vez de:

* Cambiar los datos
* Cambiar el modelo
* Cambiar el espacio latente

Cambias la importancia de cada ejemplo.


Instance-bases DA se usa sobre todo cuando hay:
* Mismo fenómeno subyacente
* Mismo modelo de relación Y = f(x)
* Diferente Distribución de datos 

**Esto es Covariate shift.

**Matematicamente 


![[Pasted image 20260125171608.png]]

**De donde salen esos pesos (esto es lo importante)

**Idea teórica ideal

El peso correcto sería:


![[Pasted image 20260125171656.png]]

Interpretación sencilla:

* Si un tipo de datos aparece mucho en target y poco en source --- PESO ALTO
* Si aparece mucho en source y poco en target --- PESO BAJO

![[Pasted image 20260125171751.png]]


![[Pasted image 20260125171812.png]]


![[Pasted image 20260125171825.png]]



**QUE NO HACE INSTANCE-BASES DA

* No cambia la representación
* No aprende invariancia
* No corrige cambios en la relación Y l X


Asume que el modelo correcto ya existe.

**Cuándo funciona bien ( condiciones necesarias )

Instace-bases DA funciona SOLO si:

* Source y target se solapan
* Hay ejemplos soruce parecidos al target
* El ruido no domina
* La relación Y = f(x) es estable


**Porque falla ( y por qué falla feo)

* Caso 1: Poco solapamiento
Si el target vive en una región del espacio X

* No cubierta por source

*Caso 2: Ruido confundido con similitud

* Datos ruidosos parecen "distintos"
* Se les da peso bajo injustamente

Caso 3: Pesos extremos

* Dominan el entrenamiento
* Provocan overfitting

Caso 4: Cambio semántico real

![[Pasted image 20260125172317.png]]










