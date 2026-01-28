"Los datos pueden ser distintos, pero las características importantes no deberían serlo"


En lugar de usar los datos crudos, se busca:

* Un espacio de representación común.


**¿Que hacen?

* Extraen features
* alinean las distribuciones de esas features

**Ejemplo 

Dos imagenes

* Una simulada
* Una real

Se ven distintas, pero:

* Bordes
* Formas
* Patrones

deberían estar en el mismo espacio latente

**Métodos más conocidos

**A.Coral (Correlation Alignment)

Alinea:
* Media
* Varianza
* Correlaciones

Muy simple, Muy usado como baseline

**B. MMD (Maximum Mean Discrepancy)**

Mide:
* Qué tan diferentes son 2 distribuciones

Minimiza esa diferencia en el espacio latente.

No necesita GANs
Computacionalmente pesado


**Cuando usar 

* Diferencias estadísticas claras
* Features interpretables
* Como primer enfoque serio








