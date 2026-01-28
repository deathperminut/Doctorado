
"Si ambos dominios se pueden reconstruir desde un mismo espacio latente, entonces comparten estructura"


**¿Que hacen?

* Codifican source y target
* Comparten o alinean el espacio latente
* Reconstruyen cada dominio


Los autoencoders se usan para:

* Encontrar un espacio latente común
* Separar estructura de estilo/ruido/dominio

En DA, el latente es el verdadero protagonista.


**TIPO 1 : Autoencoder Compartido (Shared Encoder)

Un solo encoder procesa:
* Source 
* Target

Pero cada dominio puede tener:

* Su propio decoder

![[Pasted image 20260125175301.png]]

Se fuerza a que Z deba servir para ambos dominios

**Ventajas

* Simplicidad
* Latente alineado automáticamente

**Problemas
* Puede colapsar a representar solo lo común
* Ignora información especifica de dominio


**Tipo 2: Autoencoders Paralelos con Regularización

**Idea

Dos autoencoders
* Uno para source 
* Uno para target

Se agrega una pérdida para:

Alinear los latentes
![[Pasted image 20260125175921.png]]



**Donde falla estos métodos

**1. el latente aprende estilo, no fisica

Muy común. 

El AE es excelente reconstruyendo

* Ruido
* Textura
* Artefactos

Pero eso no ayuda a Y

**2. Colapso del espacio latente

Z se vuelve:
* Demasiado pequeño
* Poco informativo

