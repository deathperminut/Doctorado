
**Idea Central (la intuición que manda todo)

El modelo ya sabe algo útil, solo hay que ajustarlo para el nuevo dominio.

A diferencia de otros enfoques:

* No cambias los datos
* No fuerzas alineamientos latentes
* No usar discriminadores

Cambiar el propio modelo


**Qué significa "adaptar el modelo"

Adaptar el modelo puede significar:

* Ajustar algunos parámetros
* Ajustar ciertas capas
* Ajustar estadísticas internas
* Ajustar qué partes del modelo se mueven

El dominio se "absorbe" en los pesos.

**Cuándo aparece este enfoque (motivación real)

Model-bases DA aparece cuando:

* Tienes un modelo pre-entrenado fuerte
* El dominio target tiene pocos datos -- ME PASA
* No puedes permitir modelos complejos -- YA ESTA LO SUFICIENTEMENTE COMPLEJO
* Quieres control y estabilidad



**Fine-Tuning Clásico

Entrenas un modelo en source y luego

* Continuar entrenando con datos target

Overfitting fácil

b) Fine-tuning parcial

* congelas capas tempranas
* ajustar capas finales
adaptación limitada


**Tipo 2: Adapatación Batch Normalization (BN)

Las capas BN guardan:
* Media
* Varianza

Si cambias de dominio
Estas estadisticas ya no sirven
**Qué se hace
* Recalcular estadisticas BN con datos target
* Sin tocar los pesos

Solo corrige cambios estadisticos
No sirve si el dominio cambia semanticamente


**Multi-head /Domain-specific heads

Compartes Backbone

Pero cada dominio tiene

* Su propio head

Ventajas
* Preserva especificidad
* Evita borrar información

Problemas

* Necesita saber el dominio en inferencia 
* Escala mal a muchos dominios



**Cuando es peligroso usarlo

Cuando el target es muy pequeño
cuando el dominio cambia fuerte
Cuando necesitas explicabilidad global


