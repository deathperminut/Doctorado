Si el problema es que los datos son distintos... hagamos que se parezcan

En lugar de cambiar el modelo , cambias los datos

¿ Qué hacen exactamente ?
* Transforman los datos del dominio fuente
* O "traducen" datos del source para que se vean como target


**Métodos típicos**

A. Data Augmentation dirigido al dominio

Ejemplo:

* Agregar ruido
* Cambiar contraste
* Blur
* Distorsiones

Muy Común en visión
Simple 


B. Image-to-Image Translation 

Usan modelos generativos:

* CycleGAN
* Pix2Pix
* Style Transfer

Ejemplo caro 

* Imagen simulada a Versión realista
* Imagen limpia a versión con ruido experimental

El modelo se entrena con datos transformados 

**USAR CUANDO LAS DIFERENCIAS SON VISUALES , CUANDO EL DOMINIO TARGET ES INACCESIBLE PARA ETIQUETAR , SIM - TO - REAL**   (( EN MI CASO INTRODUCIR UNA GAN PARA ADAPTARLAS A LOS DOMINIOS TARGET))




**PROBLEMAS REALES**

**Introducción de artefactos no físicos
* GANS o transformaciones pueden crear patrones que no existen
* El modelo aprende texturas falsas, bordes artificiales, ruido estructurado inexistente

Grave en ciencia y física: el modelo aprende estilo, no fenómeno.

**Pérdida de trazabilidad

 Después no sabes:
 * Si el error viene del modelo
 * O de la transformación de datos

se rompe la cadena dato -- fenómeno -- predicción


**No Escala bien

Cada nuevo dominio implica un generador nuevo o ajustado
alto costo computacional


**Cuando no usarla 
- Cuando el dominio target reprenta un fenómeno distinto
- Cuando necesitas interpretabilidad física
- Cuando el generador no esta validado fisicamente
