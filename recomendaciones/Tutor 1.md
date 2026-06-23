
**Luis Fernando Mulcue Nieto** : 
* **La clase de universalidad**: Al usar solo Heisenberg, el modelo de _Deep Learning_ solo aprende una física con simetría continua (p. 33). No sabemos si la red colapsará o fallará si se enfrenta a un sistema puramente discreto (Ising, simetría \(Z_{2}\)), donde los dominios no tienen paredes de dominio suaves ("domain walls"), sino fronteras abruptas.
* 
- 1. **Validación del modelo:** Para demostrar que el marco basado en Inteligencia Artificial realmente entiende la "física subyacente" (Physics-Informed), debería ser capaz de transicionar entre un modelo Heisenberg y un modelo Ising simplemente apagando o modificando componentes (por ejemplo, forzando interacciones uniaxiales extremas).

MIRAR ESTADOS CON TENDENCIA A COMPORTAMIENTO ISING , VARIANDO CON VALORES ALTOS LOS TEMAS DE ANISOTROPIA, PARA VER LA DIFERENCIA ENTRE AMBOS ESTADOS. 

ENTRE PAREDES SIMPLES Y CASOS MAS DRASTICOS 

-- **tocaria hablar con andres** para pensar que hacer en este aspecto 

pensaria en reducir el enfoque de la tesis a el campo de acción requerido 


# Tutor 2 

**DAVID AUGUSTO CÁRDENAS PEÑA**

MEJORAS:
##### GENERALES

1)  La propuesta tiene potencial para un trabajo doctoral. Sin embargo, gran parte del documento sufre de un enfoque de ensamble de herramientas **-- REVISAR CON PROFE**

2) En etapas sensibles, como la adaptación de dominios, se limita a ensamblar herramientas como MMD o CORAL tratadas externamente y desconectadas de las restricciones analíticas de la física magnética. Para elevar el rigor científico del anteproyecto, se recomienda reorientar la propuesta hacia nuevos operadores, métricas o modelos, en lugar de presentar el trabajo como una integración tecnológica correctamente validada. -- REVISAR EL TEMA DE ADAPTACIÓN DE DOMINIOS **--PENDIENTE**



3) Aunque la estructura del documento es apropiada, se evidencian dos falencias generales. Primero, la formulación teórica requiere mayor rigurosidad, ya que se presentan algunas variables que no están descritas, no tienen un espacio definido, o comparten notación.**-- READY**

4) Segundo, las imágenes presentan estilos distintos y su calidad es variable. En particular, la Figura 26 fue generada con Google Gemini y muestra una marca de agua. Se recomienda quitar la marca de agua y unificar el estilo de las figuras para mejorar la calidad del documento. -- esta la resolvemos ya con el desglose correcto de la nomenclatura. **--READY**

## OBJETIVO 1

2) La propuesta argumenta la necesidad de mapeos bidireccionales (cycle-consistent). Para tal fin, se define una función de pérdida de consistencia determinista, punto a punto, basada en la norma euclídea. ¿Si el generador directo G(p,z) produce una textura que físicamente es idéntica a la que produciría otro parámetro válido p', el estimador inverso F penalizaría con un error alto si predice p' en lugar de p? ¿Es posible que forzar la consistencia en regiones de alta degeneración obligue al regresor F a memorizar correlaciones artificiales para adivinar cuál fue el parámetro exacto de origen, en lugar de mapear el conjunto completo de soluciones físicas equivalentes? ¿como podriamos mitigar el error para las zonas donde se presenta alta divergencia de estructuras entre parametros? ¿ ya que tenemos una función de costo deterministica ? como evaluamos esas zonas de alta divergencia **-- REVISAR FLUJO CON EL PROFESOR**

3) La propuesta emplea transfer learning para estimar parámetros a partir de imágenes y aprovechar el conocimiento de modelos preexistentes. Estos modelos de imágenes existentes fueron preentrenados principalmente con imágenes RGB y escenarios/escenas muy diversos y bajo el principio de aprendizaje de texturas simples a la entrada y complejas a la salida. Por otra parte, las imágenes del documento (representativas de la tarea) muestran texturas bien definidas desde el inicio. Teniendo en cuenta estos dos hechos, presento dos observaciones: Primero, ¿por qué se conectan las imágenes en la entrada tradicional del modelo backbone forzando a reextraer características que ya pueden estar explícitas en la textura de las imágenes consideradas? Segundo, dado que las imágenes de microscopía viven en una “variedad” mucho más acotada que las imágenes tradicionales, ¿por qué no se consideran modelos de DL mucho más livianos y no preentrenados? Esta segunda observación se alinea con el problema de la biyectividad y la ill-possedness del “problema inverso” resultante. **-- REVISAR FLUJO CON EL PROFESOR**



4) En cuanto a la optimización, hay dos inquietudes. Primero, la función de pérdida total propuesta combina al menos cinco componentes no lineales de diferentes escalas y naturalezas. Este tipo de optimizaciones masivas es altamente propensa a colapsos de gradiente, a gradientes contradictorios (lo que impide la convergencia) o a la dominancia de un solo término (por ejemplo, la pérdida de reconstrucción puede canibalizar el término físico). Segundo, la propuesta plantea el uso de la Optimización Bayesiana (OB) para sintonizar los hiperparámetros de ponderación. Sin embargo, evaluar esta función objetivo de forma reiterada puede resultar computacionalmente prohibitivo. Además, no se debe emplear la OB como una herramienta plug-and-play. Se recomienda realizar un análisis más riguroso sobre el proceso de optimización y/o justificar formalmente la viabilidad computacional de la OB sobre la función de costo con la complejidad expuesta. Esto en aras de robustecer el aporte de este objetivo y que no se vea como un uso despreocupado de herramientas elaboradas. **--REVISAR CON EL PROFESOR**


5) Aunque en la Sección 10.3.2, se reconoce que métricas como el MSE evalúan mal las estructuras por falta de invariancia geométrica y se proponen un trabajo futuro para mitigar esta falla, no queda claro si dicha corrección hace parte del trabajo doctoral y se desarrollará durante la candidatura o, por el contrario, se sale del scope del trabajo doctoral. En el primer caso, ¿es posible introducir bloques de registro rígido en el costo de reconstrucción? -- ESTUDIAR

* INCLUIR EN EL LOSS PARTE DE LAS METRICAS EVALUADAS A NIVEL DE FISICA
   ( MAGNETIZACIÓN, ABS. MAGNETIZACIÓN Y  DE PRONTO **Peak wavevector** EN LA FUNCIÓN DE COSTO CON UN PARAMETRO DE REGULARIZACIÓN PEQUEÑO)
* USAR SPIN CORRELATION FUNCTION YA SE USARIA PARA EVALUAR POSTERIORMENTE

OJO SON MEDIDAS DE RELACIÓN GLOBAL PARA ANALIZARLAS LOCALMENTE PODRIAMOS HACER UNA VENTANA DESLIZANTE , mas otras metricas que me acompañen 





2) En las pruebas de concepto iniciales presentadas (Figura 33), se observa que las predicciones del parámetro físico KDM muestran una dispersión y una degradación masiva de la precisión a temperaturas intermedias y altas, debido a la pérdida del orden magnético inducida por el caos térmico cerca de la temperatura de Curie. Sin embargo, la metodología no propone ninguna estrategia (atención, regularización, filtrado, etc.) que aísle o mitigue esta debilidad predictiva en regímenes de alta temperatura. **-- ESTUDIAR -- DISCUTIR CON EL PROFESOR**

![[Pasted image 20260621120456.png]]
![[Pasted image 20260623082633.png]]
![[Pasted image 20260623082730.png]]

![[Pasted image 20260623082805.png]]

![[Pasted image 20260623082858.png]]

![[Pasted image 20260623082918.png]]

![[Pasted image 20260623083047.png]]


# ADAPTACIÓN DE DOMINIO 

1) El documento se enfoca casi exclusivamente en la Adaptación de Dominios (AD)
estadística no supervisada (como MMD o CORAL), abordando la brecha como un
problema puramente estadístico de caja negra. Luego de una revisión propia del estado del arte, se encuentran otras alternativas a la AD, por ejemplo, Physics-Based Forward Modeling, Domain Randomization, Physics-Guided Image-to-Image Translation, Contrastive Self-Supervised Learning. Se recomienda fortalecer el estado del arte para garantizar una selección más rigurosa de la alternativa de solución. -- PROFUNDIZAR EN EL ESTADO DEL ARTE

2) La sección 7.1.2, Simulation Procedure, propone un barrido sistemático de los parámetros para simular las configuraciones/estados de espín. Sin embargo, no se profundiza en el procedimiento de barrido sistemático. Bajo la suposición de un barrido uniforme, la distribución de parámetros del Hamiltoniano estará perfectamente balanceada. Sin embargo, la misma propuesta señala que, en los materiales reales, los parámetros están definidos por la composición química y pueden concentrarse en regiones estrechas de interés práctico (por ejemplo, en texturas estables de skyrmiones). Esta diferencia en las distribuciones puede afectar negativamente el entrenamiento del modelo. -- REVISAR SI YA LO ESTAMOS REALIZANDO 

3) El documento detalla con precisión las métricas de error para los datos sintéticos (Sección 10.2), pero deja en el aire cómo se medirá cuantitativamente el éxito del regresor frente a una imagen de MFM real. Esto es importante porque las imágenes de microscopía real (MFM/STM) de los datasets públicos no vienen acompañadas de los valores numéricos exactos de los parámetros del Hamiltoniano que las originaron, que precisamente constituyen las incógnitas que se desean descubrir. Al depender sólo de una validación cualitativa, el trabajo no puede garantizar que una similitud visual corresponda a una precisión paramétrica debido al problema de la degeneración, ya declarado en la propuesta.   -- REVISAR 



## OBJETIVO ESPECIFICO 3

* UMAP es un excelente algoritmo de reducción de dimensionalidad no lineal para la exploración visual, que opera bajo el principio de preservar las estructuras locales a expensas de distorsionar las distancias absolutas a gran escala. Otras técnicas de reducción de dimensión para efectos de visualización (por ejemplo, tSNE) tienen sus pros y contras. En la propuesta no se realiza un análisis objetivo del algoritmo de reducción de dimensiones, lo que hace que la selección resulte arbitraria.


* Teniendo en cuenta que cada objetivo debe ser validable, la propuesta carece de una métrica cuantitativa agregada para evaluar la fidelidad de las explicaciones. Al final, el éxito del objetivo específico 3 queda supeditado a una inspección cualitativa o subjetiva por parte de expertos (en el mejor de los casos) o del propio candidato (con un posible sesgo de confirmación). Se recomienda explorar métricas cuantitativas de fidelidad explicativa para respaldar la validación objetiva de los resultados. 

* Esta observación está relacionada con la naturaleza post-hoc del análisis de explicabilidad y la información física que el modelo puede explotar. Considerando que la propuesta se apoya firmemente en arquitecturas complejas de aprendizaje profundo (como DenseNet121 y cVAEs) para resolver la alta dimensionalidad y la degeneración intrínseca de las texturas de espín, ¿cuáles son los impedimentos metodológicos y físicos fundamentales que restringen o impiden el desarrollo de técnicas de explicabilidad ad-hoc o intrínsecas dentro de este anteproyecto doctoral?



* Como continuación de la observación anterior, ¿por qué no se puede considerar una técnica que dé explicabilidad inherente a la estimación de los parámetros del Hamiltoniano? Hago esta pregunta porque dentro de Physics Informed Deep Learning (PIDL) hay tres grandes enfoques: i) Usar restricciones físicas como términos de la función de costo, un wrapper a la regularización. Esto se ve claramente en el objetivo específico 1. ii) Usar el conocimiento físico para generar datos sintéticos a partir de procedimientos altamente costosos (por ejemplo, mediante la solución de un gran número de problemas de optimización no lineales). Este tipo de PIDL se explota en el objetivo específico 2. iii) Finalmente, en lugar de “rogarle” al optimizador que respete la física mediante la función de pérdida, este enfoque modifica el diseño interno de la red neuronal para que sea matemáticamente imposible violar una ley física. Considero que este enfoque tendría, como ventaja adicional, la explicabilidad intrínseca del modelo desarrollado.




## PAULO CESAR 

### APARTADO DE ESTADO DEL ARTE 

* La noción de degeneración. En el contexto es claro que la idea de degeneración
obedece a que una combinación de parámetros que generan patrones visualmente
indistinguibles, sin embargo, la noción en Física puede ser aún más precisa. Ade-
más, una lectura más profunda de esta noción, quizás puede aportar en el diseño
de la arquitectura de la red soportada con Física (desde luego, esto es una mera
hipótesis).


* En el texto se presenta como un ejemplo la dinámica del vector de magnetización
según la ecuación de Landau-Lifshitz-Gilbert, y se incorpora una función de costo
basada en el residual de dicha ecuación; sin embargo, en el caso que se quiere
resolver en esta propuesta se busca abordar el problema a partir del Hamiltoniano
(microscópico) y sus correspondientes estados estacionarios (via Monte Carlo o
Redes Neuronales). Considerando lo anterior, me pregunto si las redes neuronales
que buscan diseñar obedecen a estados estacionarios y dinámicos (series de tiem-
po), este punto merece una aclaración. Por otro lado, la función de costo para el
caso Hamiltoniano no me queda clara, es un residual o qué es, en el texto indi-
can que es una penalización, a partir de lo cual quiero preguntar si tienen algún
intervalo para los posibles valores de los parámetros del Hamiltoniano.





* En el documento se menciona que la temperatura es un parámetro de un Hamilto-niano (ver Figura 3), sin embargo, la temperatura no entra en la descripción del Hamiltoniano (por razones físicas), ver ecuación (8) del capítulo 7. En ese sentido, vale la pena que exploren cómo se relaciona la temperatura con la simulación y qué implicación tiene en el diseño de la red neuronal.
* 
* Frente a la novedad del trabajo, toda la estructura de las redes neuronales que se proponen es bastante robusta y seguramente puede tener diferentes aplicaciones, sin embargo, considero importante que se estime realmente la complejidad computacional de la red comparada con la simulación Monte-Carlo.
* Frente al diseño de la red tengo dos comentarios. El primero tiene que ver con la
especificidad de la propuesta desde la perspectiva del diseño de las redes neuronales, si bien el lenguaje es apropiado para el nivel, para una persona no experta en el área como yo, es fácil de perderse en la lectura, por lo cual solicito que el documento sea evaluado por alguien experto en el área. En segundo lugar, me quedo con preguntas del tipo, por ejemplo, en la función de costo que se compone de varios términos, por qué debe ser tan estructurada, y si efectivamente cada término aporta en algo en la solución, y cómo se puede rastrear este efecto en caso de que sea posible.


el fenomeno de montecarlo - al tratar de disminuir en ferromagnetico, ambas tienen la misma probabilidad de existir



















