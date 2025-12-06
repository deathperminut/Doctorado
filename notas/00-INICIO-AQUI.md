# 👋 Bienvenido a tu Sistema de Notas Obsidian

Este es tu **vault** de Obsidian conectado a Git. Todo lo que escribas aquí se versionará automáticamente.

## 🗂️ Estructura de Carpetas

- **📚 literatura/** - Resúmenes y análisis de papers que leas
- **💡 ideas/** - Ideas de investigación y experimentos
- **👥 meetings/** - Notas de reuniones con tu advisor/colaboradores
- **📅 daily-notes/** - Diario diario de tu trabajo
- **📝 templates/** - Plantillas reutilizables (no las edites, úsalas para crear notas nuevas)

## 🚀 Cómo Usar las Plantillas

### Opción 1: Configurar Templates Plugin (Recomendado)

1. Ve a **Settings** (icono de engranaje, abajo izquierda)
2. En el menú izquierdo, busca **Core plugins**
3. Activa el plugin **"Templates"**
4. Vuelve a Settings → **Templates** (ahora aparecerá en Options)
5. En "Template folder location" pon: `templates`
6. En "Date format" pon: `YYYY-MM-DD`
7. Cierra Settings

Ahora puedes crear notas con plantillas:
- Crea una nota nueva
- Presiona `Cmd + P` (Command Palette)
- Escribe "template" y selecciona "Insert template"
- Elige la plantilla que necesites

### Opción 2: Copiar y Pegar

Simplemente abre una plantilla de `templates/` y copia el contenido a tu nueva nota.

## 📝 Plugins Recomendados para Activar

Ve a **Settings → Core plugins** y activa:

✅ **Daily notes** - Crea notas diarias automáticamente
✅ **Templates** - Usa las plantillas que creé
✅ **Graph view** - Visualiza conexiones entre notas
✅ **Backlinks** - Ve qué notas enlazan a la actual
✅ **Outgoing links** - Ve enlaces desde la nota actual
✅ **Tag pane** - Organiza por etiquetas
✅ **Quick switcher** - Navegación rápida con `Cmd + O`

## 🎯 Configurar Daily Notes

1. Settings → **Core plugins** → Activa "Daily notes"
2. Settings → **Daily notes**:
   - Date format: `YYYY-MM-DD`
   - New file location: `daily-notes`
   - Template file location: `templates/daily-note.md`

Ahora puedes crear una nota diaria con el ícono de calendario en el panel izquierdo.

## 🔗 Cómo Funcionan los Links

En Obsidian puedes conectar notas usando `[[nombre-de-nota]]`:

Ejemplo:
- Escribes: `Este concepto se relaciona con [[Machine Learning]]`
- Obsidian crea un link a la nota "Machine Learning"
- Si la nota no existe, se crea al hacer clic

Esto crea tu **segundo cerebro** con ideas conectadas.

## 🏷️ Uso de Tags

Usa tags para categorizar:
- `#literatura` - Papers
- `#idea` - Ideas
- `#experimento` - Experimentos
- `#meeting` - Reuniones
- `#por-leer` - Papers pendientes
- `#importante` - Notas críticas

## ⚡ Atajos de Teclado Útiles

- `Cmd + N` - Nueva nota
- `Cmd + O` - Abrir nota rápidamente
- `Cmd + P` - Command palette (todas las acciones)
- `Cmd + E` - Alternar entre preview/edición
- `[[` - Crear link a otra nota
- `Cmd + Click` - Abrir link en nueva pestaña

## 🎨 Ejemplo de Uso

### Para leer un paper:

1. Crea nota en `literatura/`
2. Usa template "literatura"
3. Llena la información del paper
4. Conecta con otras notas usando `[[links]]`

### Para una reunión:

1. Crea nota en `meetings/`
2. Usa template "meeting"
3. Documenta la reunión
4. Marca action items con `- [ ]`

### Para ideas:

1. Crea nota en `ideas/`
2. Usa template "idea"
3. Desarrolla la idea
4. Conecta con literatura relevante

## 💾 Git Integration

Todo lo que escribas aquí está en Git. Al final del día:

```bash
cd ~/Projects/Doctorado
git add notas/
git commit -m "Notas del día: [descripción]"
git push
```

O puedes instalar el plugin **Obsidian Git** (Community plugin) para que haga commits automáticos.

## 📚 Recursos

- [Obsidian Help](https://help.obsidian.md)
- [Community Plugins](https://obsidian.md/plugins)

---

**Próximos pasos:**
1. Activa los core plugins recomendados
2. Configura daily notes
3. Crea tu primera nota usando una plantilla
4. ¡Empieza a construir tu segundo cerebro!
