# Sistema de Notas - Obsidian Vault

Este directorio es el **vault de Obsidian** para gestionar todo el conocimiento del doctorado.

## Estructura

```
notas/
├── literatura/     # Resúmenes y anotaciones de papers
├── ideas/          # Ideas de investigación y brainstorming
├── meetings/       # Notas de reuniones con advisor/colaboradores
└── daily-notes/    # Diario de investigación
```

## Configuración de Obsidian

1. Descargar Obsidian: https://obsidian.md
2. Open vault → Open folder as vault
3. Seleccionar: `/Users/juansebastianmendezrondon/Projects/Doctorado/notas`

## Plugins Recomendados

### Core Plugins (incluidos)
- Daily notes
- Templates
- Graph view
- Backlinks

### Community Plugins
- **Dataview**: Queries sobre tus notas
- **Calendar**: Vista de calendario para daily notes
- **Admonition**: Bloques de advertencia/información
- **Obsidian Git**: Auto-commit (opcional)

## Plantillas

### Template: Literatura
```markdown
---
paper: "Título del Paper"
authors: "Autor et al."
year: 2024
tags: [literatura, machine-learning]
---

# [Título del Paper]

## Resumen
¿Qué problema resuelve?

## Metodología
¿Cómo lo resuelven?

## Resultados Clave
- Resultado 1
- Resultado 2

## Ideas para mi Investigación
- Idea 1
- Idea 2

## Citaciones
```

### Template: Meetings
```markdown
---
date: {{date}}
attendees: [Advisor, Yo]
tags: [meeting]
---

# Meeting {{date}}

## Agenda
- [ ] Tema 1
- [ ] Tema 2

## Discusión

## Action Items
- [ ] Tarea 1
- [ ] Tarea 2

## Próxima Reunión
```

## Workflow

### 1. Daily Notes
Cada día, documenta:
- Qué trabajaste
- Papers que leíste
- Problemas encontrados
- Ideas nuevas

### 2. Literature Notes
Al leer un paper:
1. Crear nota en `literatura/`
2. Usar template de literatura
3. Vincular con conceptos relacionados usando [[wikilinks]]

### 3. Ideas
Captura ideas inmediatamente en `ideas/`
- Conectar con literatura relevante
- Esbozar experimentos potenciales

## Ventajas del Sistema

- ✅ Todo en Markdown = versionado con Git
- ✅ Búsqueda instantánea
- ✅ Gráfico de conocimiento visual
- ✅ Links bidireccionales entre notas
- ✅ Funciona offline

---

**Tip**: Usa tags consistentes (#metodologia, #experimento, #literatura) para organizar mejor.
