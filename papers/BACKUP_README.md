# Sistema de Backup Automático a Google Drive

Scripts para hacer backup automático de PDFs de papers a Google Drive con versionamiento.

## 📋 Requisitos

### 1. Instalar rclone

```bash
brew install rclone
```

### 2. Configurar Google Drive

```bash
rclone config
```

Pasos en el wizard:
1. Presiona `n` → New remote
2. Nombre: `gdrive`
3. Storage: `drive` (Google Drive)
4. Client ID/Secret: **Enter** (dejar en blanco)
5. Scope: `1` (acceso completo)
6. Service Account: `n`
7. Edit advanced config: `n`
8. Auto config: `y` → Se abrirá el navegador
9. Inicia sesión en tu cuenta de Google
10. Configure as team drive: `n`
11. Confirma: `y`
12. Quit: `q`

### 3. Verificar configuración

```bash
rclone listremotes
```

Deberías ver: `gdrive:`

## 🚀 Uso

### Hacer backup de un paper específico

```bash
# Desde la carpeta papers/
./backup_pdf_to_gdrive.sh PreliminaryDraft
./backup_pdf_to_gdrive.sh EstimationPaper1
./backup_pdf_to_gdrive.sh OptimizationData
```

### Listar backups existentes

```bash
# Ver todos los backups
./list_backups.sh

# Ver backups de un paper específico
./list_backups.sh PreliminaryDraft
```

## 📁 Estructura en Google Drive

Los backups se guardan en:
```
Google Drive/
└── Doctorado/
    └── PDFs/
        ├── PreliminaryDraft/
        │   ├── main.pdf (versión actual)
        │   ├── main_20260126_120530.pdf
        │   ├── main_20260126_143022.pdf
        │   └── ...
        ├── EstimationPaper1/
        │   └── ...
        └── OptimizationData/
            └── ...
```

## ⚙️ Funcionamiento

El script:
1. ✅ Verifica que el PDF existe
2. ✅ Sube una versión con timestamp (ej: `main_20260126_120530.pdf`)
3. ✅ Actualiza la versión actual (`main.pdf`)
4. ✅ Mantiene historial completo de versiones
5. ✅ Muestra los últimos 5 backups

## 🔄 Automatización (Opcional)

### Opción 1: Después de cada compilación

Crea un alias en tu `~/.zshrc` o `~/.bashrc`:

```bash
alias compile-paper='cd /Users/juansebastianmendezrondon/Projects/Doctorado/papers/PreliminaryDraft && pdflatex main.tex && pdflatex main.tex && cd .. && ./backup_pdf_to_gdrive.sh PreliminaryDraft'
```

Uso: `compile-paper`

### Opción 2: Backup programado (cron)

Backup diario a las 6 PM:

```bash
# Editar crontab
crontab -e

# Agregar línea:
0 18 * * * cd /Users/juansebastianmendezrondon/Projects/Doctorado/papers && ./backup_pdf_to_gdrive.sh PreliminaryDraft >> /tmp/backup.log 2>&1
```

### Opción 3: Git hook (después de commit)

Crea `.git/hooks/post-commit` en tu repo:

```bash
#!/bin/bash
cd /Users/juansebastianmendezrondon/Projects/Doctorado/papers
./backup_pdf_to_gdrive.sh PreliminaryDraft
```

Dale permisos: `chmod +x .git/hooks/post-commit`

## 🛠️ Comandos útiles de rclone

```bash
# Listar archivos en Google Drive
rclone ls gdrive:Doctorado/PDFs/PreliminaryDraft/

# Descargar un backup específico
rclone copy gdrive:Doctorado/PDFs/PreliminaryDraft/main_20260126_120530.pdf ~/Downloads/

# Ver espacio usado
rclone about gdrive:

# Sincronizar carpeta completa
rclone sync PreliminaryDraft/ gdrive:Doctorado/PDFs/PreliminaryDraft/
```

## ⚠️ Notas Importantes

- El script **NO** elimina backups antiguos automáticamente
- Cada backup ocupa ~52MB (tamaño del PDF)
- Google Drive gratis tiene 15GB de espacio
- Puedes eliminar backups viejos manualmente desde Google Drive o con:
  ```bash
  rclone delete gdrive:Doctorado/PDFs/PreliminaryDraft/main_20260101_000000.pdf
  ```

## 🔍 Solución de Problemas

### Error: "rclone not found"
```bash
brew install rclone
```

### Error: "Remote 'gdrive' not configured"
```bash
rclone config
# Sigue los pasos de configuración arriba
```

### Error: "PDF not found"
```bash
# Compila el documento primero
cd PreliminaryDraft/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📊 Ejemplo de Uso Completo

```bash
# 1. Hacer cambios al LaTeX
cd PreliminaryDraft/
vim main.tex

# 2. Compilar
pdflatex main.tex && pdflatex main.tex

# 3. Hacer backup
cd ..
./backup_pdf_to_gdrive.sh PreliminaryDraft

# 4. Ver backups
./list_backups.sh PreliminaryDraft
```

## 📝 Papers Soportados

- ✅ `PreliminaryDraft` - Anteproyecto doctoral
- ✅ `EstimationPaper1` - Paper de estimación
- ✅ `OptimizationData` - Paper de optimización

Para agregar más papers, solo edita la variable en el script o pasa el nombre de la carpeta como parámetro.
