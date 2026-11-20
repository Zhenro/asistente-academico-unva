# Asistente Virtual Inteligente de Consultas Académicas (UNVA)

Sistema integral de atención, orientación y resolución de consultas académicas e institucionales basado en **Generación Aumentada por Recuperación (RAG)** y modelos multimodales de última generación, diseñado para la comunidad estudiantil y secretaría de la **Universidad Nacional del Valle Alto (UNVA)**.

---

## 🌟 Características Principales del Sistema

### 1. 🎙️ Asistencia Multimodal por Telegram (`@UNDVABOT`)
* Soporte para consultas en **texto** y **notas de voz** estudiantiles.
* Transcripción fonética inteligente con **Google Gemini Multimodal** y glosario institucional (*Cashea*, *TernaNet*, *UNVA*, *PI*, materias y sedes).
* Despacho asíncrono con respuesta inmediata para evitar tiempos de espera o cancelaciones de conexión.

### 2. 🎓 Perfil Estudiantil Personalizado (Integración TernaNet)
* Vinculación de expedientes académicos mediante Cédula de Identidad (`/estudiante <cédula>` o mediante el botón web).
* Cruce automático de normativas institucionales con datos reales del estudiante:
  * Porcentaje de créditos acumulados vs. requisitos de **Servicio Comunitario** (50% de UC) y **Pasantías**.
  * Estado de cuotas de financiamiento con **Cashea** y solvencias de biblioteca, caja y control de estudios.
  * Promedio acumulado, materias en curso y estatus de graduación.
* Incluye 7 perfiles de prueba con casos académicos variados:
  * **José Andrés Fermín**
  * **Diego Torres**
  * **Maryander Torres**
  * **Ismael Pernía**
  * **Anibal Figuera**
  * **Félix Ruiz**
  * **Paola Rada**

### 3. 🔐 Panel Administrativo de Secretaría (`/admin`)
* Acceso protegido por contraseña institucional (`unva2026`).
* **Gestor de Documentos PDF:** Carga de nuevos reglamentos vía arrastrar y soltar (drag & drop), eliminación y visualización.
* **Reindexación en Caliente (Hot-Reindex):** Actualización instantánea de la base de conocimiento vectorial sin reiniciar el servidor ni interrumpir a los usuarios.
* **Padrón de Expedientes:** Inspección visual de los estudiantes registrados en TernaNet.

### 4. 📊 Analítica Institucional y Detección de Vacíos Normativos
* Registro de consultas por canal (Web vs. Telegram).
* Sistema de evaluación estudiantil con votos de utilidad (**👍 / 👎**).
* **Detección Automática de Vacíos Normativos:** Alerta a las autoridades académicas sobre preguntas no cubiertas en los reglamentos actuales.
* Exportación de reportes completos a formato **CSV** para auditorías académicas.

---

## 🎯 Alineación con los Objetivos del Proyecto Universitario

* **Objetivo 1 (Diagnóstico):** Identificación de inquietudes y trámites frecuentes (admisión, TernaNet, evaluación continua, inasistencias, servicio comunitario, pasantías y grado).
* **Objetivo 2 (Datos y Arquitectura):** Indexación vectorial y recuperación semántica de alta eficiencia a partir de documentos institucionales oficiales en PDF.
* **Objetivo 3 (Desarrollo e Integración):** Backend asíncrono con **FastAPI**, interfaz web moderna responsiva y webhook de Telegram en la nube.
* **Objetivo 4 (Evaluación y Validación):** Trazabilidad con citación de fuentes oficiales, analítica con votos 👍/👎 y pruebas con expedientes de estudiantes reales.

---

## 🏛️ Documentación Oficial Incorporada (`data/documents/`)

1. **`01_Guia_Admisiones_Inscripcion_UNVA.pdf`**:
   * Requisitos de nuevo ingreso en carpeta marrón (título de bachiller, notas, fotos, cédula, OPSU).
   * Procedimiento de 5 pasos en plataforma TernaNet.
   * Sedes: Guatire (C.C. Daymar), Caracas (El Paraíso), Puerto La Cruz, Barcelona y El Tigre.
   * Pagos por transferencia y cuotas con **Cashea**. Turnos diurno y sabatino.
2. **`02_Reglamento_Evaluacion_Estudiantil_UNVA.pdf`**:
   * Evaluación continua (3 cortes: 30%, 30%, 40%).
   * Escala de 1 a 20 puntos, nota mínima aprobatoria: 10 puntos (redondeo de 0.5 solo en nota final).
   * Límite de inasistencias del 25% (Pérdida por Inasistencia - PI).
   * Normativa de prelaciones y retiro de asignaturas.
3. **`03_Oferta_Academica_Carreras_UNVA.pdf`**:
   * Programas TSU: Informática, Diseño Gráfico, Administración, Contaduría, Relaciones Industriales, Turismo, Fisioterapia y Enfermería.
   * Prosecución a Licenciaturas e Ingenierías.
4. **`04_Normativa_Servicio_Comunitario_Pasantias_Grado_UNVA.pdf`**:
   * Servicio Comunitario obligatorio (120 horas académicas, requisito de 50% de UC aprobadas).
   * Pasantías y Acreditación de Experiencia Laboral técnica.
   * Requisitos para acto de grado (solvencias de Control de Estudios, Biblioteca, Caja y fondo negro).
5. **`05_Calendario_Academico_Periodos_UNVA.pdf`**:
   * Estructura de los 3 períodos anuales (I, II y III).
   * Cronograma de inscripciones, evaluaciones, entregas de solvencias y grados.

---

## 🚀 Inicio Rápido Local

### 1. Iniciar el Servidor
```bash
python run.py
```
Abre en tu navegador:
👉 **`http://127.0.0.1:8000`** (Asistente Web)
👉 **`http://127.0.0.1:8000/admin`** (Panel Administrativo - Clave: `unva2026`)

### 2. Configurar Claves de Entorno (`.env`)
```env
GEMINI_API_KEY=tu_clave_de_google_aistudio
GEMINI_MODEL=gemini-3.1-flash-lite
TELEGRAM_BOT_TOKEN=tu_token_de_botfather
TELEGRAM_BOT_USERNAME=UNDVABOT
ADMIN_PASSWORD=unva2026
```

---

## 📁 Estructura del Proyecto

```text
CIT/
├── .github/workflows/
│   ├── ci.yml              # Integración continua (21 pruebas automatizadas)
│   └── keep_alive.yml      # Ping periódico para mantener el servicio activo
├── data/
│   ├── documents/          # PDFs normativos e institucionales oficiales (UNVA)
│   ├── students.json       # Base de datos simulada de expedientes TernaNet (7 perfiles)
│   └── analytics.json      # Almacén de métricas, vacíos normativos y feedback (UTC-4)
├── frontend/
│   ├── index.html          # Interfaz web del estudiante (Modo General + TernaNet, SSE, feedback)
│   └── admin.html          # Panel de administración, analítica y expedientes
├── scripts/
│   ├── generate_documents.py # Generador de PDFs institucionales oficiales (UNVA)
│   └── telegram_bot.py     # Bot de Telegram en modo polling local
├── src/
│   ├── config.py           # Variables de entorno y configuración general
│   ├── ingestion.py        # Procesamiento, chunking y extracción de texto de PDFs
│   ├── rag_engine.py       # Motor RAG BM25 ultraligero (<35MB RAM), SSE async y audio
│   ├── ternanet.py         # Conector, detección de cédulas y formateo de expedientes
│   ├── analytics.py        # Motor de métricas, votaciones 👍/👎 y exportación CSV segura
│   └── api.py              # Endpoints FastAPI, rate limiting y webhook de Telegram
├── tests/
│   └── test_project.py     # Suite de pruebas unitarias y de integración (21 tests)
├── requirements.txt        # Dependencias de Python
├── render.yaml             # Configuración de despliegue en Render
├── run.py                  # Script de arranque local
└── README.md
```
