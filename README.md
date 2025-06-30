# Estudio Comparativo de Pequeños Modelos de Lenguaje Afinados (SLMs) y Modelos Grandes de Lenguaje (LLMs) para la Extracción de Información

## Descripción del Proyecto

Este repositorio contiene el código fuente, notebooks y datos para la tesis de maestría que realiza un estudio comparativo entre Small Language Models (SLMs) afinados y Large Language Models (LLMs) para tareas de extracción de información en documentos multimodales.

## 📊 Dataset

El proyecto utiliza el dataset **SP-DocVQA (Single Page Document Visual Question Answering)**, que contiene:
- Documentos de una sola página con texto e imágenes
- Preguntas sobre el contenido de los documentos
- Respuestas esperadas para evaluación
- Datos OCR extraídos de los documentos

## 🏗️ Estructura del Proyecto

```
# Notebooks principales de análisis
├── dataset_creation.ipynb     # Creación y preprocesamiento del dataset
├── EDA.ipynb                 # Análisis exploratorio de datos
├── openai-4-mini.ipynb       # Experimentos con GPT-4 Mini
├── claude-3.5-sonnet.ipynb   # Experimentos con Claude 3.5 Sonnet
├── Llama3.2_(11B)-Vision.ipynb # Experimentos con Llama 3.2 Vision
├── results.ipynb             # Análisis de resultados y métricas
├── data/                         # Datos procesados
│   ├── df_concat_with_text_and_image_tokens.pkl
│   ├── train/, val/, test/       # Divisiones del dataset
├── spdocvqa_qas/                 # Datos de preguntas y respuestas
├── spdocvqa_ocr/                 # Datos OCR extraídos
├── images/                       # Imágenes del dataset
├── results/                      # Resultados de experimentos por modelo
├── final_results_test/           # Resultados finales en conjunto de prueba
├── models/                       # Definiciones de modelos de base de datos
├── utils/                        # Utilidades y funciones auxiliares
├── core/                         # Configuraciones centrales
├── latex/                        # Archivos LaTeX para tablas de métricas
└── requirements.txt              # Dependencias del proyecto
```

## 🤖 Modelos Evaluados

### Large Language Models (LLMs)
- **GPT-4.1 Mini**: Versión más eficiente de GPT-4
- **GPT-4.1 Nano**: Versión ultraligera de GPT-4
- **Claude 3.5 Sonnet**: Modelo de Anthropic

### Small Language Models (SLMs)
- **Llama 3.2 (11B) Vision**: Modelo multimodal de Meta
  - Configuraciones evaluadas:
    - Solo texto
    - Solo imagen
    - Multimodal (texto + imagen)
  - Estados de entrenamiento:
    - Modelo base (epoch 0)
    - Fine-tuned epoch 1
    - Fine-tuned epoch 2

## 📈 Métricas de Evaluación

- **ANLS (Average Normalized Levenshtein Similarity)**: Métrica principal para VQA
- **WER (Word Error Rate)**: Tasa de error a nivel de palabra
- **CER (Character Error Rate)**: Tasa de error a nivel de carácter
- **Parsing Error Rate**: Tasa de errores de análisis de respuestas

## 🔧 Configuración del Entorno

### Requisitos Previos
- Python 3.8+
- CUDA (para entrenamiento con GPU)
- Docker y Docker Compose (para base de datos)

### Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd TESIS/Notebooks
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar base de datos**
```bash
docker-compose up -d
```

5. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env con tus claves de API
```

## 🚀 Uso

### 1. Preparación de Datos
```bash
jupyter notebook dataset_creation.ipynb
```
Este notebook:
- Carga los datos de SP-DocVQA
- Extrae texto OCR de las imágenes
- Calcula tokens para LLMs
- Preprocesa imágenes para modelos multimodales

### 2. Análisis Exploratorio
```bash
jupyter notebook EDA.ipynb
```
Incluye:
- Distribución de tipos de preguntas
- Análisis de complejidad de documentos
- Estadísticas de tokens y longitud de texto

### 3. Experimentos con Modelos

#### LLMs (GPT-4, Claude)
```bash
jupyter notebook openai-4-mini.ipynb
jupyter notebook claude-3.5-sonnet.ipynb
```

#### SLMs (Llama)
```bash
jupyter notebook Llama3.2_\(11B\)-Vision.ipynb
```

### 4. Análisis de Resultados
```bash
jupyter notebook results.ipynb
```
Genera:
- Tablas de métricas comparativas
- Gráficos de rendimiento por epoch
- Análisis por tipo de pregunta
- Gráficos radar para comparación multimodal

## 📊 Resultados Principales

### Rendimiento ANLS por Modelo
| Modelo | Solo Texto | Solo Imagen | Multimodal |
|--------|------------|-------------|------------|
| GPT-4.1 Mini | 0.66 | 0.82 | 0.84 |
| GPT-4.1 Nano | 0.67 | 0.83 | 0.85 |
| Llama 3.2 (Fine-tuned) | 0.36 | 0.84 | 0.86 |

### Hallazgos Clave
- **Llama 3.2 supera a los LLMs en modalidad multimodal**: Con un ANLS de 0.86 vs 0.84-0.85 de GPT-4
- **Rendimiento equivalente en solo imagen**: Llama 3.2 iguala el rendimiento de GPT-4.1 Mini (0.84) y está muy cerca de GPT-4.1 Nano (0.83)
- **Los LLMs mantienen ventaja en solo texto**: GPT-4 supera significativamente a Llama en tareas de solo texto
- **El fine-tuning es crucial para SLMs**: Mejora dramática del rendimiento de Llama 3.2 tras el entrenamiento
- **La modalidad multimodal maximiza el potencial**: Combinando texto e imagen se obtiene el mejor rendimiento
- **Eficiencia vs rendimiento**: Llama 3.2 ofrece rendimiento competitivo o superior con menor costo computacional

## 🛠️ Scripts Importantes

- `utils/utils.py`: Funciones de utilidad para procesamiento de texto y tokens
- `core/settings.py`: Configuraciones centrales del proyecto
- `data_utils/docvqa_imdbs_data.py`: Utilidades específicas para DocVQA
- `training/`: Scripts de fine-tuning para Llama

## 📝 Estructura de Datos

### Dataset Principal
- **Entrenamiento**: ~40.000 ejemplos
- **Validación**: ~5000 ejemplos  
- **Prueba**: ~5000 ejemplos

### Tipos de Preguntas
- Layout: Preguntas sobre estructura del documento
- Table: Preguntas sobre contenido tabular
- List: Preguntas sobre listas e ítems
- Form: Preguntas sobre formularios
- Figure: Preguntas sobre gráficos e imágenes
- FreeText: Preguntas abiertas sobre el contenido
- Yes/No: Preguntas de respuesta binaria

## 🔍 Metodología

1. **Preprocesamiento**: Extracción OCR, tokenización, redimensionamiento de imágenes
2. **Experimentación**: Evaluación sistemática de todos los modelos
3. **Fine-tuning**: Entrenamiento de Llama 3.2 con LoRA
4. **Evaluación**: Métricas estándar de VQA y análisis por categorías
5. **Análisis**: Comparación estadística y visualización de resultados

## 📋 Base de Datos

El proyecto utiliza PostgreSQL para almacenar:
- Metadatos de documentos
- Preguntas y respuestas
- Resultados de experimentos
- Métricas calculadas

Estructura principal:
- `documents`: Información de documentos y OCR
- `questions`: Preguntas y respuestas del dataset
- `results`: Resultados de predicciones por modelo

## 🤝 Contribución

Este es un proyecto de tesis académica. Para consultas o colaboraciones:
- Revisar la documentación en los notebooks
- Consultar los resultados en `final_results_test/`
- Ver análisis detallados en `results.ipynb`

## 📚 Referencias

- SP-DocVQA Dataset: [SP-DocVQA](https://www.docvqa.org/)
- Llama 3.2 Vision: Meta AI
- GPT-4 Mini/Nano: OpenAI
- ANLS Metric: 

```
@inproceedings{inproceedings,
author = {Bai, Haoli and Liu, Zhiguang and Meng, Xiaojun and Wentao, Li and Liu, Shuang and Luo, Yifeng and Xie, Nian and Zheng, Rongfu and Wang, Liangwei and Hou, Lu and Wei, Jiansheng and Jiang, Xin and Liu, Qun},
year = {2023},
month = {01},
pages = {13386-13401},
title = {Wukong-Reader: Multi-modal Pre-training for Fine-grained Visual Document Understanding},
doi = {10.18653/v1/2023.acl-long.748}
}
```
## 📄 Licencia

Este proyecto es parte de una tesis académica de la Universidad Internacional de La Rioja (UNIR).

---

**Autor**: Andres Vasquez Restrepo
**Programa**: Maestría en Inteligencia Artificial
**Universidad**: Universidad Internacional de La Rioja (UNIR)  
**Año**: 2025
