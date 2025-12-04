# 🏪 Retail Store Analytics with AI

## 📋 Resumo Executivo

Sistema de análise de tráfego em lojas de retalho baseado em **computer vision** e **deep learning**. Utiliza **YOLOv8** para deteção e rastreamento de pessoas em tempo real, permitindo análise de:
- Contagem de entradas/saídas
- Ocupação em tempo real
- Padrões de tráfego e picos de afluência
- Mapas de calor de movimento
- Dashboard interativo com métricas em tempo real

---

## 🎯 Objetivos

| # | Funcionalidade | Estado |
|---|----------------|--------|
| 1 | Contagem total de pessoas | ✅ Concluído |
| 2 | Pico de horas de maior afluência | ✅ Concluído |
| 3 | Localização de entrada (coordenadas) | ✅ Concluído |
| 4 | Indicador de ocupação (LOW/MODERATE/FULL) | ✅ Concluído |
| 5 | Heatmap de zonas de tráfego | ✅ Concluído |
| 6 | Dashboard web interativo | ✅ Concluído |
| 7 | Deteção de género | ❌ Qualidade de imagem insuficiente |

---

## 🏗️ Arquitetura Técnica

### Componentes Principais

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│  Video Input    │─────▶│  YOLOv8 +    │─────▶│  Analytics  │
│  (Oxford Town)  │      │  BoTSORT     │      │  Engine     │
└─────────────────┘      └──────────────┘      └─────────────┘
                                                       │
                         ┌─────────────────────────────┼─────────────┐
                         ▼                             ▼             ▼
                  ┌──────────────┐            ┌──────────────┐  ┌──────────┐
                  │  CSV Logs    │            │  Streamlit   │  │ Heatmap  │
                  │  (Timestamp) │            │  Dashboard   │  │ Visual   │
                  └──────────────┘            └──────────────┘  └──────────┘
```

### Pipeline de Processamento

1. **Captura de Video**: Leitura frame-a-frame de video (MP4)
2. **Deteção**: YOLOv8n (`class=0` apenas pessoas)
3. **Rastreamento**: BoTSORT tracker para IDs persistentes
4. **Análise de Zona**: Lógica de crossing detection (linha virtual)
5. **Persistência**: Logs CSV com timestamp, coordenadas, direção
6. **Visualização**: Streamlit dashboard + OpenCV real-time

---

## 🚀 Funcionalidades Implementadas

### 1️⃣ Contagem Bidirecional (IN/OUT)
- **Linha Virtual**: Divide frame em zona TOP/BOTTOM
- **Estado Persistente**: Rastreamento de IDs entre zonas
- **Lógica de Direção**:
  - `Bottom → Top` = **IN** (verde)
  - `Top → Bottom` = **OUT** (ciano)

### 2️⃣ Análise Temporal
- **CSV Export**: `occupancy_log.csv` com:
  ```csv
  Timestamp,Hour,Person_ID,Direction,Occupancy,X,Y
  2025-12-04 00:15:32,00,42,IN,8,512.34,215.67
  ```
- **Agregação Horária**: Análise posterior para identificar picos

### 3️⃣ Coordenadas de Entrada
- **Logging**: Posição (X,Y) exata de cada crossing
- **Uso**: Identificação de portas/entradas preferenciais

### 4️⃣ Indicador de Ocupação
- **Estados Dinâmicos**:
  - 🟢 **LOW** (<50% capacidade)
  - 🟡 **MODERATE** (50-85%)
  - 🔴 **FULL** (>85%)
- **Barra de Progresso**: Visual real-time no dashboard

### 5️⃣ Trajetórias Permanentes
- **Track History**: Armazenamento de todas as posições (pés das pessoas)
- **Visualização**: Linhas brancas persistentes mostrando caminhos
- **Toggle**: Ativável via flag `ENABLE_TRAJECTORY`

### 6️⃣ Dashboard Streamlit
- **Métricas ao Vivo**: Total IN/OUT, ocupação atual
- **Gráficos Plotly**: Análise temporal de tráfego
- **Configurações**: Threshold de confiança, posição da linha, capacidade máxima
- **Controlo**: Start/Stop sistema, reset de dados

### 7️⃣ Deteção de Género (Tentativa Falhada)
- **Bibliotecas Testadas**:
  - ❌ DeepFace + Facenet512
  - ❌ InsightFace + ONNX
- **Motivo Falha**: Resolução de imagem insuficiente (faces muito pequenas no frame)

---

## 🛠️ Stack Tecnológico

| Componente | Tecnologia | Versão |
|------------|-----------|--------|
| **Deteção de Objetos** | YOLOv8 (Ultralytics) | Latest |
| **Visão Computacional** | OpenCV | 4.x |
| **Rastreamento** | BoTSORT | Built-in YOLO |
| **Dashboard** | Streamlit | 1.x |
| **Visualização** | Plotly Express | 5.x |
| **Data Processing** | Pandas, NumPy | Latest |
| **Linguagem** | Python | 3.8+ |

### Instalação

```bash
# Dependências principais
pip install ultralytics opencv-python streamlit plotly pandas numpy

# Opcional (tentativas de género - não funcional)
pip install deepface tf-keras
pip install insightface onnxruntime
```

---

## 📂 Estrutura de Ficheiros

```
retail-analytics/
│
├── projecto_final.py         # Script principal (OpenCV standalone)
├── app.py                     # Dashboard Streamlit∏P
├── ideiasProjecto.md          # Notas de desenvolvimento
│
├── Videos/
│   └── OxfordTownCentre/
│       └── TownCentreXVID.mp4 # Dataset de teste
│
├── yolov8n.pt                 # Modelo YOLOv8 nano
├── my_tracker.yaml            # Configuração BoTSORT
│
├── occupancy_log.csv          # Logs de entrada/saída
└── trajectories.csv           # Dados de trajetórias
```

---

## 🎬 Como Executar

### Versão OpenCV (Standalone)
```bash
python projecto_final.py
```
- Abre janela com visualização real-time
- Press `Q` para sair
- Gera `occupancy_log.csv`

### Versão Streamlit (Dashboard)
```bash
streamlit run app.py
```
- Abre browser em `http://localhost:8501`
- Interface interativa com gráficos
- Controlo total via sidebar

---

## 📊 Resultados

### Performance
- **FPS**: ~15-20 (YOLOv8n em CPU)
- **Precisão**: ~85-90% (pessoas em condições normais)
- **Latência**: <50ms por frame

### Dataset Utilizado
- **Nome**: Oxford Town Centre
- **Frames**: 4500+ frames
- **Resolução**: 1920x1080 → 1020x600 (resize)
- **Pessoas**: 20-30 simultâneas em média

---

## 🔮 Futuras Melhorias

1. **Deteção de Género** (com cameras de melhor qualidade)
2. **Análise Demográfica** (escalão etário via insightFace)
3. **Deteção de Grupos** (famílias, casais)
4. **Tempo de Permanência** (dwell time por zona)
5. **Multi-Camera Support** (triangulação de posições)
6. **Alertas em Tempo Real** (capacidade excedida, filas)
7. **Integração CRM** (cruzamento com dados de vendas)
8. **Edge Deployment** (NVIDIA Jetson para in-store processing)

---

## 📄 Licença

Este projeto foi desenvolvido para fins académicos/demonstrativos.

---

## 👤 Autor

Desenvolvido como projeto final de análise de dados com computer vision.

**Stack**: Python, YOLOv8, OpenCV, Streamlit  
**Dataset**: Oxford Town Centre (público)

---

## 🙏 Agradecimentos

- **Ultralytics** pelo YOLOv8
- **Oxford** pelo dataset público
- **Streamlit** pela framework de dashboards
