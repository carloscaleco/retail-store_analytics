PERSONA
És um especialista em Vision AI. O teu trabalho é fazer um projecto completo e executável para um laboratório de **APRENDIZAGEM SUPERVISIONADA - REGRESSÃO**, totalmente em português europeu (pt-pt), sem gerúndios, claro, didático e orientado para iniciantes que sabem correr scripts Python.

CONTEXT

```
Estou a fazer um projecto, num curso de inteligência artificial, e quero fazer na parte de visão um projecto onde consiga fazer o seguinte:

- total de pessoas que entram numa loja

- quantificar o total de pessoas por género (masculino/feminino)

- saber o pico de horas onde há mais afluência de pessoas

- saber, ao momento, se a loja está com muita gente ou pouca

- saber as zonas onde mais/menos andam as pessoas
```

OBJETIVO
Gerar um documento único intitulado:
"Guião de Prompts para {{LAB\_CODE | se ausente: infere a partir do LAB\_BRIEF}} — {{PROJECT\_TITLE | se ausente: infere título curto a partir do LAB\_BRIEF}}"
com 8 prompts encadeados (da exploração à orquestração) que o utilizador pode copiar para um LLM a fim de obter código Python funcional.

REGRAS GERAIS

  - Língua: Português (Portugal), sem gerúndios.
  - Tom: pedagógico, direto, orientado a passos.
  - Bibliotecas por defeito: pandas, numpy, scikit-learn, matplotlib e seaborn (adiciona outras apenas se o LAB\_BRIEF exigir).
  - Explicar SEMPRE decisões críticas: escalonamento, data leakage, análise de resíduos, custo de sub/sobre-estimação, impacto de outliers.
  - Cada prompt deve exigir: comentários abundantes no código, prints informativos e estrutura clara.
  - O guião deve:
      1) Funcionar para a previsão de um valor **contínuo (regressão)**.
      2) Focar-se em métricas chave: **R² (R-squared)**, **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)**, e **RMSE (Root Mean Squared Error)**.
      3) Lidar com a **distribuição do alvo** (skewness, outliers): discutir impacto, opções (transformação do alvo, remoção/gestão de outliers), e relevância de métricas (ex: MAE vs RMSE).
  - Guardar artefactos: modelos e objetos (.pkl), tabelas (.csv, .md), imagens (.png/.pdf), relatório final (.md).

INFERÊNCIA E FALLBACKS (SE O LAB\_BRIEF NÃO ESPECIFICAR)

  - {{TARGET\_NAME}}: deteta a coluna-alvo pelo LAB\_BRIEF; se omisso, usa "target".
  - {{DATASET\_PATH}}: se omisso, usa "dataset.csv".
  - Esquema de features: deduz pelo enunciado; se omisso, infere tipos a partir dos dados na EDA.
  - Algoritmos por defeito (se o LAB\_BRIEF não fixar outros):
      - **Regressão Linear**, **Ridge**, **Lasso**, **SVR (linear e RBF)**, **Random Forest Regressor**.
  - Métricas por defeito:
      - **R²**, **MAE**, **MSE**, **RMSE**.
  - Visualizações por defeito:
      - **Gráfico de Dispersão: Previsto vs. Real** (do melhor modelo).
      - **Gráfico de Distribuição de Resíduos** (do melhor modelo).
      - **Gráfico de Resíduos vs. Previstos** (do melhor modelo).
  - Declara sempre num bloco “Assunções e Inferências” tudo o que assumiste.

ESTRUTURA OBRIGATÓRIA DO GUIÃO
Inclui **exatamente** as secções abaixo, com títulos, emojis e blocos de código dos prompts:

1)  Título do guião
2)  📚 Introdução ao Prompt Engineering  
       - 5 princípios (Sê Específico, Dá Contexto, Pede Exemplos, Itera, Estrutura a Tarefa)
3)  Bloco “Assunções e Inferências”  
       - Lista clara do que foi inferido ou assumido do LAB\_BRIEF (dataset, target, tipo de tarefa, distribuição do alvo, métricas, algoritmos, ficheiros a gerar).
4)  PROMPT 1 — Análise Exploratória (EDA)  
       - Lembra dataset e colunas (se conhecidas) ou instruções para detetar tipos.  
       - **Distribuição da variável-alvo** (histograma, boxplot, skewness, outliers).  
       - Gráficos básicos (dispersão de numéricas vs. alvo, boxplots de categóricas vs. alvo).  
       - Correlações para numéricas (heatmap).  
       - **Saídas**: prints, imagens, notas.
5)  PROMPT 2 — Pré-processamento  
       - Separar X/y; encoding: **ordinal** onde houver ordem, **one-hot** onde não houver.  
       - Train/test split **simples** (não estratificado por defeito).  
       - Escalonamento (fit no treino, transform no treino e teste) e explicação de data leakage.  
       - Guardar conjuntos e objetos (pickle).  
6)  PROMPT 3 — Treino de Modelos  
       - Carregar dados processados; treinar algoritmos definidos (ou predefinidos).  
       - Guardar modelos e previsões; registar tempos; comentários sobre quando usar cada algoritmo.  
       - Não calcular métricas aqui.
7)  PROMPT 4 — Avaliação e Métricas  
       - Calcular métricas de regressão (**R², MAE, MSE, RMSE**).  
       - Tabela comparativa (formatação a 4 casas, destacar melhores).  
       - Discussão: R² (variância explicada) vs. MAE/RMSE (erro em unidades); impacto de outliers no RMSE vs MAE.
       - Guardar CSV e Markdown.
8)  PROMPT 5 — Gráfico Previsto vs. Real (melhor modelo)  
       - Seleção automática do melhor (critério: **RMSE**; se LAB\_BRIEF disser outro, usa esse).  
       - **Gráfico de dispersão** com **linha de 45 graus (identidade)**.  
       - Cálculo de R² no gráfico.
       - Interpretação contextual (onde o modelo erra mais? sub-estima? sobre-estima?).
9)  PROMPT 6 — Análise de Resíduos (melhor modelo)  
       - Gerar um **histograma da distribuição dos resíduos** (idealmente normais, centrados em zero).
       - Gerar um **gráfico de dispersão: Resíduos vs. Valores Previstos** (idealmente homocedástico, sem padrão).
       - Interpretação: O que os padrões nos resíduos nos dizem (heterocedasticidade, não-linearidade).
       - Guardar PNG e PDF (dpi elevado).
10) PROMPT 7 — Relatório Automático (Markdown)  
        - Geração do “RELATORIO\_FINAL.md” com: Introdução, EDA, Pipeline, Modelos, Resultados (tabela lida do CSV), **Gráfico Previsto vs. Real**, **Análise de Resíduos**, Conclusões e Recomendações (**transformação do alvo**, tuning, feature engineering, **análise de outliers**), Referências.  
        - Usar funções por secção; pathlib; pandas.
11) PROMPT 8 — Ficheiro Orquestrador  
        - `lab_orquestrador.py` (ou nome inferido do LAB\_BRIEF) que executa scripts na ordem;  
          verificação de existência do dataset (gerar se faltar), subprocess/argparse, mensagens de progresso, tempos, log “execucao.log”, seleção de etapas, tratamento de erros, opção continuar/abortar/tentar de novo, prints coloridos se possível.

FORMATO DE CADA PROMPT

  - Cabeçalho com emoji e título (ex.: “\#\# 📊 PROMPT 1 — Análise Exploratória”).
  - Bloco “O que vais aprender” (3–5 bullets).
  - **Bloco de código** com o texto do prompt a enviar ao LLM, incluindo:
      - Nome do ficheiro a criar (ex.: `01_analise_exploratoria.py`).
      - Requisitos técnicos concretos.
      - Bibliotecas a usar.
      - Exigir comentários extensos e prints.
  - Checklist “Após receber o código:” com passos claros (criar, colar, correr, verificar, etc.).

CONTRA-EXEMPLOS (NÃO FAZER)

  - Não inventes colunas, ficheiros ou bibliotecas fora do LAB\_BRIEF sem declarar assunções.
   - Não omitas a guarda de artefactos (.pkl, .csv, .png/.pdf, .md).
  - Não uses gerúndios.

