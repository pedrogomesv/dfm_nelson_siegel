# Deep Factor Model para Previsões de Curva de Juros

Implementação e Replicabilidade do artigo "Deep Factor Model para Previsões de Curva de Juros". Este repositório contém o código para treinar e avaliar diferentes modelos de previsão da Estrutura a Termo da Taxa de Juros (ETTJ), incluindo modelos de linha de base e modelos mais avançados baseados em redes neurais profundas (Atenção, Grafos e Transformers).

## Preparação dos Dados

*   Os modelos utilizam dados da curva de juros em formato CSV. O caminho padrão esperado é `data/raw/ettj_padronizado.csv`.
*   O arquivo CSV deve conter uma coluna de data (`Date`) e colunas para as taxas de diferentes maturidades.

## Executando os Modelos

O projeto está organizado em diferentes scripts para executar os modelos:

1.  **Fluxo Principal (Baselines e Otimização):**
    *   O script `src/run_all_baselines.py` executa os modelos de linha de base (Random Walk, DNS, DNSS, NNSS) e o modelo ResNET, opcionalmente otimiza seus hiperparâmetros e re-executa os melhores. É um bom ponto de partida para obter resultados gerais.
    *   O script `src/run_best_models.py` pode ser usado se a otimização já foi feita e você deseja apenas executar os melhores modelos.

2.  **Modelos Avançados (Attention, Graph, Transformer):**
    *   **Geração de Sequências (Pré-requisito para Graph/Transformer):** Os modelos de Grafo e Transformer requerem dados pré-processados em janelas sequenciais. Utilize o script `src/data/create_sequences.py` para gerar esses arquivos (salvos por padrão em `data/processed/sequences`).
    *   **Execução Individual:** Cada modelo avançado pode ser treinado e avaliado de forma independente executando seu respectivo script:
        *   `src/models/attention/attention.py` (Usa dados originais)
        *   `src/models/graph/graph.py` (Requer sequências)
        *   `src/models/transformers/transformers.py` (Requer sequências)

**Configuração:**

*   Cada script aceita diversos argumentos de linha de comando para configurar caminhos de dados, diretórios de saída, hiperparâmetros do modelo, horizontes de previsão, etc.
*   Para ver todos os argumentos disponíveis para um script específico, use a flag `-h` ou `--help` (ex: `python src/run_all_baselines.py --help`).

## Resultados

Os resultados de cada execução (métricas, gráficos, modelos salvos) são armazenados nos diretórios de saída especificados (`--output_dir`), geralmente organizados em subdiretórios correspondentes a cada modelo ou etapa.
