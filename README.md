# Fine-tuning e regiões de atenção em CNNs

Projeto de pesquisa sobre o efeito do fine-tuning na classificação de imagens e nas regiões de atenção utilizadas por uma rede neural convolucional.

Pergunta central:

> O fine-tuning altera apenas a acurácia ou também muda as regiões utilizadas por uma CNN?

O estudo utiliza uma ResNet-18 pré-treinada no ImageNet e o conjunto Oxford-IIIT Pet, que possui rótulos de 37 raças e máscaras de segmentação. São comparadas duas estratégias:

- `frozen`: backbone congelado e treinamento apenas do classificador;
- `full`: fine-tuning de toda a rede, com taxas de aprendizado distintas para backbone e classificador.

A análise combina métricas de classificação, calibração, custo de treinamento e propriedades espaciais dos mapas Grad-CAM, incluindo energia dentro do animal, pointing game e sobreposição com a máscara.

## Estrutura

- `article/main.tex`: artigo no formato IEEE;
- `article/references.bib`: referências bibliográficas;
- `cnn_finetuning_gradcam_colab.ipynb`: notebook para execução no Google Colab;
- `src/experiment.py`: treinamento, avaliação e análise Grad-CAM;
- `src/aggregate_results.py`: agregação de execuções com diferentes sementes;
- `requirements.txt`: dependências do projeto;
- `run_experiment.ps1`: execução de um experimento;
- `run_all_seeds.ps1`: execução de múltiplas sementes.

## Execução

Crie um ambiente virtual, instale as dependências e execute:

```powershell
python -m pip install -r requirements.txt
./run_experiment.ps1
```

Para uma verificação rápida:

```powershell
python src/experiment.py --epochs 1 --train-fraction 0.10 --test-limit 128 --gradcam-samples 20 --output-dir results/quick
```

Para uma semente do experimento principal:

```powershell
python src/experiment.py --epochs 15 --train-fraction 1.0 --gradcam-samples 200 --seed 42 --output-dir results/seed_42
```

As três execuções e a agregação podem ser iniciadas com:

```powershell
./run_all_seeds.ps1
```

## Compilação do artigo

Em uma instalação LaTeX com `latexmk`:

```powershell
cd article
latexmk -pdf main.tex
```

O arquivo `article/main.tex` usa diretamente a classe `IEEEtran` no modo de conferência. O agregador copia a figura da semente 42 para `results/aggregate/figures/gradcam_examples.png`; recompile o PDF depois de gerar esse arquivo.

## Estado do projeto

Os arquivos versionados em `results/smoke*` são apenas validações funcionais reduzidas. Eles não sustentam conclusões sobre desempenho ou alinhamento espacial. Antes da entrega quantitativa, execute as sementes 42, 123 e 2026, agregue os resultados e atualize o artigo exclusivamente a partir dos CSVs gerados.
