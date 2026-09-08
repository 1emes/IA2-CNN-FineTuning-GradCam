# Fine-Tuning e regiões de atenção em CNNs

Projeto sobre o efeito do fine-tuning na classificação de imagens e nas regiões de atenção utilizadas por uma rede neural convolucional.

## Arquivos

- [`article/main.pdf`](article/main.pdf): versão final revisada do artigo no padrão IEEE;
- [`cnn_finetuning_gradcam_colab.ipynb`](cnn_finetuning_gradcam_colab.ipynb): notebook para execução no Google Colab.

## Execução do notebook

1. Abra o notebook no Google Colab.
2. Execute primeiro a validação rápida para verificar a instalação e o fluxo.
3. Execute o experimento completo com as sementes 42, 123 e 2026.
4. Agregue os resultados e atualize o artigo somente com os dados gerados nessa execução.

O notebook baixa o Oxford-IIIT Pet, treina uma ResNet-18 nas condições `frozen` e `full`, calcula métricas de classificação e calibração e gera mapas Grad-CAM.

## Observação

Os resultados preliminares preservados no notebook servem apenas para validar o fluxo. Conclusões quantitativas dependem da execução completa das três sementes.
