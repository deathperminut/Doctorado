---
paper: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
tags: [literatura, deep-learning, transformers, ejemplo]
status: leido
---

# Attention Is All You Need

## 📋 Metadata
- **Autores:** Vaswani, Shazeer, Parmar, et al.
- **Año:** 2017
- **Journal/Conference:** NIPS 2017
- **DOI/URL:** https://arxiv.org/abs/1706.03762
- **Zotero:** [[Referencias]]

## 🎯 Problema que Resuelve
Los modelos de secuencia a secuencia anteriores dependían de RNNs/LSTMs que eran difíciles de paralelizar y tenían problemas con dependencias de largo alcance.

## 💡 Contribución Principal
Introduce la arquitectura **Transformer**, que usa exclusivamente mecanismos de atención (self-attention) sin recurrencia, permitiendo mayor paralelización y mejor captura de dependencias.

## 🔬 Metodología
- Multi-head self-attention
- Positional encoding
- Feed-forward networks
- Layer normalization
- Residual connections

## 📊 Resultados Clave
- SOTA en traducción automática (WMT 2014 EN-DE: 28.4 BLEU)
- Entrenamiento mucho más rápido que modelos recurrentes
- Mejor interpretabilidad a través de attention weights

## 💭 Fortalezas
- Altamente paralelizable
- Captura dependencias de largo alcance
- Base para GPT, BERT, y modelos modernos

## ⚠️ Limitaciones
- Requiere más memoria para secuencias largas (O(n²))
- Necesita positional encoding explícito

## 🔗 Conexiones
Papers relacionados:
- [[BERT]] - Usa transformers bidireccionales
- [[GPT]] - Transformers autoregresivos

Conceptos clave:
- [[Self-Attention]]
- [[Multi-Head Attention]]

## 💡 Ideas para mi Investigación
- ¿Puedo aplicar transformers a mi dominio específico?
- Investigar variantes eficientes para secuencias largas

## 📝 Citas Importantes
> "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."

---
**Estado:** #leido
**Relevancia:** ⭐⭐⭐⭐⭐
