# Transformer Attention & Internal Interpretability Visualizer

An interactive, dark-themed React + Vite + TypeScript web application designed for deep interpretability and visualization of a trained sequence sorting Transformer model ($V=10$, $L=5$, $d_{model}=32$, 2 Attention Heads, 2 Encoder Layers, 13,802 total parameters).

---

## 🌟 Key Application Features & Modules

### 1. 🧠 Attention & Step-by-Step Execution Trace (`AttentionHeatmap`, `ArchitectureDiagram`, `MathExplainer`)
- **Interactive Sequencer**: Modify any digit in the 5-length input sequence using step buttons or presets (Mixed Random, Reverse Sorted, Fully Sorted, Duplicate Values).
- **Forward Pass Execution Trace**: Step through individual layers (Embeddings, Transformer Layer 1, Transformer Layer 2, Classifier Output Head) and attention heads.
- **Self-Attention Heatmaps**: Observe dynamic Query-Key-Value routing and attention weight distributions ($L \times L$).
- **Learned Ordering Bias**: Visualizes the $10 \times 10$ bilinear Query-Key magnitude bias matrix $\text{Bias}(u, v) = \mathbf{e}_u \mathbf{W}_q \mathbf{W}_k^T \mathbf{e}_v^T$.

---

### 2. 🌐 Vocabulary & Positional Embedding Space (`EmbeddingScatterPlot`)
- **2D PCA Projection**: Projects the 32-dimensional token embeddings ($d_{model}=32$) into 2D principal component space ($PC_1, PC_2$).
- **Multi-View Inspection Modes**:
  - **Token Embeddings Only**: Inspect raw digit vectors ($0 \dots 9$) and see their continuous spatial ordering.
  - **Token + Positional Encoding**: Evaluate how sinusoidal positional encodings ($PE_1 \dots PE_5$) shift token representations in high-dimensional space without destroying relative magnitude relationships.
  - **All 50 Combinations**: Scatter plot of all $10 \text{ tokens} \times 5 \text{ positions}$ simultaneously.
- **Interactive Point Inspector**: Hover over any scatter point to inspect its exact $(PC_1, PC_2)$ coordinates, vector L2 norm $\|E\|_2$, and sample vector dimensions.

---

### 3. 🔍 Full Network Parameter Inspector (`ParameterInspector`)
- **Complete Architecture Breakdown**: Comprehensive tree and group cards detailing all **13,802 parameters** across 5 functional layer groups:
  1. **Embeddings**: Token Embedding ($10 \times 32$), Positional Encoding ($5 \times 32$).
  2. **Layer 1**: LayerNorm 1, $W_q, W_k, W_v, W_o$ ($32 \times 32$ each), LayerNorm 2, FFN FC1 ($64 \times 32$), FFN FC2 ($32 \times 64$).
  3. **Layer 2**: LayerNorm 1, $W_q, W_k, W_v, W_o$ ($32 \times 32$ each), LayerNorm 2, FFN FC1 ($64 \times 32$), FFN FC2 ($32 \times 64$).
  4. **Final LayerNorm**: Weight ($32$), Bias ($32$).
  5. **Classifier FC**: Weight ($10 \times 32$), Bias ($10$).
- **Interactive 2D Weight Heatmap Grid**: Inspect exact values, tensor shapes, min/max/mean/std stats, and row/column coordinates for any tensor in the network.

---

### 4. ⚡ Classifier Logit Influence & Linear Contribution (`LogitInfluenceVisualizer`)
- **Last Layer to Output Projection**: Dissects how the 32-dimensional output vectors ($h_i \in \mathbb{R}^{32}$) from Transformer Layer 2 directly drive predictions in the output fully connected layer ($W_{fc\_out} \in \mathbb{R}^{10 \times 32}$).
- **Logit Attribution Formula**:
  $$\text{Logit}(pos, v) = \sum_{k=0}^{31} h_{pos}[k] \cdot W_{fc\_out}[v, k] + b_{fc\_out}[v]$$
- **Feature Channel Spectrum**:
  - **Top Positive Drivers**: Identifies feature dimensions in $h$ that align with digit $v$'s classification weights.
  - **Top Negative Suppressors**: Identifies feature dimensions that misalign and penalize target digit predictions.
  - **Full 32-Dimension Bar Chart**: Element-wise product spectrum $h_i[k] \cdot W_{fc\_out}[v, k]$.

---

## 🛠️ Local Development & Running the App

```bash
# Navigate to web folder
cd web

# Install dependencies
npm install

# Start local development server
npm run dev

# Build production bundle
npm run build
```

---

## 📁 Project Structure

```
web/
├── src/
│   ├── components/
│   │   ├── ArchitectureDiagram.tsx    # Interactive Transformer pipeline diagram
│   │   ├── AttentionHeatmap.tsx       # Softmax attention weight heatmaps
│   │   ├── EmbeddingScatterPlot.tsx   # 2D PCA embedding space visualizer
│   │   ├── LogitInfluenceVisualizer.tsx# Classifier logit attribution breakdown
│   │   ├── MathExplainer.tsx          # Real-time mathematical trace panel
│   │   └── ParameterInspector.tsx     # Full network weight parameter deep-dive
│   ├── model/
│   │   └── transformer.ts             # Pure TS forward pass, PCA engine, attribution
│   ├── model_weights.json             # Serialized weight parameters from PyTorch
│   ├── App.tsx                        # Main dashboard & tabbed layout
│   └── main.tsx                       # React application entry point
├── package.json                       # Project configuration and scripts
├── vite.config.ts                     # Vite build setup
└── README.md                          # Application documentation
```
