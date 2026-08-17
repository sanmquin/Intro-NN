import modelWeights from '../model_weights.json';

export type ModelKey = '1l_1h' | '1l_2h' | '2l_2h';

export interface ModelMetadata {
  key: ModelKey;
  name: string;
  layers: number;
  heads: number;
  description: string;
}

export const MODEL_CONFIGS: Record<ModelKey, ModelMetadata> = {
  '1l_1h': {
    key: '1l_1h',
    name: '1 Layer, 1 Head (Minimal)',
    layers: 1,
    heads: 1,
    description: 'Compact single-head architecture. Proves that 1 layer with 1 global attention head is sufficient for sequence sorting.',
  },
  '1l_2h': {
    key: '1l_2h',
    name: '1 Layer, 2 Heads (Optimal)',
    layers: 1,
    heads: 2,
    description: 'Optimal non-redundant single-layer architecture. 2 heads split d_model into two 16D sub-spaces for specialized token routing.',
  },
  '2l_2h': {
    key: '2l_2h',
    name: '2 Layers, 2 Heads (Baseline)',
    layers: 2,
    heads: 2,
    description: 'Original baseline architecture. Layer 2 refines representation features, demonstrating multi-stage contextual processing.',
  },
};

// Interfaces for weight shapes
export interface LayerNormWeights {
  weight: number[];
  bias: number[];
}

export interface AttnWeights {
  W_q: number[][];
  W_k: number[][];
  W_v: number[][];
  W_o: number[][];
}

export interface FFNWeights {
  fc1: {
    weight: number[][];
    bias: number[];
  };
  fc2: {
    weight: number[][];
    bias: number[];
  };
}

export interface LayerWeights {
  norm1: LayerNormWeights;
  attn: AttnWeights;
  norm2: LayerNormWeights;
  ff: FFNWeights;
}

export interface TransformerWeights {
  n_layers: number;
  n_heads: number;
  d_model: number;
  embedding: number[][];
  pe: number[][];
  layers: LayerWeights[];
  norm: LayerNormWeights;
  fc_out: {
    weight: number[][];
    bias: number[];
  };
}

const allWeights = modelWeights as unknown as Record<ModelKey, TransformerWeights>;

export function getRawWeights(modelKey: ModelKey = '1l_2h'): TransformerWeights {
  return allWeights[modelKey] || allWeights['1l_2h'];
}

// GELU activation function matching PyTorch's nn.GELU() approximation
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

// 1D Vector LayerNorm
export function layerNorm(x: number[], lnWeights: LayerNormWeights, eps: number = 1e-5): number[] {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) {
    mean += x[i];
  }
  mean /= d;

  let variance = 0;
  for (let i = 0; i < d; i++) {
    variance += Math.pow(x[i] - mean, 2);
  }
  variance /= d;

  const output = new Array(d);
  for (let i = 0; i < d; i++) {
    const xNorm = (x[i] - mean) / Math.sqrt(variance + eps);
    output[i] = xNorm * lnWeights.weight[i] + lnWeights.bias[i];
  }
  return output;
}

export function linearProject(x: number[], wRow: number[], bias: number = 0): number {
  let val = bias;
  for (let i = 0; i < x.length; i++) {
    val += x[i] * wRow[i];
  }
  return val;
}

export interface HeadActivation {
  headIndex: number;
  queries: number[][];   // [L, d_k]
  keys: number[][];      // [L, d_k]
  values: number[][];    // [L, d_k]
  rawScores: number[][]; // [L, L]
  attnWeights: number[][]; // [L, L] after softmax
  context: number[][];   // [L, d_k]
}

export interface LayerActivation {
  layerIndex: number;
  input: number[][];            // [L, d_model]
  norm1: number[][];            // [L, d_model]

  Q: number[][];                // [L, d_model]
  K: number[][];                // [L, d_model]
  V: number[][];                // [L, d_model]

  heads: HeadActivation[];
  concatenatedContext: number[][]; // [L, d_model]
  attnProjected: number[][];       // [L, d_model]
  attnResidual: number[][];        // [L, d_model]

  norm2: number[][];            // [L, d_model]
  ffnMid: number[][];           // [L, d_ff]
  ffnOut: number[][];           // [L, d_model]
  ffnResidual: number[][];      // [L, d_model]
}

export interface TransformerActivationTrace {
  modelKey: ModelKey;
  inputTokens: number[];
  embeddings: number[][];      // [L, d_model]
  posEncodings: number[][];    // [L, d_model]
  initialSum: number[][];      // [L, d_model]

  layers: LayerActivation[];
  finalNorm: number[][];       // [L, d_model]
  logits: number[][];          // [L, vocab_size]
  predictions: number[];       // sorted output tokens
}

/**
 * Runs inference on a 5-length digit array using the specified trained weights.
 * Returns a complete activation trace containing all intermediate calculations.
 */
export function runInference(inputTokens: number[], modelKey: ModelKey = '1l_2h'): TransformerActivationTrace {
  const weights = getRawWeights(modelKey);
  const L = inputTokens.length; // 5
  const d_model = weights.d_model; // 32
  const n_heads = weights.n_heads;
  const d_k = d_model / n_heads;

  // 1. Embeddings & PE
  const embeddings: number[][] = [];
  const posEncodings: number[][] = [];
  const initialSum: number[][] = [];

  for (let i = 0; i < L; i++) {
    const token = inputTokens[i];
    const emb = [...weights.embedding[token]];
    embeddings.push(emb);

    const pe = [...weights.pe[i]];
    posEncodings.push(pe);

    const sum = emb.map((v, idx) => v + pe[idx]);
    initialSum.push(sum);
  }

  // 2. Transformer Encoder Layers
  let currentRepresentations = initialSum.map(row => [...row]);
  const layersTrace: LayerActivation[] = [];

  for (let l = 0; l < weights.layers.length; l++) {
    const lw = weights.layers[l];
    const layerInput = currentRepresentations.map(row => [...row]);

    // Norm 1
    const norm1 = layerInput.map(row => layerNorm(row, lw.norm1));

    // Q, K, V projections
    const Q: number[][] = [];
    const K: number[][] = [];
    const V: number[][] = [];

    for (let i = 0; i < L; i++) {
      const qRow: number[] = [];
      const kRow: number[] = [];
      const vRow: number[] = [];

      for (let c = 0; c < d_model; c++) {
        qRow.push(linearProject(norm1[i], lw.attn.W_q[c]));
        kRow.push(linearProject(norm1[i], lw.attn.W_k[c]));
        vRow.push(linearProject(norm1[i], lw.attn.W_v[c]));
      }
      Q.push(qRow);
      K.push(kRow);
      V.push(vRow);
    }

    // Split into Heads
    const headsTrace: HeadActivation[] = [];
    for (let h = 0; h < n_heads; h++) {
      const headQueries: number[][] = [];
      const headKeys: number[][] = [];
      const headValues: number[][] = [];

      for (let i = 0; i < L; i++) {
        headQueries.push(Q[i].slice(h * d_k, (h + 1) * d_k));
        headKeys.push(K[i].slice(h * d_k, (h + 1) * d_k));
        headValues.push(V[i].slice(h * d_k, (h + 1) * d_k));
      }

      // Attention scores and weights
      const rawScores: number[][] = [];
      const attnWeights: number[][] = [];

      for (let i = 0; i < L; i++) {
        const scoresRow: number[] = [];
        let maxScore = -Infinity;

        for (let j = 0; j < L; j++) {
          let dot = 0;
          for (let c = 0; c < d_k; c++) {
            dot += headQueries[i][c] * headKeys[j][c];
          }
          const score = dot / Math.sqrt(d_k);
          scoresRow.push(score);
          if (score > maxScore) maxScore = score;
        }
        rawScores.push(scoresRow);

        // Softmax
        const expRow = scoresRow.map(s => Math.exp(s - maxScore));
        const sumExp = expRow.reduce((sum, val) => sum + val, 0);
        const softmaxRow = expRow.map(v => v / sumExp);
        attnWeights.push(softmaxRow);
      }

      // Context computation
      const headContext: number[][] = [];
      for (let i = 0; i < L; i++) {
        const contextRow = new Array(d_k).fill(0);
        for (let c = 0; c < d_k; c++) {
          for (let j = 0; j < L; j++) {
            contextRow[c] += attnWeights[i][j] * headValues[j][c];
          }
        }
        headContext.push(contextRow);
      }

      headsTrace.push({
        headIndex: h,
        queries: headQueries,
        keys: headKeys,
        values: headValues,
        rawScores,
        attnWeights,
        context: headContext,
      });
    }

    // Concatenate heads
    const concatenatedContext: number[][] = [];
    for (let i = 0; i < L; i++) {
      const concatRow: number[] = [];
      for (let h = 0; h < n_heads; h++) {
        concatRow.push(...headsTrace[h].context[i]);
      }
      concatenatedContext.push(concatRow);
    }

    // Project back
    const attnProjected: number[][] = [];
    for (let i = 0; i < L; i++) {
      const projRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        projRow.push(linearProject(concatenatedContext[i], lw.attn.W_o[c]));
      }
      attnProjected.push(projRow);
    }

    // Attention Residual connection
    const attnResidual: number[][] = [];
    for (let i = 0; i < L; i++) {
      const resRow = layerInput[i].map((v, idx) => v + attnProjected[i][idx]);
      attnResidual.push(resRow);
    }

    // Norm 2
    const norm2 = attnResidual.map(row => layerNorm(row, lw.norm2));

    // FFN
    const ffnMid: number[][] = [];
    const ffnOut: number[][] = [];

    const d_ff = lw.ff.fc1.bias.length; // 64
    for (let i = 0; i < L; i++) {
      const midRow: number[] = [];
      for (let r = 0; r < d_ff; r++) {
        const preAct = linearProject(norm2[i], lw.ff.fc1.weight[r], lw.ff.fc1.bias[r]);
        midRow.push(gelu(preAct));
      }
      ffnMid.push(midRow);

      const outRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        const val = linearProject(midRow, lw.ff.fc2.weight[c], lw.ff.fc2.bias[c]);
        outRow.push(val);
      }
      ffnOut.push(outRow);
    }

    // FFN Residual connection
    const ffnResidual: number[][] = [];
    for (let i = 0; i < L; i++) {
      const resRow = attnResidual[i].map((v, idx) => v + ffnOut[i][idx]);
      ffnResidual.push(resRow);
    }

    layersTrace.push({
      layerIndex: l,
      input: layerInput,
      norm1,
      Q,
      K,
      V,
      heads: headsTrace,
      concatenatedContext,
      attnProjected,
      attnResidual,
      norm2,
      ffnMid,
      ffnOut,
      ffnResidual,
    });

    currentRepresentations = ffnResidual.map(row => [...row]);
  }

  // 3. Final LayerNorm
  const finalNorm = currentRepresentations.map(row => layerNorm(row, weights.norm));

  // 4. Logits and Predictions
  const logits: number[][] = [];
  const predictions: number[] = [];
  const vocab_size = weights.fc_out.bias.length;

  for (let i = 0; i < L; i++) {
    const logitsRow: number[] = [];
    let maxLogit = -Infinity;
    let predToken = 0;

    for (let v = 0; v < vocab_size; v++) {
      const val = linearProject(finalNorm[i], weights.fc_out.weight[v], weights.fc_out.bias[v]);
      logitsRow.push(val);
      if (val > maxLogit) {
        maxLogit = val;
        predToken = v;
      }
    }
    logits.push(logitsRow);
    predictions.push(predToken);
  }

  return {
    modelKey,
    inputTokens,
    embeddings,
    posEncodings,
    initialSum,
    layers: layersTrace,
    finalNorm,
    logits,
    predictions,
  };
}

/**
 * Computes the bilinear similarity bias score between query index u and key index v
 * Bias(u, v) = Embedding[u] * W_q * W_k^T * Embedding[v]^T
 */
export function getMagnitudeBilinearScores(modelKey: ModelKey = '1l_2h'): number[][] {
  const weights = getRawWeights(modelKey);
  const vocab_size = weights.embedding.length;
  const d_model = weights.d_model;
  const lw = weights.layers[0];

  const Q_proj: number[][] = [];
  const K_proj: number[][] = [];

  for (let v = 0; v < vocab_size; v++) {
    const emb = weights.embedding[v];
    const qRow: number[] = [];
    const kRow: number[] = [];
    for (let c = 0; c < d_model; c++) {
      qRow.push(linearProject(emb, lw.attn.W_q[c]));
      kRow.push(linearProject(emb, lw.attn.W_k[c]));
    }
    Q_proj.push(qRow);
    K_proj.push(kRow);
  }

  const bilinearScores: number[][] = [];
  for (let u = 0; u < vocab_size; u++) {
    const row: number[] = [];
    for (let v = 0; v < vocab_size; v++) {
      let score = 0;
      for (let c = 0; c < d_model; c++) {
        score += Q_proj[u][c] * K_proj[v][c];
      }
      row.push(score);
    }
    bilinearScores.push(row);
  }

  return bilinearScores;
}

export interface PCAResult {
  coords: { x: number; y: number }[];
  varianceExplained: [number, number];
}

export function computePCA(vectors: number[][]): PCAResult {
  const N = vectors.length;
  if (N === 0) return { coords: [], varianceExplained: [0, 0] };
  const D = vectors[0].length;

  const mean = new Array(D).fill(0);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      mean[d] += vectors[i][d];
    }
  }
  for (let d = 0; d < D; d++) {
    mean[d] /= N;
  }

  const centered: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(D);
    for (let d = 0; d < D; d++) {
      row[d] = vectors[i][d] - mean[d];
    }
    centered.push(row);
  }

  const cov: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      let sum = 0;
      for (let k = 0; k < N; k++) {
        sum += centered[k][i] * centered[k][j];
      }
      cov[i][j] = sum / (N > 1 ? N - 1 : 1);
    }
  }

  let totalVariance = 0;
  for (let i = 0; i < D; i++) {
    totalVariance += cov[i][i];
  }
  if (totalVariance === 0) totalVariance = 1;

  const powerIteration = (matrix: number[][], maxIter = 80): { vector: number[]; eigenvalue: number } => {
    let vec = new Array(D).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    vec = vec.map(v => v / (norm || 1));

    for (let iter = 0; iter < maxIter; iter++) {
      const nextVec = new Array(D).fill(0);
      for (let r = 0; r < D; r++) {
        for (let c = 0; c < D; c++) {
          nextVec[r] += matrix[r][c] * vec[c];
        }
      }
      norm = Math.sqrt(nextVec.reduce((sum, v) => sum + v * v, 0));
      if (norm < 1e-12) break;
      vec = nextVec.map(v => v / norm);
    }

    let eigenvalue = 0;
    for (let r = 0; r < D; r++) {
      let rowSum = 0;
      for (let c = 0; c < D; c++) {
        rowSum += matrix[r][c] * vec[c];
      }
      eigenvalue += vec[r] * rowSum;
    }

    return { vector: vec, eigenvalue: Math.max(0, eigenvalue) };
  };

  const pc1Result = powerIteration(cov);
  const pc1 = pc1Result.vector;

  const covDeflated: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) {
      covDeflated[r][c] = cov[r][c] - pc1Result.eigenvalue * pc1[r] * pc1[c];
    }
  }

  const pc2Result = powerIteration(covDeflated);
  const pc2 = pc2Result.vector;

  const coords = centered.map(vec => {
    let x = 0;
    let y = 0;
    for (let d = 0; d < D; d++) {
      x += vec[d] * pc1[d];
      y += vec[d] * pc2[d];
    }
    return { x, y };
  });

  const var1 = Math.min(100, (pc1Result.eigenvalue / totalVariance) * 100);
  const var2 = Math.min(100, (pc2Result.eigenvalue / totalVariance) * 100);

  return {
    coords,
    varianceExplained: [var1, var2],
  };
}

export interface TensorInfo {
  id: string;
  name: string;
  group: string;
  shape: number[];
  paramCount: number;
  data: number[] | number[][];
  min: number;
  max: number;
  mean: number;
  std: number;
}

export function getNetworkParameterDetails(modelKey: ModelKey = '1l_2h'): {
  tensors: TensorInfo[];
  totalParams: number;
  groupBreakdown: { group: string; count: number }[];
} {
  const weights = getRawWeights(modelKey);
  const tensors: TensorInfo[] = [];

  const addTensor = (id: string, name: string, group: string, shape: number[], data: number[] | number[][]) => {
    const flat: number[] = [];
    if (Array.isArray(data[0])) {
      (data as number[][]).forEach(row => flat.push(...row));
    } else {
      flat.push(...(data as number[]));
    }

    const paramCount = flat.length;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;

    for (let i = 0; i < flat.length; i++) {
      const v = flat[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const mean = sum / paramCount;

    let varSum = 0;
    for (let i = 0; i < flat.length; i++) {
      varSum += Math.pow(flat[i] - mean, 2);
    }
    const std = Math.sqrt(varSum / paramCount);

    tensors.push({ id, name, group, shape, paramCount, data, min, max, mean, std });
  };

  addTensor('embedding', 'Token Embedding Matrix (W_emb)', 'Embeddings', [10, 32], weights.embedding);
  addTensor('pe', 'Positional Encoding Matrix (PE)', 'Embeddings', [5, 32], weights.pe);

  weights.layers.forEach((lw, idx) => {
    const lName = `Layer ${idx + 1}`;
    addTensor(`l${idx}_norm1_w`, `${lName} - LayerNorm 1 Weight`, lName, [32], lw.norm1.weight);
    addTensor(`l${idx}_norm1_b`, `${lName} - LayerNorm 1 Bias`, lName, [32], lw.norm1.bias);

    addTensor(`l${idx}_attn_q`, `${lName} - Attention Query Projection (W_q)`, lName, [32, 32], lw.attn.W_q);
    addTensor(`l${idx}_attn_k`, `${lName} - Attention Key Projection (W_k)`, lName, [32, 32], lw.attn.W_k);
    addTensor(`l${idx}_attn_v`, `${lName} - Attention Value Projection (W_v)`, lName, [32, 32], lw.attn.W_v);
    addTensor(`l${idx}_attn_o`, `${lName} - Attention Output Projection (W_o)`, lName, [32, 32], lw.attn.W_o);

    addTensor(`l${idx}_norm2_w`, `${lName} - LayerNorm 2 Weight`, lName, [32], lw.norm2.weight);
    addTensor(`l${idx}_norm2_b`, `${lName} - LayerNorm 2 Bias`, lName, [32], lw.norm2.bias);

    addTensor(`l${idx}_ff1_w`, `${lName} - FFN FC1 Weight`, lName, [64, 32], lw.ff.fc1.weight);
    addTensor(`l${idx}_ff1_b`, `${lName} - FFN FC1 Bias`, lName, [64], lw.ff.fc1.bias);
    addTensor(`l${idx}_ff2_w`, `${lName} - FFN FC2 Weight`, lName, [32, 64], lw.ff.fc2.weight);
    addTensor(`l${idx}_ff2_b`, `${lName} - FFN FC2 Bias`, lName, [32], lw.ff.fc2.bias);
  });

  addTensor('final_norm_w', 'Final LayerNorm Weight', 'Final LayerNorm', [32], weights.norm.weight);
  addTensor('final_norm_b', 'Final LayerNorm Bias', 'Final LayerNorm', [32], weights.norm.bias);

  addTensor('fc_out_w', 'Classifier Output Weight (W_out)', 'Classifier FC', [10, 32], weights.fc_out.weight);
  addTensor('fc_out_b', 'Classifier Output Bias (b_out)', 'Classifier FC', [10], weights.fc_out.bias);

  let totalParams = 0;
  const groupMap = new Map<string, number>();

  tensors.forEach(t => {
    totalParams += t.paramCount;
    const current = groupMap.get(t.group) || 0;
    groupMap.set(t.group, current + t.paramCount);
  });

  const groupBreakdown = Array.from(groupMap.entries()).map(([group, count]) => ({ group, count }));

  return { tensors, totalParams, groupBreakdown };
}

export interface DimensionContribution {
  dimIndex: number;
  activation: number;
  weight: number;
  contribution: number;
}

export interface LogitAttribution {
  tokenPosition: number;
  digit: number;
  finalNormVector: number[];
  fcWeightRow: number[];
  fcBias: number;
  contributions: number[];
  totalLogit: number;
  topPositive: DimensionContribution[];
  topNegative: DimensionContribution[];
}

export function getLogitAttribution(
  trace: TransformerActivationTrace,
  positionIdx: number,
  digit: number
): LogitAttribution {
  const weights = getRawWeights(trace.modelKey);
  const finalNormVector = trace.finalNorm[positionIdx];
  const fcWeightRow = weights.fc_out.weight[digit];
  const fcBias = weights.fc_out.bias[digit];

  const contributions: number[] = new Array(32);
  const dimList: DimensionContribution[] = [];
  let sumProd = 0;

  for (let k = 0; k < 32; k++) {
    const act = finalNormVector[k];
    const w = fcWeightRow[k];
    const prod = act * w;
    contributions[k] = prod;
    sumProd += prod;

    dimList.push({
      dimIndex: k,
      activation: act,
      weight: w,
      contribution: prod,
    });
  }

  const totalLogit = sumProd + fcBias;

  const sorted = [...dimList].sort((a, b) => b.contribution - a.contribution);
  const topPositive = sorted.filter(d => d.contribution > 0).slice(0, 6);
  const topNegative = sorted.filter(d => d.contribution < 0).reverse().slice(0, 6);

  return {
    tokenPosition: positionIdx,
    digit,
    finalNormVector,
    fcWeightRow,
    fcBias,
    contributions,
    totalLogit,
    topPositive,
    topNegative,
  };
}

/**
 * Step-by-Step Cell Derivation structures for Ground-Up Matrix Explanations
 */

export interface ElementwiseProd {
  dimIndex: number;
  qVal: number;
  kVal: number;
  prod: number;
}

export interface AttentionCellDerivation {
  layerIdx: number;
  headIdx: number;
  queryPos: number;
  queryToken: number;
  keyPos: number;
  keyToken: number;
  queryVector: number[];
  keyVector: number[];
  elementwiseProds: ElementwiseProd[];
  dotProductSum: number;
  d_k: number;
  sqrt_d_k: number;
  scaledScore: number; // S_ij
  expValue: number;    // e^(S_ij)
  rowExpSum: number;   // sum_k e^(S_ik)
  softmaxWeight: number; // A_ij
  valueVector: number[];
  weightedValueVector: number[]; // A_ij * V_j
}

export function getAttentionCellDerivation(
  trace: TransformerActivationTrace,
  layerIdx: number,
  headIdx: number,
  queryPos: number,
  keyPos: number
): AttentionCellDerivation {
  const headTrace = trace.layers[layerIdx].heads[headIdx];
  const queryToken = trace.inputTokens[queryPos];
  const keyToken = trace.inputTokens[keyPos];

  const qVec = headTrace.queries[queryPos];
  const kVec = headTrace.keys[keyPos];
  const vVec = headTrace.values[keyPos];

  const d_k = qVec.length;
  const sqrt_d_k = Math.sqrt(d_k);

  const elementwiseProds: ElementwiseProd[] = [];
  let dotProductSum = 0;

  for (let c = 0; c < d_k; c++) {
    const prod = qVec[c] * kVec[c];
    dotProductSum += prod;
    elementwiseProds.push({
      dimIndex: c,
      qVal: qVec[c],
      kVal: kVec[c],
      prod,
    });
  }

  const scaledScore = headTrace.rawScores[queryPos][keyPos];
  const softmaxWeight = headTrace.attnWeights[queryPos][keyPos];

  // Re-calculate exponent sum across row for clarity
  const scoresRow = headTrace.rawScores[queryPos];
  const maxScore = Math.max(...scoresRow);
  const expsRow = scoresRow.map(s => Math.exp(s - maxScore));
  const rowExpSum = expsRow.reduce((s, e) => s + e, 0);
  const expValue = Math.exp(scaledScore - maxScore);

  const weightedValueVector = vVec.map(val => val * softmaxWeight);

  return {
    layerIdx,
    headIdx,
    queryPos,
    queryToken,
    keyPos,
    keyToken,
    queryVector: qVec,
    keyVector: kVec,
    elementwiseProds,
    dotProductSum,
    d_k,
    sqrt_d_k,
    scaledScore,
    expValue,
    rowExpSum,
    softmaxWeight,
    valueVector: vVec,
    weightedValueVector,
  };
}

export interface LearnedBiasCellDerivation {
  modelKey: ModelKey;
  uDigit: number;
  vDigit: number;
  embeddingU: number[];
  embeddingV: number[];
  qProjU: number[];
  kProjV: number[];
  elementwiseProds: ElementwiseProd[];
  bilinearScore: number;
}

export function getLearnedBiasCellDerivation(
  modelKey: ModelKey,
  uDigit: number,
  vDigit: number
): LearnedBiasCellDerivation {
  const weights = getRawWeights(modelKey);
  const embU = weights.embedding[uDigit];
  const embV = weights.embedding[vDigit];
  const lw = weights.layers[0];
  const d_model = weights.d_model;

  const qProjU: number[] = new Array(d_model);
  const kProjV: number[] = new Array(d_model);

  for (let c = 0; c < d_model; c++) {
    qProjU[c] = linearProject(embU, lw.attn.W_q[c]);
    kProjV[c] = linearProject(embV, lw.attn.W_k[c]);
  }

  const elementwiseProds: ElementwiseProd[] = [];
  let bilinearScore = 0;

  for (let c = 0; c < d_model; c++) {
    const prod = qProjU[c] * kProjV[c];
    bilinearScore += prod;
    elementwiseProds.push({
      dimIndex: c,
      qVal: qProjU[c],
      kVal: kProjV[c],
      prod,
    });
  }

  return {
    modelKey,
    uDigit,
    vDigit,
    embeddingU: embU,
    embeddingV: embV,
    qProjU,
    kProjV,
    elementwiseProds,
    bilinearScore,
  };
}
