import modelWeights from '../model_weights.json';

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
  embedding: number[][];
  pe: number[][];
  layers: LayerWeights[];
  norm: LayerNormWeights;
  fc_out: {
    weight: number[][];
    bias: number[];
  };
}

// Ensure loaded weights match our interface
const weights = modelWeights as unknown as TransformerWeights;

// GELU activation function matching PyTorch's nn.GELU() approximation
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

// 1D Vector LayerNorm
export function layerNorm(x: number[], lnWeights: LayerNormWeights, eps: number = 1e-5): number[] {
  const d = x.length;
  // Mean
  let mean = 0;
  for (let i = 0; i < d; i++) {
    mean += x[i];
  }
  mean /= d;

  // Variance
  let variance = 0;
  for (let i = 0; i < d; i++) {
    variance += Math.pow(x[i] - mean, 2);
  }
  variance /= d;

  // Normalize, scale, and shift
  const output = new Array(d);
  for (let i = 0; i < d; i++) {
    const xNorm = (x[i] - mean) / Math.sqrt(variance + eps);
    output[i] = xNorm * lnWeights.weight[i] + lnWeights.bias[i];
  }
  return output;
}

// Matrix multiplication helper: inputs of shape [L, d_in] multiplied by W^T of shape [d_in, d_out]
// W is of shape [d_out, d_in]
export function linearProject(x: number[], wRow: number[], bias: number = 0): number {
  let val = bias;
  for (let i = 0; i < x.length; i++) {
    val += x[i] * wRow[i];
  }
  return val;
}

export interface HeadActivation {
  headIndex: number;
  queries: number[][]; // [L, d_k]
  keys: number[][];    // [L, d_k]
  values: number[][];  // [L, d_k]
  rawScores: number[][]; // [L, L]
  attnWeights: number[][]; // [L, L] after softmax
  context: number[][];   // [L, d_k]
}

export interface LayerActivation {
  layerIndex: number;
  input: number[][];            // [L, d_model]
  norm1: number[][];            // [L, d_model]

  // All projections
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
 * Runs inference on a 5-length digit array using the trained weights.
 * Returns a complete activation trace containing all intermediate calculations.
 */
export function runInference(inputTokens: number[]): TransformerActivationTrace {
  const L = inputTokens.length; // should be 5
  const d_model = weights.embedding[0].length; // 32
  const n_heads = weights.layers[0].attn.W_q.length / (weights.embedding[0].length / 2) ? 2 : 2; // n_heads = 2
  const d_k = d_model / n_heads; // 16

  // 1. Embeddings
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

    // Linear projections for Q, K, V
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

        // Compute dot product scores
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
        context: headContext
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
      ffnResidual
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
    inputTokens,
    embeddings,
    posEncodings,
    initialSum,
    layers: layersTrace,
    finalNorm,
    logits,
    predictions
  };
}

/**
 * Computes the bilinear similarity bias score between query index u and key index v
 * Bias(u, v) = Embedding[u] * W_q * W_k^T * Embedding[v]^T
 * This lets us plot the learned magnitude bias.
 */
export function getMagnitudeBilinearScores(): number[][] {
  const vocab_size = weights.embedding.length;
  const d_model = weights.embedding[0].length;
  const lw = weights.layers[0]; // analyze layer 1 magnitude interaction

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

/**
 * Returns raw model weights for deep inspection.
 */
export function getRawWeights(): TransformerWeights {
  return weights;
}

export interface PCAResult {
  coords: { x: number; y: number }[];
  varianceExplained: [number, number];
}

/**
 * Computes Principal Component Analysis (PCA) on high-dimensional vectors (e.g. 32D embeddings)
 * to project them into 2D coordinates for visual scatter plots.
 */
export function computePCA(vectors: number[][]): PCAResult {
  const N = vectors.length;
  if (N === 0) return { coords: [], varianceExplained: [0, 0] };
  const D = vectors[0].length;

  // 1. Calculate Mean Vector
  const mean = new Array(D).fill(0);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      mean[d] += vectors[i][d];
    }
  }
  for (let d = 0; d < D; d++) {
    mean[d] /= N;
  }

  // 2. Center Vectors
  const centered: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(D);
    for (let d = 0; d < D; d++) {
      row[d] = vectors[i][d] - mean[d];
    }
    centered.push(row);
  }

  // 3. Compute Covariance Matrix (D x D)
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

  // Compute Total Variance (Trace of Covariance Matrix)
  let totalVariance = 0;
  for (let i = 0; i < D; i++) {
    totalVariance += cov[i][i];
  }
  if (totalVariance === 0) totalVariance = 1;

  // Power Iteration for Top Eigenvector (PC1)
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

    // Compute eigenvalue lambda = v^T * A * v
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

  // PC1
  const pc1Result = powerIteration(cov);
  const pc1 = pc1Result.vector;

  // Deflate Covariance Matrix: C_deflated = C - lambda1 * (v1 * v1^T)
  const covDeflated: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) {
      covDeflated[r][c] = cov[r][c] - pc1Result.eigenvalue * pc1[r] * pc1[c];
    }
  }

  // PC2
  const pc2Result = powerIteration(covDeflated);
  const pc2 = pc2Result.vector;

  // 4. Project Centered Data onto PC1 and PC2
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

/**
 * Extracts comprehensive details, parameter shapes, counts, and metrics for every layer in the network.
 */
export function getNetworkParameterDetails(): {
  tensors: TensorInfo[];
  totalParams: number;
  groupBreakdown: { group: string; count: number }[];
} {
  const tensors: TensorInfo[] = [];

  const addTensor = (id: string, name: string, group: string, shape: number[], data: number[] | number[][]) => {
    // Flatten data to compute metrics
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

/**
 * Computes dimension-wise influence from the last transformer layer output (finalNorm)
 * into the final fully connected output layer (fc_out) for a specific position and target digit.
 */
export function getLogitAttribution(
  trace: TransformerActivationTrace,
  positionIdx: number,
  digit: number
): LogitAttribution {
  const finalNormVector = trace.finalNorm[positionIdx]; // [32]
  const fcWeightRow = weights.fc_out.weight[digit];     // [32]
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

  // Sort dimensions by contribution descending
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
