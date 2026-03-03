# OpenFold3-preview: Software Design and Architecture

**Document Version:** 1.0  
**Last Updated:** March 2, 2026  
**Based on:** OpenFold3-preview v0.1.0 (AlphaFold3 reproduction)

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Algorithm](#core-algorithm)
4. [Module Specifications](#module-specifications)
5. [Data Flow](#data-flow)
6. [Memory Optimization](#memory-optimization)
7. [Training vs Inference](#training-vs-inference)
8. [Key Design Decisions](#key-design-decisions)
9. [Performance Characteristics](#performance-characteristics)
10. [Extension Points](#extension-points)

---

## Overview

### What is OpenFold3-preview?

OpenFold3-preview is a **biomolecular structure prediction model** that reproduces DeepMind's AlphaFold3 architecture. It predicts 3D structures of proteins, RNA, DNA, and ligands from sequence information using deep learning.

### Key Capabilities

- **Multi-molecule support**: Proteins, RNA, DNA, small molecule ligands
- **Non-canonical residues**: Modified amino acids and nucleotides
- **Complex prediction**: Heteromers, homomers, protein-nucleic acid complexes
- **Template integration**: Optional use of structural templates
- **MSA processing**: Evolutionary information from multiple sequence alignments
- **Confidence estimation**: pLDDT, PAE, PDE metrics

### Technology Stack

- **Framework**: PyTorch
- **Accelerators**: NVIDIA GPUs (CUDA)
- **Optimization Kernels**: 
  - DeepSpeed4Science (Evoformer attention)
  - cuEquivariance (triangle operations)
- **License**: Apache 2.0

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Processing                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Sequences  │  │     MSAs     │  │   Templates  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │ Input Embedder  │                        │
│                  └────────┬────────┘                        │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Trunk                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Template Embedder (Optional)               │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │              MSA Module Stack                       │     │
│  │    (MSAModuleStack - Evoformer-style processing)   │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │            PairFormer Stack (48 blocks)             │     │
│  │    (Iterative refinement of single & pair reps)    │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│                   [Recycling Loop]                          │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 Structure Generation                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Diffusion Module (200 steps)                │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────┐   │   │
│  │  │ Atom Attn  │  │  Diffusion   │  │ Atom Attn  │   │   │
│  │  │  Encoder   │  │ Transformer  │  │  Decoder   │   │   │
│  │  └────────────┘  └──────────────┘  └────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Confidence Heads                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  pLDDT   │  │   PAE    │  │   PDE    │  │Distogram │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Data Structures

#### Representations

| Representation | Shape | Description | Channels |
|---------------|-------|-------------|----------|
| **Single (s)** | `[N_token, c_s]` | Per-token features | `c_s = 384` |
| **Pair (z)** | `[N_token, N_token, c_z]` | Pairwise relationships | `c_z = 128` |
| **MSA (m)** | `[N_seq, N_token, c_m]` | Evolutionary information | `c_m = 64` |
| **Input Single (s_input)** | `[N_token, c_s_input]` | Initial embeddings | `c_s_input = 384` |

#### Key Tensors

```python
# Token-level features
token_mask: torch.Tensor      # [N_token] - valid token mask
num_atoms_per_token: torch.Tensor  # [N_token] - atoms per residue

# Atom-level features  
atom_positions: torch.Tensor  # [N_atom, 3] - 3D coordinates
atom_mask: torch.Tensor       # [N_atom] - valid atom mask
atom_resolved_mask: torch.Tensor  # [N_atom] - resolved in structure

# MSA features
msa: torch.Tensor            # [N_seq, N_token] - aligned sequences
deletion_matrix: torch.Tensor # [N_seq, N_token] - deletion counts
```

---

## Core Algorithm

### Algorithm 1: Main Structure Prediction Loop

The forward pass implements the following steps (based on AF3 Algorithm 1):

```python
def forward(self, batch: dict) -> tuple[dict, dict]:
    """
    Main inference/training loop.
    
    Args:
        batch: Input feature dictionary
        
    Returns:
        batch: Updated batch with predictions
        outputs: Output dictionary with structures and confidence scores
    """
    # Line 1-3: Feature Embedding
    s_input, s_init, z_init = self.input_embedder(batch)
    
    # Line 4-6: Template Integration (optional)
    s, z = self.template_embedder(s_init, z_init, templates)
    
    # Line 7-9: MSA Processing
    s_msa, z_msa = self.msa_module_embedder(s, z, msa_features)
    s, z = self.msa_module(s_msa, z_msa, msa_mask, pair_mask)
    
    # Line 10-14: PairFormer Iterations (with recycling)
    for cycle in range(num_recycles):
        s, z = self.pairformer_stack(s, z, single_mask, pair_mask)
        
        # Normalize and project
        s = self.linear_s(self.layer_norm_s(s))
        z = self.linear_z(self.layer_norm_z(z))
    
    # Line 15: Structure Generation via Diffusion
    if training:
        # Mini-rollout for efficiency (4 samples)
        atom_positions = self.diffusion_module(
            batch, s, z, training=True
        )
    else:
        # Full rollout for accuracy (5+ samples)
        atom_positions = self.sample_diffusion(
            batch, s, z, num_samples=5
        )
    
    # Line 16-17: Confidence Estimation
    aux_outputs = self.aux_heads(
        batch, s_input, {
            "si_trunk": s,
            "zij_trunk": z,
            "atom_positions_predicted": atom_positions
        }
    )
    
    return batch, {**aux_outputs, "atom_positions_predicted": atom_positions}
```

### Algorithm 2: Diffusion-Based Structure Generation

The diffusion module generates structures through iterative denoising:

```python
def diffusion_rollout(s, z, num_steps=200):
    """
    Implements AF3 Algorithm 20.
    
    Args:
        s: Single representation [N_token, c_s]
        z: Pair representation [N_token, N_token, c_z]
        num_steps: Number of diffusion steps
        
    Returns:
        atom_positions: Predicted 3D structure [N_atom, 3]
    """
    # Initialize with random noise
    x_noisy = sample_centered_random_rotation(atom_shape)
    
    # Create noise schedule (AF3 Eq. 24)
    sigma_schedule = create_noise_schedule(
        no_rollout_steps=num_steps,
        sigma_data=1.0,
        s_max=1.0,
        s_min=0.001,
        p=7
    )
    
    # Denoising loop
    for t in reversed(range(num_steps)):
        sigma_t = sigma_schedule[t]
        
        # Encode atom features
        a = atom_attention_encoder(x_noisy, atom_mask, t)
        
        # Transform with conditioning
        a = diffusion_transformer(
            a, s, z, t, 
            chunk_size=chunk_size
        )
        
        # Decode to updated positions
        x_pred = atom_attention_decoder(a, s, atom_mask)
        
        # Remove noise according to schedule
        x_noisy = denoise_step(x_noisy, x_pred, sigma_t)
    
    return x_noisy
```

**Noise Schedule Formula:**

$$\sigma(t) = \sigma_{data} \times \left(s_{max}^{1/p} + t \times (s_{min}^{1/p} - s_{max}^{1/p})\right)^p$$

Where:
- $t \in [0, 1]$ normalized timestep
- $\sigma_{data}$: data variance constant
- $s_{max} = 1.0$: maximum noise standard deviation
- $s_{min} = 0.001$: minimum noise standard deviation
- $p = 7$: curvature parameter

---

## Module Specifications

### 1. Input Embedder (`InputEmbedderAllAtom`)

**Purpose:** Convert discrete inputs to continuous representations

**Architecture:**
```python
class InputEmbedderAllAtom(nn.Module):
    def __init__(self, config):
        # Token feature embedding
        self.token_feat_embedder = TokenFeatEmbedder(c_s_input)
        
        # Profile MSA embedding
        self.profile_msa_embedder = ProfileMSAEmbedder(c_m)
        
        # Positional embeddings
        self.position_embd = PositionalEmbedding(c_s_input)
        
        # Pair feature embedding
        self.pair_feature_embedder = PairFeatureEmbedder(c_z)
```

**Input Features:**
- Amino acid/nucleotide type (one-hot)
- Profile MSA (frequency matrix)
- Deletion statistics
- Relative positional encodings
- Chain and interface tokens
- Bond information

**Output:**
- `s_input`: [N_token, c_s_input] initial single representation
- `z_init`: [N_token, N_token, c_z] initial pair representation

---

### 2. Template Embedder (`TemplateEmbedderAllAtom`)

**Purpose:** Integrate structural template information

**Architecture:**
```python
class TemplateEmbedderAllAtom(nn.Module):
    def __init__(self, config):
        # Template pair stack (similar to PairFormer)
        self.template_pair_stack = PairFormerStack(
            c_s=config.c_s,
            c_z=config.c_z,
            no_blocks=2
        )
        
        # Template pointwise attention
        self.template_pointwise_attention = TemplatePointwiseAttention(
            c_s=config.c_s,
            c_z=config.c_z
        )
```

**Process:**
1. Embed template backbone frames
2. Compute template pair features
3. Process through template pair stack
4. Aggregate multiple templates via attention
5. Update single and pair representations

**Usage:** Optional - can be disabled for ablation studies or when templates unavailable

---

### 3. MSA Module (`MSAModuleStack`)

**Purpose:** Extract evolutionary information from MSAs

**Architecture:**
```python
class MSAModuleStack(nn.Module):
    def __init__(self, config):
        # Stack of MSA processing blocks
        self.msa_stack = EvoformerStack(
            c_m=config.c_m,
            c_z=config.c_z,
            no_blocks=config.no_blocks,  # Typically 4
            no_heads_msa=8,
            no_heads_pair=4
        )
```

**Key Operations:**

**MSA Attention (Row-wise):**
```python
# For each sequence in MSA
msa_row_att = Attention(q=m, k=m, v=m, bias=z)
m = m + Dropout(msa_row_att)
```

**MSA Attention (Column-wise):**
```python
# For each position across sequences
msa_col_att = Attention(q=m, k=m, v=m, mask=msa_mask)
m = m + Dropout(msa_col_att)
```

**Outer Product Mean:**
```python
# Convert MSA info to pair features
z_update = einsum(m - mean(m), m - mean(m), 's i c, s j d -> i j c d')
z = z + Linear(z_update)
```

**Pair Transition:**
```python
# Process pair representation
z = TriangleMultiplication(z)
z = TriangleAttention(z)
z = Transition(z)  # Feedforward
```

---

### 4. PairFormer Stack (`PairFormerStack`)

**Purpose:** Refine single and pair representations through iterative attention

**Architecture:**
```python
class PairFormerStack(nn.Module):
    def __init__(self, config):
        self.blocks = nn.ModuleList([
            PairFormerBlock(config) 
            for _ in range(config.no_blocks)  # 48 blocks
        ])
        
class PairFormerBlock(nn.Module):
    def __init__(self, config):
        # Pair stack operations
        self.pair_stack = PairBlock(
            c_z=config.c_z,
            c_hidden_mul=config.c_hidden_mul,
            c_hidden_pair_att=config.c_hidden_pair_att,
            no_heads_pair=config.no_heads_pair
        )
        
        # Single attends to pair
        self.attn_pair_bias = AttentionPairBias(
            c_q=config.c_s,
            c_k=config.c_s,
            c_v=config.c_s,
            c_z=config.c_z,
            no_heads=config.no_heads_pair_bias
        )
        
        # Single feedforward
        self.single_transition = SwiGLUTransition(
            c_in=config.c_s,
            n=config.transition_n
        )
```

**Forward Pass (per block):**
```python
def forward(self, s, z, single_mask, pair_mask):
    # 1. Pair stack (triangular operations)
    z = self.pair_stack(z, pair_mask)
    
    # 2. Single attends to pair (with pair bias)
    s_update = self.attn_pair_bias(s, z, single_mask)
    s = s + Dropout(s_update)
    
    # 3. Single transition (feedforward)
    s = s + self.single_transition(s, mask=single_mask)
    
    return s, z
```

**Key Innovations:**

**Triangle Multiplicative Update:**
```python
# Outgoing edges
z_ij_out = sum_k (z_ik ⊙ z_jk)  # Element-wise multiplication

# Incoming edges  
z_ij_in = sum_k (z_ki ⊙ z_kj)
```

**Triangle Attention:**
```python
# Starting node attention
For each triangle (i,j,k): attend from i to j given edge (i,k) and (j,k)

# Ending node attention
For each triangle (i,j,k): attend from i to j given edge (k,i) and (k,j)
```

---

### 5. Diffusion Module (`DiffusionModule`)

**Purpose:** Generate 3D structures through iterative denoising

**Architecture:**
```python
class DiffusionModule(nn.Module):
    def __init__(self, config):
        # Condition on single/pair representations
        self.diffusion_conditioning = DiffusionConditioning(
            c_s_input=config.c_s_input,
            c_s=config.c_s,
            c_z=config.c_z
        )
        
        # Atom-level encoding
        self.atom_attn_enc = AtomAttentionEncoder(
            c_a=config.c_a,
            c_z=config.c_z,
            add_noisy_pos=True
        )
        
        # Transformer processing
        self.diffusion_transformer = DiffusionTransformer(
            c_a=config.c_a,
            c_s=config.c_s,
            no_blocks=config.no_blocks,  # 11 blocks
            no_heads=8
        )
        
        # Atom-level decoding
        self.atom_attn_dec = AtomAttentionDecoder(
            c_a=config.c_a,
            c_token=config.c_token
        )
```

**Token-Based Atom Representation:**

Each token (residue/nucleotide) has:
- **Token center**: Representative point (e.g., Cα for amino acids)
- **Local frame**: Rotation matrix defining orientation
- **Atom displacements**: Positions relative to token frame

```python
# Frame construction (protein example)
N, Ca, C = atom_positions[residue]
frame = construct_frame(N, Ca, C)  # Rotation matrix

# Local coordinates
local_coords = frame.T @ (all_atom_positions - Ca)
```

**Diffusion Transformer Block:**
```python
class DiffusionTransformerBlock(nn.Module):
    def __init__(self, config):
        # AdaLN conditioning on timestep
        self.adaLN_modulation = AdaLNModulation(
            c_s=config.c_s,
            c_a=config.c_a
        )
        
        # Self attention on atoms
        self.atom_self_attention = Attention(
            c_q=config.c_a,
            c_k=config.c_a,
            c_v=config.c_a,
            no_heads=config.no_heads
        )
        
        # Cross attention to single rep
        self.atom_to_single_cross_attention = Attention(
            c_q=config.c_a,
            c_k=config.c_s,
            c_v=config.c_s,
            no_heads=config.no_heads
        )
        
        # Feedforward
        self.feed_forward = SwiGLUTransition(
            c_in=config.c_a,
            n=config.transition_n
        )
```

---

### 6. Auxiliary Heads (`AuxiliaryHeadsAllAtom`)

**Purpose:** Predict confidence metrics and auxiliary outputs

**Heads Implemented:**

#### 6.1 pLDDT Head (Per-Residue Confidence)

```python
class PerResidueLDDTAllAtom(nn.Module):
    def __init__(self, config):
        self.lddt_head = nn.Sequential(
            Linear(config.c_s + config.c_a, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, 50)  # 50 bins
        )
```

**Input:** Concatenation of single representation and atom features  
**Output:** Binned lDDT logits [N_atom, 50]  
**Interpretation:** Higher pLDDT = more confident prediction

#### 6.2 PAE Head (Predicted Aligned Error)

```python
class PredictedAlignedErrorHead(nn.Module):
    def __init__(self, config):
        self.pae_head = nn.Sequential(
            Linear(config.c_s, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, 64)  # 64 bins
        )
```

**Input:** Single representation  
**Output:** PAE logits [N_token, N_token, 64]  
**Interpretation:** Expected error when aligning residue i to j

#### 6.3 PDE Head (Predicted Distance Error)

```python
class PredictedDistanceErrorHead(nn.Module):
    def __init__(self, config):
        self.pde_head = nn.Sequential(
            Linear(config.c_z, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, config.c_hidden),
            nn.ReLU(),
            Linear(config.c_hidden, 64)  # 64 bins
        )
```

**Input:** Pair representation  
**Output:** PDE logits [N_token, N_token, 64]  
**Interpretation:** Expected distance error between residues

#### 6.4 Distogram Head

```python
class DistogramHead(nn.Module):
    def __init__(self, config):
        self.distogram_head = Linear(config.c_z, 64)  # 64 distance bins
```

**Input:** Pair representation  
**Output:** Distogram logits [N_token, N_token, 64]  
**Use:** Inter-residue distance distribution

#### 6.5 Experimentally Resolved Head

```python
class ExperimentallyResolvedHeadAllAtom(nn.Module):
    def __init__(self, config):
        self.resolved_head = Linear(config.c_a, 2)  # Binary classification
```

**Input:** Atom features  
**Output:** Logits [N_atom, 2]  
**Use:** Predict if atom would be resolved in experimental structure

---

## Data Flow

### Training Data Pipeline

```
Raw Sequences → MSA Generation → Template Search → Feature Processing → Model Input
```

**Step-by-step:**

1. **MSA Generation** (preprocessing or on-the-fly)
   - JackHMMER against UniRef90
   - HHblits against BFD/UniClust32
   - ColabFold server (MMseqs2)
   
2. **Template Search** (optional)
   - HHsearch against PDB70
   - Select top 4 templates by score
   
3. **Feature Computation**
   - One-hot encode sequences
   - Compute MSA profiles
   - Calculate pairwise features (distances, orientations)
   - Generate positional encodings

4. **Batch Formation**
   - Crop to token budget (e.g., 512 tokens)
   - Pad to maximum size in batch
   - Apply masks

### Forward Pass Data Flow

```python
# Example forward pass with shapes
batch_size = 2
n_token = 384
n_seq = 128  # MSA depth
n_atom = 384 * 5  # ~5 atoms per token average

# 1. Input features
msa: [batch, n_seq, n_token]          # Discrete sequences
deletion_matrix: [batch, n_seq, n_token]
token_mask: [batch, n_token]
atom_positions: [batch, n_atom, 3]

# 2. After input embedder
s_input: [batch, n_token, 384]
z_init: [batch, n_token, n_token, 128]
m: [batch, n_seq, n_token, 64]

# 3. After MSA module
s: [batch, n_token, 384]
z: [batch, n_token, n_token, 128]

# 4. After PairFormer (48 blocks)
s: [batch, n_token, 384]
z: [batch, n_token, n_token, 128]

# 5. After diffusion (200 steps)
atom_positions: [batch, n_atom, 3]

# 6. After confidence heads
plddt_logits: [batch, n_atom, 50]
pae_logits: [batch, n_token, n_token, 64]
pde_logits: [batch, n_token, n_token, 64]
```

---

## Memory Optimization

### Activation Checkpointing

**Strategy:** Re-compute activations during backward pass to save memory

```python
# Enable checkpointing for expensive modules
config.architecture.template.template_pair_stack.blocks_per_ckpt = 1
config.architecture.msa.msa_module.blocks_per_ckpt = 1
config.architecture.pairformer.blocks_per_ckpt = 1

# Implementation in forward pass
from openfold3.core.utils.checkpointing import checkpoint_blocks

def forward_with_checkpointing(blocks, inputs):
    return checkpoint_blocks(
        blocks=blocks,
        args=inputs,
        blocks_per_ckpt=1  # Checkpoint every block
    )
```

### Inference Offloading

**Strategy:** Move intermediate tensors to CPU during inference

```python
# Settings for large sequences
config.settings.memory.eval.offload_inference = {
    "token_cutoff": 1024,  # Only offload for long sequences
    "msa_module": True,
    "pairformer": False,   # Keep on GPU
    "confidence_heads": True
}

# Implementation pattern
def forward_with_offload(module, inputs):
    if offload_inference:
        # Move to CPU before next operation
        inputs_cpu = tensor_tree_map(lambda t: t.cpu(), inputs)
        del inputs
        torch.cuda.empty_cache()
        
        # Move back to GPU when needed
        inputs_gpu = tensor_tree_map(lambda t: t.cuda(), inputs_cpu)
        return module(inputs_gpu)
    else:
        return module(inputs)
```

### Chunked Processing

**Strategy:** Process large tensors in sub-batches

```python
# Automatic chunk size tuning
chunk_size_tuner = ChunkSizeTuner(
    max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
    cueq_max_chunk_size=CUEQ_MAX_CHUNK_SIZE
)

# Usage in attention layers
def attention_with_chunks(q, k, v, chunk_size=None):
    if chunk_size is None:
        chunk_size = chunk_size_tuner.tune(q.shape)
    
    results = []
    for i in range(0, q.shape[0], chunk_size):
        q_chunk = q[i:i+chunk_size]
        attn_out = attention(q_chunk, k, v)
        results.append(attn_out)
    
    return torch.cat(results, dim=0)
```

### Kernel Optimizations

#### DeepSpeed Evoformer Attention

```python
# Enable DeepSpeed kernel
config.settings.memory.train.use_deepspeed_evo_attention = True
config.settings.memory.eval.use_deepspeed_evo_attention = True

# Installation requirement
# git clone https://github.com/NVIDIA/cutlass --branch v3.6.0
# export CUTLASS_PATH=/path/to/cutlass
```

**Benefits:**
- 2-3x speedup for MSA attention
- Reduced memory footprint
- Fused operations

#### cuEquivariance Triangle Kernels

```python
# Enable cuEq kernels
config.settings.memory.train.use_cueq_triangle_kernels = True

# Requires NVIDIA cuEquivariance library
```

**Benefits:**
- Optimized triangular operations
- Better utilization of tensor cores

---

## Training vs Inference

### Training Configuration

```yaml
# Training-specific settings
experiment_settings:
  mode: train
  
  # EMA for validation
  ema:
    decay: 0.999
    update_after_step: 0
    
  # Gradient checkpointing
  settings:
    blocks_per_ckpt: 1
    ckpt_intermediate_steps: true
    
  # Loss weights
  loss:
    diffusion:
      weight: 1.0
      no_mini_rollout_samples: 4  # Fewer samples for speed
    bond:
      weight: 1.0
    smooth_lddt:
      weight: 1.0
      
  # Optimizer
  optimizer:
    learning_rate: 1.8e-3
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    
  # Learning rate schedule
  lr_scheduler:
    warmup_no_steps: 1000
    start_decay_after_n_steps: 50000
    decay_factor: 0.95
```

**Key Differences:**
- Mini-rollout (4 samples) instead of full rollout
- Gradient checkpointing enabled
- Train-time data augmentation
- Mixed precision (BF16/FP32)

### Inference Configuration

```yaml
# Inference-specific settings
experiment_settings:
  mode: predict
  
  # Use EMA weights
  ema:
    enabled: true
    
  # No gradient checkpointing
  settings:
    blocks_per_ckpt: null
    
  # Full diffusion rollout
  shared:
    diffusion:
      no_full_rollout_samples: 5  # More samples for accuracy
      
  # Memory optimizations for large inputs
  memory:
    eval:
      offload_inference:
        token_cutoff: 1024
        msa_module: true
      per_sample_atom_cutoff: 5000  # Chunk metric computation
```

**Key Differences:**
- EMA weights loaded
- Full diffusion rollout (200 steps)
- Multiple random seeds for ensemble
- Optional CPU offloading

---

## Key Design Decisions

### 1. Token-Based Atom Representation

**Decision:** Represent molecules as tokens (residues/nucleotides) with associated atoms

**Rationale:**
- Handles variable number of atoms per residue
- Unified treatment of proteins, nucleic acids, ligands
- Efficient attention computation at token level

**Implementation:**
```python
# Each token has representative atoms
token_representative_atoms = get_token_representative_atoms(
    atom_type, residue_index
)

# Broadcast token features to atoms
atom_features = broadcast_token_feat_to_atoms(
    token_features, num_atoms_per_token
)
```

### 2. Diffusion over SE(3)-Equivariant Layers

**Decision:** Use diffusion model instead of AlphaFold2's structure module

**Rationale:**
- More flexible sampling
- Better handling of multimodal distributions
- Simpler architecture (no explicit frame tracking)

**Trade-offs:**
- Slower inference (200 sequential steps)
- Requires careful noise schedule tuning

### 3. Separate MSA and Pair Processing

**Decision:** Process MSAs separately before PairFormer

**Rationale:**
- MSA depth varies significantly between targets
- Early compression of MSA information
- Reduces computational burden on PairFormer

**Implementation:**
```python
# MSA module compresses to fixed channel size
m: [N_seq, N_token, c_m] -> z_update: [N_token, N_token, c_z]

# PairFormer operates on fixed-size representations
s: [N_token, c_s], z: [N_token, N_token, c_z]
```

### 4. Modular Loss Functions

**Decision:** Separate loss terms for different objectives

**Loss Components:**
```python
total_loss = (
    w_diff * diffusion_loss +
    w_bond * bond_loss +
    w_lddt * smooth_lddt_loss
)
```

**Rationale:**
- Flexible weighting during training stages
- Can disable losses for specific datasets
- Easier debugging and ablation studies

### 5. Confidence Head Separation

**Decision:** Train confidence heads separately after trunk convergence

**Rationale:**
- Prevents trunk from learning trivial confidence solutions
- Allows fine-tuning confidence without retraining entire model
- Matches AlphaFold3 training protocol

**Implementation:**
```python
# Freeze trunk, train only confidence heads
if train_confidence_only:
    freeze_model_parameters()
    unfreeze_submodules([
        aux_heads.pairformer_embedding,
        aux_heads.pde,
        aux_heads.plddt,
        aux_heads.pae
    ])
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Memory Complexity | Notes |
|-----------|----------------|-------------------|-------|
| Input Embedder | O(N_token) | O(N_token²) | Pair features dominate |
| Template Embedder | O(N_template × N_token²) | O(N_token²) | Top 4 templates used |
| MSA Module | O(N_seq × N_token²) | O(N_seq × N_token) | Linear in MSA depth |
| PairFormer | O(N_token²) | O(N_token²) | 48 blocks |
| Diffusion | O(N_steps × N_atom × N_token) | O(N_atom × N_token) | 200 steps |
| Confidence Heads | O(N_token²) | O(N_token²) | Parallel computation |

**Overall:** O(N_token²) space and time complexity

### Scaling Behavior

**Sequence Length:**
- < 256 tokens: Fast (~1 min on A100)
- 256-512 tokens: Moderate (~5 min on A100)
- 512-1024 tokens: Slow (~20 min on A100)
- > 1024 tokens: Requires offloading/memory optimization

**MSA Depth:**
- Shallow MSAs (< 50 sequences): Minimal impact
- Deep MSAs (> 500 sequences): Significant slowdown in MSA module

**Memory Usage:**
- Base model: ~2 GB (parameters)
- Peak training: 40-80 GB (depends on sequence length)
- Peak inference: 10-20 GB (with offloading)

### Benchmark Performance

**Protein Monomers (CASP16):**
- Median GDT-TS: ~85
- Median lDDT: ~80

**Protein-Protein Complexes:**
- Interface lDDT: ~70
- DockQ: ~0.5

**RNA Structures:**
- RMSD: ~3-5 Å
- Strong performance on monomeric RNA

**Protein-Ligand:**
- Ligand RMSD: ~1-2 Å
- Competitive with AlphaFold3

---

## Extension Points

### Adding New Molecule Types

**Required Changes:**

1. **Atom Type Encoding**
```python
# Add new atom types to atom_type_map
ATOM_TYPE_MAP.update({
    "new_molecule": {
        "atoms": ["atom1", "atom2", ...],
        "bonds": [(0, 1), (1, 2), ...]
    }
})
```

2. **Input Features**
```python
# Extend input embedder
class ExtendedInputEmbedder(InputEmbedderAllAtom):
    def __init__(self, config):
        super().__init__(config)
        self.new_molecule_embedder = NewMoleculeEmbedder()
```

3. **Bond Constraints**
```python
# Add bond loss terms
def extended_bond_loss(predictions, batch):
    base_loss = bond_loss(predictions, batch)
    new_bond_loss = custom_bond_constraint(predictions)
    return base_loss + new_bond_loss
```

### Custom Diffusion Schedules

**Override noise schedule:**
```python
def custom_noise_schedule(no_rollout_steps):
    t = torch.linspace(0, 1, no_rollout_steps)
    
    # Cosine schedule instead of polynomial
    return sigma_data * torch.cos(t * pi/2)
```

### Alternative Confidence Metrics

**Add new head:**
```python
class CustomConfidenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Sequential(
            Linear(config.c_z, 256),
            nn.ReLU(),
            Linear(256, 1)  # Scalar confidence
        )
    
    def forward(self, z):
        return self.head(z)

# Register in auxiliary heads
self.custom_confidence = CustomConfidenceHead(config)
```

### Multi-GPU Inference

**Distributed prediction:**
```python
# Split queries across GPUs
query_chunks = distribute_queries_across_gpus(queries, num_gpus)

# Run in parallel
results = []
for gpu_id, chunk in enumerate(query_chunks):
    result = run_on_gpu(chunk, device=gpu_id)
    results.append(result)

# Aggregate results
final_results = aggregate_predictions(results)
```

---

## References

### Key Papers

1. **AlphaFold3** (Nature, 2024) - Primary reference
2. **AlphaFold2** (Nature, 2021) - Evoformer architecture
3. **RoseTTAFold** (Science, 2021) - Three-track networks
4. **Diffusion Models** (Ho et al., 2020) - Denoising diffusion

### Code Resources

- **Main Entry Point:** `openfold3/run_openfold.py`
- **Model Definition:** `openfold3/projects/of3_all_atom/model.py`
- **Runner:** `openfold3/projects/of3_all_atom/runner.py`
- **Diffusion Module:** `openfold3/core/model/structure/diffusion_module.py`
- **PairFormer:** `openfold3/core/model/latent/pairformer.py`
- **Evoformer:** `openfold3/core/model/latent/evoformer.py`

### Documentation

- **Installation:** `docs/source/Installation.md`
- **Inference Guide:** `docs/source/Inference.md`
- **Input Format:** `docs/source/input_format.md`
- **Configuration Reference:** `docs/source/configuration_reference.md`

---

## Appendix: Configuration Examples

### Minimal Inference Config

```yaml
inference_ckpt_path: ~/.openfold3/openfold3_params.pt
shared:
  diffusion:
    no_full_rollout_samples: 5
settings:
  memory:
    eval:
      use_deepspeed_evo_attention: true
```

### Full Training Config

```yaml
experiment_settings:
  seed: 42
  ema:
    decay: 0.999
    
data_module_args:
  train:
    weighted-pdb:
      weight: 0.7
      config:
        crop:
          token_budget: 512
        template:
          n_templates: 4
          
model:
  architecture:
    input_embedder:
      c_s_input: 384
    msa:
      msa_module:
        no_blocks: 4
    pairformer:
      no_blocks: 48
    diffusion_module:
      diffusion_transformer:
        no_blocks: 11
        
loss:
  diffusion:
    weight: 1.0
  bond:
    weight: 1.0
  smooth_lddt:
    weight: 1.0
```

---

**End of Document**

*This document serves as a comprehensive reference for understanding OpenFold3-preview's software design and core algorithms. For implementation details, refer to the source code and inline documentation.*
