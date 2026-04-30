# Unified Transition Diffusion Model for Robotic Planning

## 1. Overview

This project implements a **Unified Transition Diffusion Model**, a sophisticated generative model designed for robotic motion planning. It is based on the cutting-edge ideas presented in **Unified World Models** (UWM) [1] and **CompDiffuser** [2].

The model learns the joint distribution of `(initial_config, final_config)` transition pairs. This unified approach allows for highly flexible conditional generation and, most importantly, enables **chain planning** by stitching multiple transitions together to form a coherent, long-horizon motion plan.

### Key Features:

-   **Unified Architecture**: A single model handles joint generation, conditional generation (forward/backward dynamics), and infilling.
-   **Independent Timesteps (UWM)**: The model is conditioned on two independent diffusion timesteps, `t_init` and `t_final`, allowing it to learn the full relationship between the start and end of a transition.
-   **Chain Planning (CompDiffuser)**: Implements a "constrained denoising" algorithm to generate a sequence of connected transitions (e.g., `A -> B -> C -> D`) by enforcing equality constraints at the connection points during the sampling process.
-   **Flexible Data Representation**: The data loader now correctly handles the provided 28D pose format (7D per effector) and converts it to the model's internal 22D representation (6D for feet, 3D for hands, plus contacts).

## 2. Project Structure

```
/unified
├── configs/              # Model and training configurations
│   └── unified.yaml
├── data/                 # Dataset loading and processing
│   └── dataset.py
├── models/               # Core model components
│   ├── denoiser.py       # UWM-style Transformer denoiser
│   ├── diffusion.py      # Main diffusion model logic (training, sampling, chaining)
│   └── noise_schedule.py
├── scripts/              # Executable scripts
│   ├── train.py          # Main training script
│   ├── sample.py         # Script for sampling and planning
│   └── test_e2e.py       # End-to-end test suite
├── visualization/        # Visualization tools
│   └── visualize.py
├── outputs/              # Default directory for generated samples and plots
└── checkpoints/          # Default directory for model checkpoints
```

## 3. How It Works

### 3.1. The Unified Model (UWM)

The core of the project is a Transformer-based denoiser that takes a noisy `initial` config and a noisy `final` config as input. Crucially, it is also conditioned on their respective, independently sampled timesteps, `t_init` and `t_final`. By training on all possible combinations of `t_init` and `t_final`, the model learns the entire joint distribution `p(initial, final)`.

This allows for powerful inference modes:
-   **Forward Dynamics `p(final | initial)`**: Set `t_init = 0` (clean initial) and denoise `final` from `t_final = T-1` to `0`.
-   **Backward Dynamics `p(initial | final)`**: Set `t_final = 0` and denoise `initial`.
-   **Joint Generation `p(initial, final)`**: Denoise both from `t = T-1` to `0`.

### 3.2. Chain Planning (CompDiffuser)

To generate a plan `A -> B -> C -> D`, we set up three transition "slots": `(A, B)`, `(B, C)`, and `(C, D)`. We fix `A` and `D` and initialize `B` and `C` with pure noise. Then, we denoise all slots simultaneously. After each denoising step, we enforce the chain constraint by averaging the states at the connection points:

`B_new = (B_from_first_slot + B_from_second_slot) / 2`

This "constrained denoising" process ensures that the generated waypoints `B` and `C` form a continuous path from `A` to `D`.

## 4. Usage

### 4.1. Training

First, ensure you have a dataset in the specified JSON format (see `data/dataset.py` for details). Then, run the training script:

```bash
# Navigate to the unified directory
cd /home/ubuntu/robot_pose_diffusion/unified

# Start training
python scripts/train.py \
    --config configs/unified_2080ti.yaml \
    --data /home/crc/jimmy/unified_module/unified/dataset/seiko_talos_merged.json \
    --output_dir checkpoints/test0410
```

-   The script will automatically save checkpoints and TensorBoard logs to the `output_dir`.
-   The best model (based on validation loss) will be saved as `checkpoint_best.pt`.

### 4.2. Sampling and Planning

Use the `sample.py` script for all generation tasks. It has several modes:

**A. Joint Generation (Unconditional)**

Generate random, valid `(initial, final)` transitions.

```bash
python scripts/sample.py \
    --checkpoint checkpoints/test0316/checkpoint_best.pt \
    --mode joint \
    --num_samples 30 \
    --output outputs/joint_samples0316tienkung.json
```

**B. Conditional Generation (Forward Dynamics)**

Given a specific initial state, generate possible final states.

```bash
# The condition must be a 22-dimensional comma-separated string
# (LF_6D, RF_6D, LH_3D, RH_3D, Contacts_4D)
CONDITION="0.0,0.1,0,0,0,0, 0.0,-0.1,0,0,0,0, 0.5,0.5,1, -0.5,0.5,1, 1,1,0,0"

python scripts/sample.py \
    --checkpoint checkpoints/my_run/checkpoint_best.pt \
    --mode forward \
    --condition_initial "$CONDITION" \
    --num_samples 16 \
    --guidance_scale 2.5 \
    --output outputs/forward_samples.json
```

**C. Chain Planning**

Generate a sequence of intermediate waypoints between a start and a goal state.

```bash
START_STATE="...your 22D start state..."
GOAL_STATE="...your 22D goal state..."

python scripts/sample.py \
    --checkpoint checkpoints/0428_v2/checkpoint_best.pt\
    --mode chain \
    --start="1.45267e-11,3.01423e-12,3.30277e-13,1.45525e-11,-0.17,-1.57152e-13,0.415559,0.264454,0.971211,0.425521,-0.424698,0.924157,1,1,0,0" \
    --goal="0.234402,-0.000106034,0.000132038,0.219698,-0.257853,0.00024149,0.688516,0.263693,0.971066,0.582762,-0.425193,0.923607,1,1,0,0" \
    --num_transitions 6 \
    --num_samples 8 \
    --guidance_scale 4.0 \
    --output outputs/seiko_chain_0428_test1.json
```
```bash
python scripts/sample.py \
    --checkpoint checkpoints/0428_v2/checkpoint_best.pt\
    --mode chain \
    --start="-7.95837e-17, -6.65266e-18,  6.93889e-17,-6.07421e-17,-0.17, 2.35922e-16,0.415559 ,0.264454, 0.971211,0.425521,-0.424698,0.924157,1,1,0,0" \
    --goal="0.346625,-0.000102918,0.0111614 ,1.27825e-13, -0.17,2.45577e-12, 0.62678,0.26376,0.972921, 0.821735,-0.557251, 0.926407, 1, 1, 1 ,1" \
    --num_transitions 6 \
    --num_samples 30 \
    --guidance_scale 4.0 \
    --output outputs/seiko_chain_0428_test5.json

```
```bash
python scripts/sample.py \
    --checkpoint checkpoints/0429_gated/checkpoint_best.pt\
    --mode chain \
    --start="-7.95837e-17, -6.65266e-18,  6.93889e-17,-6.07421e-17,-0.17, 2.35922e-16,0.415559 ,0.264454, 0.971211,0.425521,-0.424698,0.924157,1,1,0,0" \
    --goal="0.268628,0.141555,0.0002,0.311573,-0.169977,0.00018,0.8808, 0.264683, 0.971488,0.63043, -0.42424, 0.925591, 1, 1, 0,1" \
    --num_transitions 8 \
    --num_samples 50 \
    --guidance_scale 4.0 \
    --output outputs/seiko_chain_0428_test9.json

```




cd /home/crc/jimmy/unified_module/unified && python scripts/sample.py     
--checkpoint checkpoints/seiko_talos_3000_0325/checkpoint_best.pt     --mode chain     --start="1.45267e-11,3.01423e-12,3.30277e-13,1.91885e-08,-8.3791e-08,1.63046e-08,1.45525e-11,-0.17,-1.57152e-13,1.91885e-08,-8.3791e-08,1.62907e-08,0.415559,0.264454,0.971211,0.425521,-0.424698,0.924157,1,1,0,0"     --goal="0.234402,-0.000106034,0.000132038,0.000253223,-0.000362294,2.2021e-05,0.219698,-0.257853,0.00024149,-0.000319312,-6.65064e-05,-3.21198e-05,0.688516,0.263693,0.971066,0.582762,-0.425193,0.923607,1,1,0,0"     --num_transitions 4     --num_samples 8     --chain_mode parallel     --guidance_scale 4.0     --output outputs/seiko_chain_ee_test3.json
This will generate a plan with `3 - 1 = 2` intermediate waypoints. The script saves the best chain found, as well as all `num_samples` candidate chains for analysis.

## 5. End-to-End Test

A comprehensive test suite is included to verify all components of the model. It uses the provided dataset to run through data loading, training, all sampling modes, and visualization.

```bash
cd /home/ubuntu/robot_pose_diffusion/unified
python scripts/test_e2e.py
```

This is an excellent way to understand the full capabilities of the project. All generated plots will be saved in the `outputs/` directory.

## 6. References

[1] Zhu, C., et al. (2025). *Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets*. arXiv:2504.02792.

[2] Luo, Y., et al. (2025). *Generative Trajectory Stitching through Diffusion Composition*. arXiv:2503.05153.
