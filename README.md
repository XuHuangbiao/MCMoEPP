# MCMoE++: Incomplete Multimodal Action Quality Assessment via Modality Completion with Mixture of Experts

This is the code for our "Incomplete Multimodal Action Quality Assessment via Modality Completion with Mixture of Experts (MCMoE++)".

## Environments

- RTX 3090
- CUDA: 12.2
- Python: 3.9.23
- PyTorch: 1.12.0+cu113

## Features

The features (RGB, Audio, Flow) and label files of Rhythmic Gymnastics and Fis-V dataset can be downloaded from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.

The features (RGB, Audio) and label files of FS1000 dataset can be downloaded from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository. We adopt the same frame sampling method to extract Optical Flow features from the FS1000 dataset, which can be downloaded via this [link](https://1drv.ms/f/c/056e0e22eb875f5c/IgAq1rrVu9n_Rqd2CaYr00ADAcpMRjh4Tf_gmX3yUsYIFEc?e=g9rFJf).

The CMU-MOSI and CMU-MOSEI features can be downloaded from the [GCNet](https://github.com/zeroQiaoba/GCNet) repository.

## Pretrained Model

The pretrained weights can be downloaded from [here](https://pan.baidu.com/s/1beMiFjbrBZmsGsaQ9ZYLXw?pwd=c456) (extraction code: c456).

## Running

### The following are examples only, more details coming soon!

Please fill in or select the args enclosed by {} first.

On the **FS1000 / Fis-V / RG** datasets, run the following commands from `AQA/`. `--action-type` is selected from `TES/PCS/SS/TR/PE/CO/IN` for FS1000, `TES/PCS` for Fis-V, and `Ball/Clubs/Hoop/Ribbon` for RG.

- Training
```typescript
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset {FS1000/FisV/RG} --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {action type} --clip-num {clip number} --lr {learning rate} --epoch {epoch} --in_dim {input dimension} --n_head {number of heads} --n_encoder {number of encoder layers} --n_decoder {number of decoder layers} --n_query {number of queries} --use_pe {use positional encoding} --alpha {alpha} --margin {margin} --lr-decay {lr decay scheduler} --decay-rate {decay rate} --dropout {dropout} --seed {seed} --checkpoint-selection {task rule} [--selection-epoch {zero-based epoch}]

```

For AQA training, set `--checkpoint-selection` to the task's reported rule below. `vaf_coef` and `vaf_mse` select by complete-input SRCC and MSE, respectively, on the supplied test split. `fixed_epoch` saves the specified zero-based epoch with `--selection-epoch`. The default `vaf_coef` applies only when no rule is supplied.

| Dataset | Action | Checkpoint selection |
| --- | --- | --- |
| FS1000 | TES / PCS / SS / TR / PE / IN | `fixed_epoch`: 132 / 344 / 280 / 191 / 347 / 302, respectively |
| FS1000 | CO | `vaf_mse` |
| FisV | TES / PCS | `fixed_epoch`: 162 / 505, respectively |
| RG | Ball / Clubs / Hoop | `vaf_coef` |
| RG | Ribbon | `vaf_mse` |

- Testing
```typescript
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset {FS1000/FisV/RG} --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {action type} --clip-num {clip number} --in_dim {input dimension} --n_head {number of heads} --n_encoder {number of encoder layers} --n_decoder {number of decoder layers} --n_query {number of queries} --use_pe {use positional encoding} [--missing-ratio {0.0-0.7}] --test --ckpt {the name of the used checkpoint}

```

Without `--missing-ratio`, modality-level testing is performed on VAF and all six incomplete combinations. With `--missing-ratio`, proportional missingness testing is performed using the released JSON files in `AQA/masks/`.

On the **CMU-MOSI / CMU-MOSEI** datasets, run the following commands from `Sentiment/`:

- Training
```typescript
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset {MOSI/MOSEI} --text-path {path of deberta-large-4-UTT features} --video-path {path of manet_UTT features} --audio-path {path of wav2vec-large-c-UTT features} --label-path {path of MOSI/MOSEI label file} --missing-rate {0.0-0.7} --model-name {the name used to save model and log} --lr {learning rate} --epoch {epoch} --in_dim {input dimension} --hidden_dim {hidden dimension} --n_head {number of heads} --n_encoder {number of encoder layers} --n_decoder {number of decoder layers} --n_query {number of queries} --use_pe {use positional encoding} --alpha {alpha} --margin {margin} --dropout {dropout} --seed {seed}

```

- Testing
```typescript
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset {MOSI/MOSEI} --text-path {path of deberta-large-4-UTT features} --video-path {path of manet_UTT features} --audio-path {path of wav2vec-large-c-UTT features} --label-path {path of MOSI/MOSEI label file} --missing-rate {0.0-0.7} --model-name {the name used to save model and log} --in_dim {input dimension} --hidden_dim {hidden dimension} --n_head {number of heads} --n_encoder {number of encoder layers} --n_decoder {number of decoder layers} --n_query {number of queries} --use_pe {use positional encoding} --seed {seed} --test --ckpt {the name of the used checkpoint}

```

Sentiment testing retains the online utterance-level missingness protocol. Bare state-dict checkpoints require their validation-selected epoch to reproduce the corresponding test mask.

## Acknowledgements

This repository builds upon [MCMoE (AAAI 2026)](https://github.com/XuHuangbiao/MCMoE).

We thank the authors for their contributions to the research community.
