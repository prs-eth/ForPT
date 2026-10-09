<div align="center">
<h1>Toward A Foundation Model for Forest Point Clouds</h1>

[**Yuanwen&nbsp;Yue**](https://ywyue.github.io/), [**Stefano&nbsp;Puliti**](https://scholar.google.com/citations?user=mrysxWoAAAAJ)<sup>†</sup>, [**Damien&nbsp;Robert**](https://drprojects.github.io/)<sup>†</sup>, [**Atakan&nbsp;Topaloğlu**](https://atakan-topaloglu.github.io/), [**Binbin&nbsp;Xiang**](https://scholar.google.com/citations?user=12YyiiIAAAAJ), [**Maciej&nbsp;Wielgosz**](https://scholar.google.com/citations?user=j563WT8AAAAJ), [**Jan&nbsp;Dirk&nbsp;Wegner**](https://scholar.google.com/citations?user=sxLG1rgAAAAJ&hl=en), [**Rasmus&nbsp;Astrup**](https://scholar.google.com/citations?user=8agrzhcAAAAJ), [**Christian&nbsp;Rupprecht**](https://chrirupp.github.io/),
[**Konrad&nbsp;Schindler**](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)

ETH&nbsp;Zurich,
University&nbsp;of&nbsp;Oxford,
Norwegian&nbsp;Institute&nbsp;of&nbsp;Bioeconomy&nbsp;Research&nbsp;(NIBIO),
University&nbsp;of&nbsp;Zurich

<sup>†</sup>Equal contribution

<a href="https://arxiv.org/abs/2609.24787"><img src="https://img.shields.io/badge/arXiv-ForPT-red" alt="Paper PDF"></a>
<a href="https://prs-eth.github.io/ForPT"><img src="https://img.shields.io/badge/Project_Page-ForPT-green" alt="Project Page"></a>
<a href='https://huggingface.co/prs-eth/ForPT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>

<img
  src="assets/pca.png"
  alt="teaser"
  width="800"
/>

</div>

**ForPT** is a self-supervised pretrained point cloud model for forests. We pretrain a [LitePT](https://github.com/prs-eth/LitePT) backbone on a large corpus of unlabeled airborne (ALS), UAV (ULS) and mobile (MLS) laser scans covering diverse forest ecosystems, and evaluate it on four forestry tasks: semantic segmentation, instance segmentation, tree species classification and tree age regression. Self-supervised pretraining accelerates convergence, improves performance when annotations are scarce, and transfers better than task-specific supervised pretraining.

## Preparation

### Environment
ForPT relies on [LitePT](https://github.com/prs-eth/LitePT). Follow the [instructions in the LitePT repository](https://github.com/prs-eth/LitePT#environment) to install the environment for LitePT. Then install the few additional packages used by ForPT:
```shell
pip install pandas matplotlib seaborn tqdm
```

### Data
All data should be placed in `ForPT/data`, organized as expected by the configs:

```
ForPT/data/
├── forpt_ssl_corpus/              # unlabeled pretraining corpus (ALS / ULS / MLS)
│   └── knn_track/                 # labeled subset for kNN evaluation during pretraining
├── forinstancev2_sem/             # FOR-instanceV2 (semantic segmentation)
│   ├── train/ val/ test/
│   └── train_{5,10,20,50}.txt     # label-efficient training subsets
├── forspecies20k/                 # FOR-species20K (species classification)
│   ├── processed/                 # full-resolution trees: <species>/<tree>.txt (x,y,z), test/, train/val/test splits
│   └── processed_8192/            # same trees downsampled to 8,192 points, plus train_{1,5,10,20,50}p.txt subsets
└── forage/                        # tree age regression
    ├── FORage_tree_metadata_train_val.csv  # official FORage train/val metadata
    ├── FORage_tree_metadata_test.csv       # hidden test set: sampleID, file (no age)
    ├── pc/<name>/coord.npy        # one point cloud per tree (test: <name> = sampleID)
    ├── forage_{train,val,test}_8192points_uniform.pth  # cached 8,192-point samples
    └── train_{5,10,20,50}p.txt    # label-efficient training subsets
```

We provide the preprocessed datasets on Hugging Face:

<!-- TODO: ForPT corpus and FOR-instanceV2 instance segmentation download links -->
| Dataset | Task | Download | Local folder |
|:-|:-|:-:|:-|
| ForPT corpus | Self-supervised pretraining | - | `forpt_ssl_corpus` |
| FOR-instanceV2 | Semantic segmentation | [prs-eth/forpt-forinstancev2-sem](https://huggingface.co/datasets/prs-eth/forpt-forinstancev2-sem) | `forinstancev2_sem` |
| | Instance segmentation | - | `forinstancev2_ins` |
| FOR-species20K | Species classification | [prs-eth/forpt-forspecies20k](https://huggingface.co/datasets/prs-eth/forpt-forspecies20k) | `forspecies20k` |
| FORage | Age regression | [prs-eth/forpt-forage](https://huggingface.co/datasets/prs-eth/forpt-forage) | `forage` |

Point clouds are packed into `.tar` files, while split lists, metadata CSVs and the FORage `.pth` caches are stored as-is. To download everything and unpack it into the layout above, run the following from the `ForPT` root (requires `pip install -U huggingface_hub`):

```shell
mkdir -p data && cd data

# FOR-instanceV2 (~14 GB): train.tar, val.tar, test.tar
hf download prs-eth/forpt-forinstancev2-sem --repo-type dataset --local-dir forinstancev2_sem
# FOR-species20K (~168 GB): one tar per species folder in processed/ and processed_8192/
hf download prs-eth/forpt-forspecies20k --repo-type dataset --local-dir forspecies20k
# FORage (~13 GB): pc.tar
hf download prs-eth/forpt-forage --repo-type dataset --local-dir forage

# Unpack every tar next to itself, then delete the archive
find forinstancev2_sem forspecies20k forage -name "*.tar" | while read f; do
    tar -xf "$f" -C "$(dirname "$f")" && rm "$f"
done
rm -rf {forinstancev2_sem,forspecies20k,forage}/{.cache,.gitattributes}
```

FOR-species20K is the largest download. The full-resolution `processed/` folder is only used by the `*-allpts` configs. If you only need the 8,192-point version used by all other configs, add `--include "processed_8192/*"` to its download command. Interrupted downloads resume when you rerun the same command.

## Model Zoo

### Pretrained backbone
| Model | Params | Pretraining data | Config | Checkpoint |
|:-|:-:|:-:|:-:|:-:|
| ForPT (LitePT-S) | 12.4M | ForPT corpus | [link](configs/00_ssl_pretrain/pretrain-forpt.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth) |

### Downstream tasks (100% labels)
| Task | Benchmark | Model | Config | Checkpoint |
|:-|:-:|:-|:-:|:-:|
| Semantic segmentation | FOR-instanceV2 | LitePT-scratch | [link](configs/01_forinstancev2_sem/semseg-litepts-scratch-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/01_forinstancev2_sem/semseg-litepts-scratch-p100/model/model_best.pth) |
| | | ForPT-linear probing | [link](configs/01_forinstancev2_sem/semseg-forpt-lin-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/01_forinstancev2_sem/semseg-forpt-lin-p100/model/model_best.pth) |
| | | ForPT-decoder probing | [link](configs/01_forinstancev2_sem/semseg-forpt-dec-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/01_forinstancev2_sem/semseg-forpt-dec-p100/model/model_best.pth) |
| | | ForPT-finetune | [link](configs/01_forinstancev2_sem/semseg-forpt-ft-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/01_forinstancev2_sem/semseg-forpt-ft-p100/model/model_best.pth) |
| Instance segmentation | FOR-instanceV2 | Coming soon | - | - |
| Species classification | FOR-species20K | LitePT-scratch | [link](configs/03_forspecies20k/cls-litepts-scratch-allpts.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/03_forspecies20k/cls-litepts-scratch-allpts/model/model_last.pth) |
| | | ForPT-linear probing | [link](configs/03_forspecies20k/cls-forpt-lin-allpts.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/03_forspecies20k/cls-forpt-lin-allpts/model/model_last.pth) |
| | | ForPT-finetune | [link](configs/03_forspecies20k/cls-forpt-ft-allpts.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/03_forspecies20k/cls-forpt-ft-allpts/model/model_last.pth) |
| Age regression | FORage | LitePT-scratch | [link](configs/04_forage/age-litepts-scratch-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/04_forage/age-litepts-scratch-p100/model/model_best.pth) |
| | | ForPT-linear probing | [link](configs/04_forage/age-forpt-lin-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/04_forage/age-forpt-lin-p100/model/model_best.pth) |
| | | ForPT-finetune | [link](configs/04_forage/age-forpt-ft-p100.py) | [Download](https://huggingface.co/prs-eth/ForPT/resolve/main/04_forage/age-forpt-ft-p100/model/model_best.pth) |

All checkpoints are hosted on [prs-eth/ForPT](https://huggingface.co/prs-eth/ForPT) under the `<task>/<config name>/model/` layout. To download them into `ForPT/weights`, where the configs expect them (e.g. `weights/00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth`), run from the `ForPT` root:

```shell
hf download prs-eth/ForPT --local-dir weights --exclude ".gitattributes" --exclude "README.md"
```

## Training

Configs are grouped by task under `configs/`. The `-d` flag of `scripts/train.sh` selects the config folder, and experiments are saved to `exp/${FOLDER}/${EXP_NAME}`.

Config names follow the pattern `{task}-{init}-{protocol}-{labels}`:
- **Initialization and protocol**
  - `litepts-scratch`: LitePT-S trained from scratch (supervised baseline)
  - `forpt-lin`: linear probing on frozen ForPT features
  - `forpt-dec`: initialize from ForPT, freeze the encoder and train the decoder
  - `forpt-ft`: initialize from ForPT and fine-tune the full model
  - `{sem,ins,species,age}-pretrain-lin`: linear probing on a backbone pretrained with supervision on another task
- **Labels**: `p001`, `p005`, `p010`, `p020`, `p050` and `p100` train on 1%–100% of the training labels. `allpts` (species only) trains on train+val at full point density.

### Self-supervised pretraining
We pretrain on 16 nodes with 4 GPUs each. On a SLURM cluster, launch the script once per node; the node rank and master address are read from SLURM:
```shell
srun --nodes=16 --ntasks-per-node=1 sh scripts/train.sh -g 4 -m 16 -d 00_ssl_pretrain -c pretrain-forpt -n pretrain-forpt
```
`batch_size` in the configs is the total across all GPUs, so running on fewer GPUs increases the batch per GPU.

All downstream experiments use a single node with 4 GPUs.

### Semantic segmentation (FOR-instanceV2)
```shell
# LitePT-scratch
sh scripts/train.sh -g 4 -d 01_forinstancev2_sem -c semseg-litepts-scratch-p100 -n semseg-litepts-scratch-p100
# ForPT-linear probing
sh scripts/train.sh -g 4 -d 01_forinstancev2_sem -c semseg-forpt-lin-p100 -n semseg-forpt-lin-p100
# ForPT-decoder probing
sh scripts/train.sh -g 4 -d 01_forinstancev2_sem -c semseg-forpt-dec-p100 -n semseg-forpt-dec-p100
# ForPT-finetune
sh scripts/train.sh -g 4 -d 01_forinstancev2_sem -c semseg-forpt-ft-p100 -n semseg-forpt-ft-p100
```

### Species classification (FOR-species20K)
```shell
# LitePT-scratch
sh scripts/train.sh -g 4 -d 03_forspecies20k -c cls-litepts-scratch-allpts -n cls-litepts-scratch-allpts
# ForPT-linear probing
sh scripts/train.sh -g 4 -d 03_forspecies20k -c cls-forpt-lin-allpts -n cls-forpt-lin-allpts
# ForPT-finetune
sh scripts/train.sh -g 4 -d 03_forspecies20k -c cls-forpt-ft-allpts -n cls-forpt-ft-allpts
```

### Age regression (FORage)
```shell
# LitePT-scratch
sh scripts/train.sh -g 4 -d 04_forage -c age-litepts-scratch-p100 -n age-litepts-scratch-p100
# ForPT-linear probing
sh scripts/train.sh -g 4 -d 04_forage -c age-forpt-lin-p100 -n age-forpt-lin-p100
# ForPT-finetune
sh scripts/train.sh -g 4 -d 04_forage -c age-forpt-ft-p100 -n age-forpt-ft-p100
```

To run another protocol or label fraction, change the config name, e.g. `-c semseg-forpt-lin-p010` for linear probing with 10% of the labels.


## Checklist
- [x] Release pretrained models.
- [x] Release pretraining code and configs.
- [x] Release semantic segmentation, species classification and age regression code.
- [ ] Release instance segmentation code.


## Acknowledgment
This work uses parts of the code from [LitePT](https://github.com/prs-eth/LitePT), [Pointcept](https://github.com/Pointcept/Pointcept), [Sonata](https://github.com/facebookresearch/sonata), [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D), and [FORage](https://github.com/SingleTree-EU/FORage). We thank the authors for making their code publicly available.


## Citation

If you find this project useful, please cite:
```
@article{yue2026forpt,
    title={Toward a foundation model for forest point clouds},
    author={Yue, Yuanwen and Puliti, Stefano and Robert, Damien and Topalo{\u{g}}lu, Atakan and Xiang, Binbin and Wielgosz, Maciej and Wegner, Jan Dirk and Astrup, Rasmus and Rupprecht, Christian and Schindler, Konrad},
    journal={arXiv preprint arXiv:2609.24787},
    year={2026}
}
```
