_base_ = ["../_base_/default_runtime.py"]

# misc settings
batch_size = 128  # total batch size across all GPUs
num_worker = 64
batch_size_val = 32
empty_cache = False
enable_amp = False

wandb_project = "ForPT-forspecies20k"
evaluate = True
save_path = "exp/03_forspecies20k/cls-forpt-lin-p020"
weight = "weights/00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth"

# model settings
model = dict(
    type="DefaultClassifierLinearProbeCat",
    num_classes=33,
    backbone_embed_dim=1008,
    backbone=dict(
        type="LitePT-v1m2",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=True,
)

# scheduler settings
epoch = 300
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "ForSpecies20KDataset"
data_root = "data/forspecies20k/processed_8192"
cache_data = False
class_names = [
    'Abies_alba',
    'Acer_campestre',
    'Acer_pseudoplatanus',
    'Acer_saccharum',
    'Betula_pendula',
    'Carpinus_betulus',
    'Corylus_avellana',
    'Crataegus_monogyna',
    'Eucalyptus_miniata',
    'Euonymus_europaeus',
    'Fagus_sylvatica',
    'Fraxinus_angustifolia',
    'Fraxinus_excelsior',
    'Larix_decidua',
    'Picea_abies',
    'Picea_glauca',
    'Pinus_contorta',
    'Pinus_nigra',
    'Pinus_pinaster',
    'Pinus_radiata',
    'Pinus_resinosa',
    'Pinus_sylvestris',
    'Populus_deltoides',
    'Populus_tremuloides',
    'Prunus_avium',
    'Pseudotsuga_menziesii',
    'Quercus_faginea',
    'Quercus_ilex',
    'Quercus_petraea',
    'Quercus_robur',
    'Quercus_rubra',
    'Tilia_cordata',
    'Ulmus_laevis'
    ]

data = dict(
    num_classes=33,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train_20p",
        data_root=data_root,
        class_names=class_names,
        num_points=None,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        num_points=None,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        num_points=None,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=True,
    ),
)

# hooks
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="ClsTesterSubmit")
