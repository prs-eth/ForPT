_base_ = ["../_base_/default_runtime.py"]

# misc settings
batch_size = 128  # total batch size across all GPUs
num_worker = 64
batch_size_val = 64
batch_size_test = 64
empty_cache = False
enable_amp = False

evaluate = True
wandb_project = "ForPT-forage"
save_path = "exp/04_forage/age-ins-pretrain-lin-p100"
data_root = "data/forage"
metadata_file = "FORage_tree_metadata_train_val.csv"  # official FORage train/val metadata
weight = "weights/02_forinstancev2_ins/insseg-litepts-scratch-p100/model/model_last.pth"

# model settings
model = dict(
    type="DefaultReggressorLinearProbeCat",
    num_targets=1,
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
    criteria=[dict(type="HuberLoss", reduction="mean", beta=20)],
    freeze_backbone=True,

)

# scheduler settings
epoch = 9000
eval_epoch = 300

optimizer = dict(type="AdamW",lr=0.0001,weight_decay=0.05)

scheduler = dict(
    type="PolyLR",
    total_steps=epoch,
    power=1.0,
)

param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "ForAgeDataset"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        metadata_file=metadata_file,
        pc_folder="pc",
        target_col="age",
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", always_apply=False, p=0.5),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "regression"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        metadata_file=metadata_file,
        pc_folder="pc",
        target_col="age",
        transform=[
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
                keys=("coord", "grid_coord", "regression"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
        inference_mode=True,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        metadata_file="FORage_tree_metadata_test.csv",  # hidden test set, no age
        pc_folder="pc",
        target_col="age",
        transform=[
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
                keys=("coord", "grid_coord", "regression"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
        inference_mode=True,
    ),
)

# hooks
hooks = [
    dict(
        type="ForestFormerCheckpointLoader",
        keywords="unet.",
        replacement="backbone.",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="RegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="RegTesterSubmit")
