"""
ForPT self-supervised pretraining.
"""

_base_ = ["../_base_/default_runtime.py"]
# misc settings
batch_size = 128  # total batch size across all GPUs
num_worker = 128
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
knn_evaluate_interval = 10
find_unused_parameters = False
save_path = "exp/00_ssl_pretrain/pretrain-forpt"

aux = False
# model settings
model = dict(
    type="Sonata-v1m1",
    # backbone - student & teacher
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
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=900,
    head_hidden_channels=3376,
    head_embed_channels=214,
    head_num_prototypes=3376,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.05,
    mask_size_base=0.05,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.7,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.07,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# scheduler settings
epoch = 30
eval_epoch = 30
base_lr = 0.001
base_wd = 0.0001

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)

scheduler = dict(type="MultiStepLR", milestones=[1], gamma=1)

# dataset settings
transform = [
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.05}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", ),
        local_feat_keys=("local_coord", ),
    ),
]

knn_transform = [
         dict(type="CenterShift", apply_z=True),
         dict(
             type="GridSample",
             grid_size=0.05,
             hash_type="fnv",
             mode="train",
             return_grid_coord=True,
             return_inverse=True,
         ),
         dict(type="ToTensor"),
         dict(
             type="Collect",
             keys=("coord", "grid_coord",  "inverse"),
             feat_keys=("coord", ),
         ),
     ]

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=[
        "ground",
        "wood",
        "leaf",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="DefaultDataset",
                split=["train", "val", "test"],
                data_root="data/forpt_ssl_corpus",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
        ],
    )
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="KNNSegEvaluator_forest",
        data_root="data/forpt_ssl_corpus/knn_track",
        write_cls_iou=True,
    ),
    dict(type="CheckpointSaver", save_freq=1),
]
