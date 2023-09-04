_base_ = [
    '../_base_/models/cascade_mask_rcnn_dat_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = '<path-to-pretrained-model>'

model = dict(
    backbone=dict(
        type='DAT',
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 4, 18, 2],
        stage_spec=[
            ["N", "D"], 
            ["N", "D", "N", "D"], 
            ["N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D"], 
            ["D", "D"]],
        heads=[3, 6, 12, 24],
        groups=[1, 2, 3, 6],
        use_pes=[True, True, True, True],
        strides=[8, 4, 2, 1],
        offset_range_factor=[-1, -1, -1, -1],
        use_dwc_mlps=[True, True, True, True],
        use_lpus=[True, True, True, True],
        use_conv_patches=True,
        ksizes=[9, 7, 5, 3],
        nat_ksizes=[7, 7, 7, 7],
        drop_path_rate=0.5,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

lr = 4e-4
bsz_per_gpu = 2
n_workers = 2
data = dict(train=dict(pipeline=train_pipeline), samples_per_gpu=bsz_per_gpu, workers_per_gpu=n_workers)

optimizer = dict(_delete_=True, type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(norm_decay_mult=0.,
                     custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 
                                  'relative_position_bias_table': dict(decay_mult=0.),
                                  'rpe_table': dict(decay_mult=0.),
                                  'norm': dict(decay_mult=0.)
                                 }
                                   )
                )
fp16 = None
optimizer_config = dict(
    type='OptimizerHook'
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunnerHF', max_epochs=36)
log_config = dict(interval=50)
