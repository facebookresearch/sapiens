# for batch in each gpu is 256, 8 gpu
# lr = 5e-4 * 256 * 8 / 512 = 0.002
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.002,
        weight_decay=0.025,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.attention_biases': dict(decay_mult=0.0),
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-6 / 0.002,
        by_epoch=True,
        end=5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=5)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1000)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
