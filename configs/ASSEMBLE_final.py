gpus = (0, 1,)
log_dir = 'exp'
workers = 6
print_freq = 30
seed = 3035

network = dict(
    backbone="MocHRBackbone",
    sub_arch='hrnet48',
    counter_type = 'withMOE', #'withMOE' 'baseline'
    resolution_num = [0,1,2,3],
    loss_weight = [1., 1./2, 1./4, 1./8],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = False,
    pretrained_backbone="../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth",


    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=96,
        stages_channel = [384, 192, 96, 48],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='ASSEMBLE',
    root=['/home/cho092871/Desktop/Datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final',
          '/home/cho092871/Desktop/Datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
        #   '/home/cho092871/Desktop/Datasets/jhu_crowd_v2.0/jhu_crowd_v2.0',
          '/home/cho092871/Desktop/Datasets/UCF-QNRF/UCF-QNRF_ECCV18'],
    test_set='test.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc_x2.txt',
    num_classes= len(network['resolution_num']),
    den_factor=100,
    extra_train_set =None
)


optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-4,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
    )


lr_config = dict(
    NAME='None',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=250,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS=0,   # the number of epochs to warmup the lr_rate
    WARMUP_LR=5.0e-07,
    MIN_LR= 1.0e-07
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(768, 768),  # height width
    route_size=(256, 256),  # height, width
    base_size=None,
    batch_size_per_gpu=6,
    shuffle=True,
    begin_epoch=0,
    end_epoch= 50,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path= "/home/cho092871/Desktop/Networks/STEERER/exp/ASSEMBLE/MocHRBackbone_hrnet48/ASSEMBLE_final_2024-10-04-14-26",  # "/home/cho092871/Desktop/Networks/STEERER/exp/ASSEMBLE/MocHRBackbone_hrnet48/ASSEMBLE_final_2024-10-01-22-34",
    pretrained = None, # "/home/cho092871/Desktop/Networks/STEERER/exp/ASSEMBLE/MocHRBackbone_hrnet48/ASSEMBLE_final_2024-09-20-17-09/Ep_441_mae_59.899008520254604_mse_227.68671281631933.pth",
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span =   [-800, -600, -400, -200, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)

test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=3072,
    loc_base_size=3072,
    loc_threshold = 0.15,
    batch_size_per_gpu=1,
    patch_batch_size=2,
    flip_test=False,
    multi_scale=False,
    model_file = './exp/QNRF/MocHRBackbone_hrnet48/QNRF_final_2023-10-28-13-44/QNRF_mae_78.4_mse_135.6.pth'
    # model_file = './exp/QNRF/MocHRBackbone_hrnet48/QNRF_HR_2022-10-21-01-51/Ep_359_mae_75.40686944287694_mse_134.
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


