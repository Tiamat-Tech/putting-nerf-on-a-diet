class NerfConfig:
    # MODEL CONFIG
    model = "nerf"
    net_activation = "relu"
    rgb_activation = "sigmoid"
    sigma_activation = "relu"
    min_deg_point = 0
    max_deg_point = 10
    deg_view = 4
    # reduce sampling along ray for speedup
    num_coarse_samples = 32
    num_fine_samples = 64
    use_viewdirs = True
    near = 2
    far = 6
    noise_std = None
    # TODO @Alex: set white_bkgd as flag if we add LLFF dataset
    white_bkgd = True
    net_depth = 8
    net_width = 256
    net_depth_condition = 1
    net_width_condition = 128
    skip_layer = 4
    num_rgb_channels = 3
    num_sigma_channels = 1
    lindisp = True
    legacy_posenc_order = False
    randomized = True

    # DATA CONFIG
    W = 800
    H = 800
    IMAGE_SHAPE = (W, H, 3)
    # TODO @Alex: flexible focal if we add LLFF dataset
    FOCAL = 555.5555155968841
    CHUNK = 4096
    DOWNSAMPLE = 2
