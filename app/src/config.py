class NerfConfig:
    net_activation = "relu"
    rgb_activation = "sigmoid"
    sigma_activation = "relu"
    min_deg_point = 0
    max_deg_point = 10
    deg_view = 4
    num_coarse_samples = 64
    num_fine_samples = 128
    use_viewdirs = True
    near = 2
    far = 6
    noise_std = None
    # TODO @Alex: set white_bkgd as flag if we have LLFF dataset
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
