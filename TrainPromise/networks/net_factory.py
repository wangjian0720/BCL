from networks.unet import UNet, UNet_C_Drop

def net_factory(net_type="unet", in_chns=1, class_num=3):
    
    # Unet
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_unimatch_v2":
        net = UNet_C_Drop(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net



