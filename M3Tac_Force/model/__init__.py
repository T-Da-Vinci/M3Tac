def get_network(network_name, logger):
    if network_name == 'FSwin_MAP':
        from .FSwin_MAP.vision_transformer import FSwin_MAP
        from .FSwin_MAP.train import get_all_config
        args, config = get_all_config()
        net = FSwin_MAP(config, img_size=args.img_size, num_classes=args.num_classes)
        return net
