import configparser


def load_config():
    import configs
    config = configparser.ConfigParser()
    config.read(configs.root+'/settings/config.ini')
    for section in config.sections():
        for options in config.options(section):
            if options == 'root':
                configs.root = config.get(section, options)
            elif options == 'algo':
                configs.algo = config.get(section, options)
            elif options == 'workload':
                configs.workload = config.get(section, options)
            elif options == 'beta':
                configs.beta = float(config.get(section, options))
            elif options == 'iteration':
                configs.iteration = int(config.get(section, options))
            elif options == 'fixed_episodic_reward':
                configs.fixed_episodic_reward = int(config.get(section, options))
            elif options == 'epsilon':
                configs.epsilon = float(config.get(section, options))
            elif options == 'learning_rate':
                configs.learning_rate = float(config.get(section, options))
            elif options == 'gamma':
                configs.gamma = float(config.get(section, options))
            elif options == 'placement_penalty':
                configs.placement_penalty = int(config.get(section, options))
            elif options == 'pp_apply':
                configs.pp_apply = config.get(section, options)
            else:
                print('Invalid Option found {}'.format(options))
