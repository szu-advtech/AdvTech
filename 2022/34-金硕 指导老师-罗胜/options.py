
MetaSense_ActivityOpt = {
    'name': 'metasense_activity',
    'batch_size': 64,
    'batch_size_target': 8,
    'seq_len': 256,
    'ecdf_bin_size': 50,
    'file_path': './dataset/ichar_minmax_scaling_all.csv',

    'input_dim': 6,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,

    ###---- 10 available domains for evaluation (full)
    'domains': ['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha', 'PH0038-iygoo', 'PH0041-hmkim', 'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim', 'WA4697-jhryu'],
    'domain_information': {
        'PH0007-jskim': {
            'sampling_rate': 100
        },
        'PH0012-thanh': {
            'sampling_rate': 200
        },
        'PH0014-wjlee': {
            'sampling_rate': 400
        },
        'PH0034-ykha': {
            'sampling_rate': 400
        },
        'PH0038-iygoo': {
            'sampling_rate': 500
        },
        'PH0041-hmkim': {
            'sampling_rate': 500
        },
        'PH0045-sjlee': {
            'sampling_rate': 200
        },
        'WA0002-bkkim': {
            'sampling_rate': 200
        },
        'WA0003-hskim': {
            'sampling_rate': 100
        },
        'WA4697-jhryu': {
            'sampling_rate': 100
        }
    },

    'classes': ['walking', 'running', 'sitting', 'standing', 'lying',
                'stairup', 'stairdown', 'jumping', 'stretching'],
    'num_class': 9
}



MetaSense_SpeechOpt = {
    'name': 'metasense_speech',
    'batch_size': 64,
    'batch_size_target': 8,
    'seq_len': 32000,
    'file_path': './dataset/icsr_minmax_scaling_all.csv',
    'input_dim': 1,
    'learning_rate': 0.001,
    'weight_decay': 0.001,

    ###---- 10 available domains for evaluation (full)
    'domains': ['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha', 'PH0038-iygoo', 'PH0041-hmkim', 'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim', 'WA4697-jhryu'],
    'domain_information': {
        'PH0007-jskim': {
            'sampling_rate': 100
        },
        'PH0012-thanh': {
            'sampling_rate': 200
        },
        'PH0014-wjlee': {
            'sampling_rate': 400
        },
        'PH0034-ykha': {
            'sampling_rate': 400
        },
        'PH0038-iygoo': {
            'sampling_rate': 500
        },
        'PH0041-hmkim': {
            'sampling_rate': 500
        },
        'PH0045-sjlee': {
            'sampling_rate': 200
        },
        'WA0002-bkkim': {
            'sampling_rate': 200
        },
        'WA0003-hskim': {
            'sampling_rate': 100
        },
        'WA4697-jhryu': {
            'sampling_rate': 100
        }
    },
    ###--- End

    'classes': ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off',
                'stop', 'go', 'forward',  'backward', 'follow', 'learn',],
    'num_class': 14
}

HHAROpt = {
    'name': 'hhar',
    'batch_size': 32,
    'batch_size_target': 8,
    'seq_len': 256,
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0.001,
    'file_path': './dataset/hhar_minmax_scaling_all.csv',

    ###---- 24 available domains for evaluation
    'users': ['a', 'b', 'c', 'd', 'e', 'f'],
    'models': ['nexus4', 's3', 's3mini', 'lgwatch'],
    'devices': ['lgwatch_1', 'lgwatch_2', 'gear_1', 'gear_2', 'nexus4_1', 'nexus4_2',
                's3_1', 's3_2', 's3mini_1', 's3mini_2'],

    'classes': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'],
    'num_class': 6,

}

DSAOpt = {
    'name': 'dsa',
    'batch_size': 64,
    'batch_size_target': 8,
    'seq_len': 125,
    'input_dim': 9,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'file_path': './dataset/dsa_minmax_scaling_all.csv',

    ###---- 40 available domains for evaluation (full)
    'users': ['p' + str(i) for i in range(1, 9)],
    'positions': ['T', 'RA', 'LA', 'RL', 'LL'],

    'activities': ['a0' + str(i) if (i < 10) else 'a' + str(i) for i in range(1, 20)],
    'classes': ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs',
                'descending_stairs', 'standing_in_elevator', 'moving_in_elevator', 'walking_in_parking_lot',
                'walking_on_treadmill_4_flat', 'walking_on_treadmill_4_inclined', 'running_on_treadmill_8',
                'exercising_on_stepper', 'exercising_on_cross_trainer', 'cycling_on_bike_horizontal',
                'cycling_on_bike_vertical', 'rowing', 'jumping', 'playing_basketball'],
    'num_class': 19
}
