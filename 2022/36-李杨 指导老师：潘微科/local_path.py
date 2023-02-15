import os

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(base_dir, 'data')

ml_dir = os.path.join(data_dir, 'ml-100k')

result_path = os.path.join(base_dir, 'results')
