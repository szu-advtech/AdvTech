# Config file for tydiqa h5df dataset

# h5df
# |-- features: +-None, 3, 2048;  `input_ids, input_mask, segment_ids'
# |-- labels: None, 3; `start_pos, end_pos, answer_type`
# |-- metas: None, 3; ``unique_id, example_id, language_id`
# |-- offsets: None, 2, 2048; `start_offset, end_offset`

# define h5df group name
feature_group_name = 'features'
label_group_name = 'labels'
meta_group_name = 'metas'
offset_group_name = 'offsets'

# define shapes of h5 datasets
data_shapes = {feature_group_name: [1, 3, 2048],
               label_group_name: [1, 3],
               meta_group_name: [1, 3],
               offset_group_name: [1, 2, 2048]}


def save_h5_data(df, data, dset_name):
    """
    Append a block of data into h5df[dset_name] group if existed, create otherwise.
    Args:
        df: h5df File Object.
        data (List[]): list of data with shape `[none, m]`
        dset_name (str): naem of the group in h5df.
    """
    shape_list = data_shapes[dset_name].copy()
    if not df.__contains__(dset_name):
        shape_list[0] = None
        df.create_dataset(dset_name, data=data, maxshape=tuple(shape_list), dtype='int64',
                          chunks=tuple(data_shapes[dset_name]), compression="gzip")
    else:
        dataset = df[dset_name]
        len_old = dataset.shape[0]
        len_new = len_old + len(data)
        shape_list[0] = len_new
        dataset.resize(tuple(shape_list))
        dataset[len_old:len_new] = data

