from tqdm.std import tqdm

from prTree.quadTree import Index


# 建树
def build_tree(traj_data, x_range, y_range, max_items, max_depth):
    qtree = Index(bbox=(x_range[0], y_range[0], x_range[1], y_range[1]), max_items=max_items, max_depth=max_depth)
    point_num = 0

    for traj in tqdm(traj_data):
        for point in traj:
            point_num += 1
            x, y = point
            qtree.insert(point_num, (x, y, x, y))

    print("traj point nums:", point_num)

    return qtree
