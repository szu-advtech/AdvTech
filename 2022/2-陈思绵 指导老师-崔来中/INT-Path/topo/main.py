import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt

# it is necessary that the script is run from the console from any directory
# and the import of modules occurs correctly
SCRIPT_DIRECTORY = str(os.path.abspath(__file__).rsplit(sep='\\', maxsplit=2)[0]).replace('\\', '/')
sys.path.insert(0, SCRIPT_DIRECTORY)
from graph_tools import *


# DEFAULT_SOURCE = SCRIPT_DIRECTORY + '/input/graph.txt'
# DEFAULT_ALGORITHM = 'dfs'
# DEFAULT_STORAGE = SCRIPT_DIRECTORY + '/output/'
DEFAULT_SOURCE = '../topo.txt'
DEFAULT_ALGORITHM = 'dfs'
DEFAULT_STORAGE = '../output/'


def get_logger() -> logging.Logger:
    """
    Function that return logger withs style settings

    Returns
    -------
    logging.Logger :
        logger with style settings by logging.StreamHandler

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main(start: object, file_path: str = DEFAULT_SOURCE,
         algorithm: str = DEFAULT_ALGORITHM, storage: str = DEFAULT_STORAGE) -> bool:

    logger = get_logger()
    #调整大小
    fig = plt.figure(figsize=(6, 5))
    #plt.title(f'{algorithm.upper()} for graph from ...{get_filename(file_path)}')
    #标题
    plt.title(f'Euler Balance Algorithm')
    plt.title('Euler (Balance) Algorithm',loc='center')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    camera = Camera(fig)
    graph = graph_builder(file_path)
    if create_gif(graph, camera, func=algorithm, start=start, source=file_path, storage=storage):
        logger.info("Successful completion of the program!")
        return True
    else:
        logger.error('An error has occurred!')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create graph traversing visualization.')

    parser.add_argument('start', type=str, help='The node of the graph from which to start the search.')
    parser.add_argument('-algo', '--algorithm', type=str, default=DEFAULT_ALGORITHM,
                        help=f'The name of the algorithm you want to apply to the graph `dfs` or `bfs`,'
                             f' default: {DEFAULT_ALGORITHM}.')
    parser.add_argument('--storage', type=str, default=DEFAULT_STORAGE,
                        help=f'Directory where you want to save the gif file, default: {DEFAULT_STORAGE}.')
    parser.add_argument('--source', type=str, default=DEFAULT_SOURCE,
                        help=f'Directory to .txt file to read graph, default: {DEFAULT_SOURCE}.')

    args = parser.parse_args()

    main(args.start, args.source, args.algorithm, args.storage)
