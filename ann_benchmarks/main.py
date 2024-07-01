import argparse
from dataclasses import replace
import h5py
import logging
import logging.config
import multiprocessing.pool
import os
import random
import shutil
import sys
from typing import List

import docker
import psutil

from .definitions import (Definition, InstantiationStatus, algorithm_status,
                          get_definitions, list_algorithms)
from .constants import INDEX_DIR
from .datasets import DATASETS, get_dataset
from .results import build_result_filepath
from .runner import run, run_docker

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("annb")


def positive_int(input_str: str) -> int:
    """
    Validates if the input string can be converted to a positive integer.
    验证输入字符串是否可以转换为正整数，并返回该正整数。如果输入字符串不能转换为正整数，则会引发 argparse.ArgumentTypeError 异常。
    Args:
        input_str (str): The input string to validate and convert to a positive integer.

    Returns:
        int: The validated positive integer.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a positive integer.
    """
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer")

    return i


def run_worker(cpu: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    """
    Executes the algorithm based on the provided parameters.
    基于提供的参数执行算法
    The algorithm is either executed directly or through a Docker container based on the `args.local`
     argument. The function runs until the queue is emptied. When running in a docker container, it 
    executes the algorithm in a Docker container.

    Args:
        cpu (int): The CPU number to be used in the execution.
        args (argparse.Namespace): User provided arguments for running workers. 
        queue (multiprocessing.Queue): The multiprocessing queue that contains the algorithm definitions.

    Returns:
        None
    """
    while not queue.empty():
        definition = queue.get()
        # 检查参数 args.local 是否为 True，表示算法将在本地执行。
        if args.local:
            run(definition, args.dataset, args.count, args.runs, args.batch)
        else:
            memory_margin = 500e6  # reserve some extra memory for misc stuff
            mem_limit = int((psutil.virtual_memory().available - memory_margin) / args.parallelism)
            cpu_limit = str(cpu) if not args.batch else f"0-{multiprocessing.cpu_count() - 1}"
            print('i am in run_worker')
            run_docker(definition, args.dataset, args.count, args.runs, args.timeout, args.batch, cpu_limit, mem_limit)
            print("run_worker terminate!")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    '''这行代码添加了一个名为 --dataset 的参数，它用于指定要加载训练点的数据集。
    metavar 参数指定了在帮助文档中显示的参数名称，help 参数提供了关于参数作用的说明，default 参数设置了参数的默认值，
    choices 参数指定了可选的数据集名称，它们来自于 DATASETS 字典的键。'''
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
        default="glove-100-angular",
        choices=DATASETS.keys(),
    )
    '''这行代码添加了两个参数，-k 和 --count，它们都用于指定要搜索的最近邻的数量。
    default 参数设置了参数的默认值为 10，type 参数指定了参数的类型为 positive_int，这意味着它必须是一个正整数。
    '''
    parser.add_argument(
        "-k", "--count", default=10, type=positive_int, help="the number of near neighbours to search for"
    )
    '''
    这行代码添加了一个名为 --definitions 的参数，它用于指定算法的基本目录路径。
    '''
    parser.add_argument(
        "--definitions", metavar="FOLDER",
        help="base directory of algorithms. Algorithm definitions expected at 'FOLDER/*/config.yml'",
        default="ann_benchmarks/algorithms"
    )
    '''
    这行代码添加了一个名为 --algorithm 的参数32，它用于指定要运行的特定算法的名称。
    metavar 参数指定了在帮助文档中显示的参数名称，help 参数提供了关于参数作用的说明，
    default 参数设置了参数的默认值为 None，表示运行所有算法。'''
    parser.add_argument("--algorithm", metavar="NAME", help="run only the named algorithm", default=None)
    '''这行代码添加了一个名为 --docker-tag 的参数，它用于指定要运行的算法所在的特定 Docker 镜像的名称。
    metavar 参数指定了在帮助文档中显示的参数名称，help 参数提供了关于参数作用的说明，default 参数设置了参数的默认值为 None。
    '''
    parser.add_argument(
        "--docker-tag", metavar="NAME", help="run only algorithms in a particular docker image", default=None
    )
    '''这行代码添加了一个名为 --list-algorithms 的参数，如果设置了此参数，程序将打印所有已知算法的名称并退出。
    '''
    parser.add_argument(
        "--list-algorithms", help="print the names of all known algorithms and exit", action="store_true"
    )
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")
    #用于指定每个算法实例运行的次数，并仅使用最佳结果
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only" " the best result",
        default=5,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1" "if no timeout should be set",
        default=2 * 3600,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, then will run everything locally (inside the same " "process) rather than using Docker",
    )
    parser.add_argument("--batch", action="store_true", help="If set, algorithms get all queries at once")
    parser.add_argument(
        "--max-n-algorithms", type=int, help="Max number of algorithms to run (just used for testing)", default=-1
    )
    parser.add_argument("--run-disabled", help="run algorithms that are disabled in algos.yml", action="store_true")
    parser.add_argument("--parallelism", type=positive_int, help="Number of Docker containers in parallel", default=1)

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None
    return args


def filter_already_run_definitions(
        definitions: List[Definition],
        dataset: str,
        count: int,
        batch: bool,
        force: bool
) -> List[Definition]:
    """Filters out the algorithm definitions based on whether they have already been run or not.
    检查每个算法定义是否存在已有的结果。如果不存在已有结果，或者参数 force=True，则保留该算法定义；否则，丢弃它。
    This function checks if there are existing results for each definition by constructing the 
    result filename from the algorithm definition and the provided arguments. If there are no 
    existing results or if the parameter `force=True`, the definition is kept. Otherwise, it is
    discarded.

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.
        dataset (str): The name of the dataset to load training points from.
        force (bool): If set, re-run algorithms even if their results already exist.

        count (int): The number of near neighbours to search for (only used in file naming convention).
        batch (bool): If set, algorithms get all queries at once (only used in file naming convention).

    Returns:
        List[Definition]: A list of algorithm definitions that either have not been run or are 
                          forced to be re-run.
    """
    filtered_definitions = []

    for definition in definitions:
        not_yet_run = [
            query_args
            for query_args in (definition.query_argument_groups or [[]])
            if force or not os.path.exists(build_result_filepath(dataset, count, definition, query_args, batch))
        ]

        if not_yet_run:
            definition = replace(definition,
                                 query_argument_groups=not_yet_run) if definition.query_argument_groups else definition
            filtered_definitions.append(definition)

    return filtered_definitions


def filter_by_available_docker_images(definitions: List[Definition]) -> List[Definition]:
    """
    Filters out the algorithm definitions that do not have an associated, available Docker images.
    过滤掉没有关联可用 Docker 镜像的算法定义。
    This function uses the Docker API to list all Docker images available in the system. It 
    then checks the Docker tags associated with each algorithm definition against the list 
    of available Docker images, filtering out those that are unavailable. 

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.

    Returns:
        List[Definition]: A list of algorithm definitions that are associated with available Docker images.
    """
    #使用 Docker API 来创建一个 Docker 客户端对象，该对象用于与 Docker 引擎进行通信
    docker_client = docker.from_env()
    #这行代码列出了系统中所有可用的 Docker 镜像，并提取出每个镜像的标签。
    # 这里使用了一个集合推导式，遍历了所有 Docker 镜像，提取每个镜像的标签，并将其存储在 docker_tags 集合中。
    # .split(":")[0] 是为了获取镜像标签的名称部分，而不包括版本号等其他信息。
    docker_tags = {tag.split(":")[0] for image in docker_client.images.list() for tag in image.tags}

    #这一行代码检查输入的算法定义列表中每个算法定义关联的 Docker 标签是否在可用 Docker 镜像的标签集合中。
    # set(d.docker_tag for d in definitions) 构建了一个包含所有算法定义关联的 Docker 标签的集合，
    # 然后使用 .difference() 方法找到在算法定义集合中但不在可用 Docker 镜像标签集合中的标签。这些标签表示找不到对应的 Docker 镜像。
    missing_docker_images = set(d.docker_tag for d in definitions).difference(docker_tags)
    if missing_docker_images:
        logger.info(f"not all docker images available, only: {docker_tags}")
        logger.info(f"missing docker images: {missing_docker_images}")
        definitions = [d for d in definitions if d.docker_tag in docker_tags]

    return definitions


def check_module_import_and_constructor(df: Definition) -> bool:
    """
    Verifies if the algorithm module can be imported and its constructor exists.
    这个函数用于检查算法模块是否可以被导入并且它的构造函数是否存在。
    如果算法模块不能被导入，或者构造函数不存在，则函数会返回 False。
    如果算法模块可以被导入并且构造函数存在，则函数会返回 True。
    This function checks if the module specified in the definition can be imported. 
    Additionally, it verifies if the constructor for the algorithm exists within the 
    imported module.

    Args:
        df (Definition): A definition object containing the module and constructor 
        for the algorithm.

    Returns:
        bool: True if the module can be imported and the constructor exists, False 
        otherwise.
    """
    status = algorithm_status(df)
    if status == InstantiationStatus.NO_CONSTRUCTOR:
        raise Exception(
            f"{df.module}.{df.constructor}({df.arguments}): error: the module '{df.module}' does not expose the named constructor"
        )
    if status == InstantiationStatus.NO_MODULE:
        logging.warning(
            f"{df.module}.{df.constructor}({df.arguments}): the module '{df.module}' could not be loaded; skipping"
        )
        return False

    return True


def create_workers_and_execute(definitions: List[Definition], args: argparse.Namespace):
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers. 

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than 
                   one worker.
    """
    cpu_count = multiprocessing.cpu_count()
    if args.parallelism > cpu_count - 1:
        raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    if args.batch and args.parallelism > 1:
        raise Exception(
            f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
        )
    # 创建一个多进程队列 task_queue，并将所有算法定义放入队列中以供工作进程处理。
    task_queue = multiprocessing.Queue()
    for definition in definitions:
        task_queue.put(definition)

    try:
        # 根据用户指定的并行度创建多个工作进程，每个工作进程都调用 run_worker 函数来执行任务。
        workers = [multiprocessing.Process(target=run_worker, args=(i + 1, args, task_queue)) for i in
                   range(args.parallelism)]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    #无论是否正常执行完毕，都执行以下操作：
    #记录信息，表示正在终止工作进程。
    #终止所有工作进程。
    finally:
        logger.info("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]
    print("create_workers_and_execute terminate!")


def filter_disabled_algorithms(definitions: List[Definition]) -> List[Definition]:
    """
    Excludes disabled algorithms from the given list of definitions.

    This function filters out the algorithm definitions that are marked as disabled in their `config.yml`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.

    Returns:
        List[Definition]: A list of algorithm definitions excluding any that are disabled.
    """
    disabled_algorithms = [d for d in definitions if d.disabled]
    if disabled_algorithms:
    #打印出被禁用的算法
        logger.info(f"Not running disabled algorithms {disabled_algorithms}")

    return [d for d in definitions if not d.disabled]


def limit_algorithms(definitions: List[Definition], limit: int) -> List[Definition]:
    """
    这个函数用于根据给定的限制数量来限制算法定义的数量。
    如果限制是负数，那么将返回所有的算法定义。在进行有效的采样之前，应该对算法定义进行洗牌，然后再使用此函数来限制算法的数量。
    Limits the number of algorithm definitions based on the given limit.

    If the limit is negative, all definitions are returned. For valid 
    sampling, `definitions` should be shuffled before `limit_algorithms`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.
        limit (int): The maximum number of definitions to return.

    Returns:
        List[Definition]: A trimmed list of algorithm definitions.
    """
    return definitions if limit < 0 else definitions[:limit]


def main():
    args = parse_arguments()  # 从命令行获取参数

    if args.list_algorithms:  # 检查用户是否选择了列出可用算法的选项
        list_algorithms(args.definitions)
        sys.exit(0)

    if os.path.exists(INDEX_DIR):  # 如果索引目录 INDEX_DIR 存在，就删除它。这可能是为了确保每次运行都是从头开始的。
        shutil.rmtree(INDEX_DIR)

    dataset, dimension = get_dataset(args.dataset)  # 获取指定数据集的信息
    '''
    print("++++++")
    print(dimension, dataset.attrs.get("point_type", "float"), dataset.attrs["distance"])
    print("++++++")
    sys.exit(0)
    '''

    definitions: List[Definition] = get_definitions(
        dimension=dimension,
        point_type=dataset.attrs.get("point_type", "float"),
        distance_metric=dataset.attrs["distance"],
        count=args.count
    )

    '''
    for definition in definitions:
        print("Algorithm Name:", definition.algorithm)
        print("Module:", definition.module)
        print("Constructor:", definition.constructor)
        print("Disabled:", definition.disabled)
        print("Docker Tag:", definition.docker_tag)
    sys.exit(0)
    '''
    random.shuffle(definitions)  # 随机打乱算法定义的顺序，可能是为了在运行时更好地分散算法的执行。

    definitions = filter_already_run_definitions(definitions,  # 过滤掉已经运行过的算法定义，这可能是为了避免重复运行相同的算法。
                                                 dataset=args.dataset,
                                                 count=args.count,
                                                 batch=args.batch,
                                                 force=args.force,
                                                 )

    if args.algorithm:  # 如果指定了命令行参数 algorithm，就只运行该算法，过滤掉其他算法。
        logger.info(f"running only {args.algorithm}")
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    # 如果不是本地运行，就过滤掉没有可用Docker镜像的算法定义；否则，检查算法是否能够通过模块导入和构造函数检查
    if not args.local:
        definitions = filter_by_available_docker_images(definitions)
    else:
        definitions = list(filter(
            check_module_import_and_constructor, definitions
        ))

    # 过滤掉已禁用的算法
    definitions = filter_disabled_algorithms(definitions) if not args.run_disabled else definitions
    # 限制运行的算法数量，确保不超过指定的最大算法数。
    definitions = limit_algorithms(definitions, args.max_n_algorithms)

    # 检查是否有要运行的算法，如果没有，抛出异常；否则，记录算法的执行顺序。
    if len(definitions) == 0:
        raise Exception("Nothing to run")
    else:
        logger.info(f"Order: {definitions}")
    # 使用create_workers_and_execute()函数创建并执行工作进程，可能是为了并行运行多个算法，以加快整个测试的执行速度。
    create_workers_and_execute(definitions, args)
