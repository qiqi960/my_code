from multiprocessing import freeze_support

from ann_benchmarks.main import main

if __name__ == "__main__": #用于检查当前脚本是否被直接运行，而不是被其他脚本导入。如果这个脚本是直接运行的，则执行后续的代码块
    freeze_support()
    main()
