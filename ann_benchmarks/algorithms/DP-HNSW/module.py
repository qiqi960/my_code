import os

import nmslib
import numpy as np   # 导入numpy模块别名为np，用于处理数值数据
import scipy.sparse   # 导入 SciPy 的稀疏矩阵模块，用于处理稀疏矩阵。

from ...constants import INDEX_DIR
from ..base.module import BaseANN

# 用于将稀疏矩阵转换为字符串格式
def sparse_matrix_to_str(matrix):
    result = []
    matrix = matrix.tocsr()
    matrix.sort_indices()
    for row in range(matrix.shape[0]):
        arr = [k for k in matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]]
        result.append(" ".join([str(k) for k in arr]))
    return result

# 将密集向量转换为字符串格式
def dense_vector_to_str(vector):
    if vector.dtype == np.bool_:
        indices = np.flatnonzero(vector)
    else:
        indices = vector
    result = " ".join([str(k) for k in indices])
    return result


class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.items()]   # 静态方法，用于将字典转换为字符串列表

     # 定义了构造函数，初始化了 NmslibReuseIndex 类的实例
    def __init__(self, metric, method_name, index_param, query_param):
        self._nmslib_metric = {"angular": "cosinesimil", "euclidean": "l2", "jaccard": "jaccard_sparse"}[metric]
        self._method_name = method_name
        self._save_index = False  # 将对象的 _save_index 属性初始化为 False，表示不保存索引。
        self._index_param = NmslibReuseIndex.encode(index_param)
        if query_param is not False:
            self._query_param = NmslibReuseIndex.encode(query_param)
            self.name = "Nmslib(method_name={}, index_param={}, " "query_param={})".format(
                self._method_name, self._index_param, self._query_param
            )
        else:
            self._query_param = None
            self.name = "Nmslib(method_name=%s, index_param=%s)" % (self._method_name, self._index_param)

        # 构建索引文件的路径，包括索引目录、方法名、度量方式和索引参数的信息
        self._index_name = os.path.join(
            INDEX_DIR, "nmslib_%s_%s_%s" % (self._method_name, metric, "_".join(self._index_param))
        )

        # 获取索引文件路径的父目录
        d = os.path.dirname(self._index_name)
        # 检查索引文件的父目录是否存在，如果不存在，则创建索引文件的父目录，以确保索引文件所在的目录存在
        if not os.path.exists(d):
            os.makedirs(d)

    # 根据给定的数据拟合索引
    def fit(self, X):
        if self._method_name == "vptree":
            # To avoid this issue: terminate called after throwing an instance
            # of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too
            # big. Select the parameters so that <total # of records> is NOT
            # less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append("bucketSize=%d" % min(int(len(X) * 0.0005), 1000))

        if self._nmslib_metric == "jaccard_sparse":
            self._index = nmslib.init(
                space=self._nmslib_metric,
                method=self._method_name,
                data_type=nmslib.DataType.OBJECT_AS_STRING,
            )
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                sparse_matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
                sparse_matrix.indices = np.hstack(X).astype(np.int32)
                sparse_matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
                sparse_matrix.data = np.ones(sparse_matrix.indices.shape[0], dtype=np.float32)
                sparse_matrix.sort_indices()
            else:
                sparse_matrix = scipy.sparse.csr_matrix(X)
            string_data = sparse_matrix_to_str(sparse_matrix)
            self._index.addDataPointBatch(string_data)
        else:
            self._index = nmslib.init(space=self._nmslib_metric, method=self._method_name)
            self._index.addDataPointBatch(X)

        if os.path.exists(self._index_name):
            print("Loading index from file")
            self._index.loadIndex(self._index_name)
        else:
            self._index.createIndex(self._index_param)
            if self._save_index:
                self._index.saveIndex(self._index_name)
        if self._query_param is not None:
            self._index.setQueryTimeParams(self._query_param)

    # 设置查询参数
    def set_query_arguments(self, ef):
        if self._method_name == "hnsw" or self._method_name == "sw-graph":
            self._index.setQueryTimeParams(["efSearch=%s" % (ef)])

    # 用于单个查询操作，返回最近邻的索引
    def query(self, v, n):
        if self._nmslib_metric == "jaccard_sparse":
            v_string = dense_vector_to_str(v)
            ids, distances = self._index.knnQuery(v_string, n)
        else:
            ids, distances = self._index.knnQuery(v, n)
        return ids

    # 用于批量查询操作
    def batch_query(self, X, n):
        if self._nmslib_metric == "jaccard_sparse":
            sparse_matrix = scipy.sparse.csr_matrix(X)
            string_data = sparse_matrix_to_str(sparse_matrix)
            self.res = self._index.knnQueryBatch(string_data, n)
        else:
            self.res = self._index.knnQueryBatch(X, n)

    # 获取批量查询的结果
    def get_batch_results(self):
        return [x for x, _ in self.res]
