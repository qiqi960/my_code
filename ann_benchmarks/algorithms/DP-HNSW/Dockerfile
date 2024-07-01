FROM ann-benchmarks

RUN apt-get update && apt-get install -y cmake libboost-all-dev libeigen3-dev libgsl0-dev
RUN pip3 install dp-hnsw
RUN cd dp-hnsw/similarity_search && cmake . -DWITH_EXTRAS=1
RUN cd dp-hnsw/similarity_search && make -j4
RUN pip3 install pybind11
RUN cd dp-hnsw/python_bindings && python3 setup.py build
RUN cd dp-hnsw/python_bindings && python3 setup.py install
RUN python3 -c 'import nmslib'
