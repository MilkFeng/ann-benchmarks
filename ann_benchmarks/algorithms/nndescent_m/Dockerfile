FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y cmake ninja-build build-essential

RUN pip3 install pybind11 numpy scikit-build-core --break-system-packages

COPY ann_benchmarks/algorithms/nndescent_m/source /tmp/source

RUN cd /tmp/source && pip3 install . --break-system-packages

RUN python3 -c 'import nndescent_m'