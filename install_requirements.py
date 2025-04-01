!pip install dask distributed dask-cuda
!pip install dask
!pip install cloudpickle
!pip install 'dask[dataframe]'
!pip install 'dask[complete]'
!pip install "bokeh!=3.0.*,>=2.4.2"

!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* cuml-cu12==24.6.* \
    cugraph-cu12==24.6.*

!pip install gpustat

#!pip install graphviz
