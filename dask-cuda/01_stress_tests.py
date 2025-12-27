#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco, James Horsch
#***************************************************************************/

from src.cmlextensions.dask_cuda_cluster.dask_cuda_cluster import DaskCudaCluster

cluster = DaskCudaCluster(num_workers=8, worker_cpu=26, nvidia_gpu=4, worker_memory=120, scheduler_cpu=8, scheduler_memory=64)
cluster.init()

from dask.distributed import Client

client = Client(cluster.get_client_url())

import dask.array as da

# Create a dask array from a NumPy array
x = da.from_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))

# Perform a computation on the dask array
y = (x + 1) * 2

# Submit the computation to the cluster for execution
future = client.submit(y.compute)

# Wait for the computation to complete and retrieve the result
result = future.result()

print(result)

import cupy as cp

client = Client(cluster.get_client_url())

# Compute-heavy custom kernel

def heavy_elementwise(x):
    for _ in range(20):
        x = cp.sin(x) ** 2 + cp.cos(x) ** 2
    return x

x = da.from_array(
    cp.random.random((30000, 30000), dtype=cp.float32),
    chunks=(3000, 3000)
)

y = da.map_blocks(heavy_elementwise, x, dtype=x.dtype)

y = y.persist()

# Large matrix multiplication

x = da.from_array(
    cp.random.random((20000, 10000), dtype=cp.float32),
    chunks=(2000, 2000)
)

y = da.from_array(
    cp.random.random((10000, 20000), dtype=cp.float32),
    chunks=(2000, 2000)
)

z = da.matmul(x, y)

result = z.sum().compute()
print(result)

# Rechunking stress test

x = da.from_array(
    cp.random.random((40000, 40000), dtype=cp.float32),
    chunks=(4000, 4000)
)

# Force heavy data movement
y = x.rechunk((10000, 1000))

result = y.std().compute()
print(result)

# Iterative computation (growing task graphs)

x = da.from_array(
    cp.random.random((15000, 15000), dtype=cp.float32),
    chunks=(1500, 1500)
)

for i in range(6):
    x = da.tanh(x @ x.T)

result = x.mean().compute()
print(result)

# Persist vs compute (cache behavior demo)

x = da.from_array(
    cp.random.random((30000, 30000), dtype=cp.float32),
    chunks=(3000, 3000)
)

# Persist keeps data on GPU memory
y = (x + 1).persist()

# Reuses cached data
z = (y * y).sum()

print(z.compute())


# Maximum Dashboard Chaos

def brutal(x):
    for _ in range(50):
        x = cp.sqrt(x * x + 1)
    return x

x = da.from_array(
    cp.random.random((35000, 35000), dtype=cp.float32),
    chunks=(2500, 2500)
)

y = da.map_blocks(brutal, x, dtype=x.dtype)
y = y.rechunk((5000, 5000))

y.mean().compute()
