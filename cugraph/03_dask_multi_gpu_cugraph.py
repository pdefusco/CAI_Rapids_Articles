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
# #  Author(s): Paul de Fusco
#***************************************************************************/

!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* cuml-cu12==24.6.* \
    cugraph-cu12==24.6.*

!pip install bokeh!=3.0.*,>=2.4.2

import dask
import cugraph
import cudf
import dask_cudf
import os
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster(
  n_workers=2,
  threads_per_worker=1,
  CUDA_VISIBLE_DEVICES="0,1",
  #rmm_managed_memory=True,
  #rmm_pool_size="20GB",
  dashboard_address='localhost:8090'
)

client = Client(cluster)

print(client)

client.dashboard_link

print("https://{}.{}".format(os.environ["CDSW_ENGINE_ID"], os.environ["CDSW_DOMAIN"]))

def compute_pagerank():

    import cudf
    import cugraph

    # Create an edge list as a DataFrame
    edgelist_df = cudf.DataFrame({
      "src": [0, 1, 2, 0, 1, 1, 1, 3],
      "dst": [1, 2, 3, 3, 1, 0, 0, 0],
      "weight": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 5.0]
    })

    # Create a cuGraph graph from the edge list
    G = cugraph.Graph()
    G.from_cudf_edgelist(edgelist_df, source="src", destination="dst", edge_attr="weight")

    # Print the graph
    print(G)

    # Compute edge-weighted PageRank
    pr = cugraph.pagerank(G)

    # Print the edge-weighted PageRank scores
    print(pr)

    #dask_edges = dask_cudf.from_cudf(edges, npartitions=2)
    return pr

dask_pagerank = dask.delayed(compute_pagerank)()

pagerank_result = dask_pagerank.compute()

print(pagerank_result)
