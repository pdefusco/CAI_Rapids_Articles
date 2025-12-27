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

STORAGE = "s3a://pdf-dec-buk-70b78c3b/data/dask-example/50Brows_10Kcols_10kparts"
OUTPUT = "s3a://pdf-dec-buk-70b78c3b/data/dask-example/joined_df"

from src.cmlextensions.dask_cuda_cluster.dask_cuda_cluster import DaskCudaCluster

cluster = DaskCudaCluster(num_workers=25, worker_cpu=26, nvidia_gpu=4, worker_memory=120, scheduler_cpu=8, scheduler_memory=64)
cluster.init()

# SINGLE JOIN

from dask.distributed import Client

client = Client(cluster.get_client_url())

import dask.array as da
import os
import dask.dataframe as dd

df_left = dd.read_parquet(STORAGE)
df_left.head()

df_right = dd.read_parquet(STORAGE)
df_right.head()

df_join = df_left.set_index('unique_id').join(df_right.set_index('unique_id'), how='inner', on='unique_id', lsuffix='_left2', rsuffix='_right2')

df_join.head()

# MULTI DF JOIN

df3 = dd.read_parquet(STORAGE)
df4 = dd.read_parquet(STORAGE)
df5 = dd.read_parquet(STORAGE)

df_join = df_join.set_index('unique_id').join(df3.set_index('unique_id'), how='inner', on='unique_id', lsuffix='_left3', rsuffix='_right3')
df_join = df_join.set_index('unique_id').join(df4.set_index('unique_id'), how='inner', on='unique_id', lsuffix='_left4', rsuffix='_right4')
df_join = df_join.set_index('unique_id').join(df5.set_index('unique_id'), how='inner', on='unique_id', lsuffix='_left5', rsuffix='_right5')

df_join.head()

df_join = df_join.categorize('col1_left2')
df_join = df_join.reset_index(drop=False)

df_join.head()

df_pivot = dd.pivot_table(df_join, index='unique_id', columns='col1_left2', values='col2_left2', aggfunc='mean')

df_pivot_flat = df_pivot.reset_index()

df_pivot_flat.to_parquet(OUTPUT, engine='pyarrow', overwrite=True)
