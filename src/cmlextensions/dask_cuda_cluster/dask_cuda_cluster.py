# Copyright 2025 Cloudera. All Rights Reserved.
#
# This file is licensed under the Apache License Version 2.0
# (the "License"). You may not use this file except in compliance
# with the License. You may obtain  a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. Refer to the
# License for the specific permissions and limitations governing your
# use of the file.

import os
#import cdsw

# PBJ Runtimes do not have the cdsw library installed.
# Instead, the cml library is added to workloads in recent CML releases.
try:
    import cml.utils_v1 as utils
    cdsw = utils._emulate_cdsw()
except ImportError:
    import cdsw

DEFAULT_DASHBOARD_PORT = os.environ["CDSW_APP_PORT"]


class DaskCudaCluster:
    """Dask Cluster built on CML Worker infrastructure"""

    def __init__(
        self,
        num_workers,
        worker_cpu,
        nvidia_gpu,
        worker_memory,
        scheduler_cpu,
        scheduler_memory,
        dashboard_port=DEFAULT_DASHBOARD_PORT,
    ):
        self.num_workers = num_workers
        self.worker_cpu = worker_cpu
        self.nvidia_gpu = nvidia_gpu
        self.worker_memory = worker_memory
        self.scheduler_cpu = scheduler_cpu
        self.scheduler_memory = scheduler_memory
        self.dashboard_port = dashboard_port

        self.dask_scheduler_details = None
        self.dask_worker_details = None

    def _start_dask_scheduler(self):
        dask_scheduler_cmd = f"!dask scheduler --host 0.0.0.0 --dashboard-address 127.0.0.1:{self.dashboard_port}"

        args = {
            'n': 1,
            'cpu': self.scheduler_cpu,
            'nvidia_gpu': self.nvidia_gpu,
            'memory': self.scheduler_memory,
            'code': dask_scheduler_cmd,
        }

        if hasattr(cdsw.launch_workers, 'name'):
            args['name'] = 'Dask Scheduler'

        dask_scheduler = cdsw.launch_workers(**args)

        self.dask_scheduler_details = cdsw.await_workers(
            dask_scheduler, wait_for_completion=False, timeout_seconds=600
        )

    def _add_dask_workers(self, scheduler_addr):
        worker_start_cmd = f"!dask cuda worker {scheduler_addr}"

        args = {
            'n': self.num_workers,
            'cpu': self.worker_cpu,
            'memory': self.worker_memory,
            'nvidia_gpu': self.nvidia_gpu,
            'code': worker_start_cmd,
        }

        if hasattr(cdsw.launch_workers, 'name'):
            args['name'] = 'Dask Cuda Worker'

        dask_workers = cdsw.launch_workers(**args)

        self.dask_worker_details = cdsw.await_workers(
            dask_workers, wait_for_completion=True
        )

    def get_client_url(self):
        dask_scheduler_ip = self.dask_scheduler_details["workers"][0]["ip_address"]
        return f"tcp://{dask_scheduler_ip}:8786"

    def init(self):
        """
        Creates a Dask Cluster on the CML Workers infrastructure.
        """
        try:
            import dask  # pylint: disable=unused-import
        except ImportError as error:
            raise ImportError(
                "Could not import dask, for this module to work please run `pip install dask[complete]` \n -> "
                + str(error)
            ) from error

        # Start the dask scheduler process
        self._start_dask_scheduler()

        dask_scheduler_addr = self.get_client_url()

        self._add_dask_workers(dask_scheduler_addr)

        # TODO: could add cluster details, e.g., worker count and resources
        print(
            f"""
--------------------
Dask cluster started
--------------------

The Dask dashboard is running at
{self.get_dashboard_url()}

To connect to this Dask cluster from this CML Session,
use the following Python code:
  from dask.distributed import Client
  client = Client('{self.get_client_url()}')
"""
        )

    def get_dashboard_url(self):
        """
        Return the Dask dashboard url.
        """
        try:
            return self.dask_scheduler_details["workers"][0]["app_url"] + "status"
        except Error as error:
            raise Error("ERROR: Dask Cuda cluster is not running!")

    def terminate(self):
        """
        Terminates the Dask Cuda Cluster.
        """

        # TODO: stop workers only when they were created for this Dask Cluster
        cdsw.stop_workers()

        # Reset instance state
        self.dask_scheduler_ip = None
        self.dask_scheduler_addr = None
        self.dask_scheduler_details = None
        self.dask_worker_details = None
