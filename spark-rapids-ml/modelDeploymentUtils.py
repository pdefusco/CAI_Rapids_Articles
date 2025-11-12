#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
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

import os
import subprocess

# Import inference-related libraries
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import time
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any, List

# Used while configuring CDP credentials config
import getpass

# Create CDP configuration
def configure_cdp(variable, value):
    command = ["cdp", "configure", "set", variable, value]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("cdp command failed.")
        print("Error:")
        print(result.stderr)
    return None
print("CDP config function defined!")

def get_registry_endpoint(environmentName):
    command = ["cdp", "ml", "list-model-registries"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("cdp command failed.")
        print("Error:")
        print(result.stderr)
        return None
    try:
        registries = json.loads(result.stdout)['modelRegistries']
        for registry in registries:
            if registry['environmentName'] == environmentName:
                return registry['domain']
                #return json.loads(result.stdout)['modelRegistries'][0]['domain']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return None

# Obtain CDP_TOKEN JWT using CDP CLI
def get_ums_jwt_token():
    command = ["cdp", "iam", "generate-workload-auth-token",
               "--workload-name", "DE"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("cdp command failed.")
        print("Error:")
        print(result.stderr)
        return None

    try:
        return json.loads(result.stdout)['token']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return None

        print("Defined function to get CDP_TOKEN!")

# Functions to retrieve model details. These can be obtained from the AI Registry UI as well
def get_model_details(registry_endpoint, model_name, token):
    headers = {'Authorization': 'Bearer ' + token,
           'Content-Type': 'application/json'}
    client = httpx.Client(headers=headers)
    url = registry_endpoint+'/api/v2/models'
    params = {
        'name': model_name,
    }
    result = client.get(url, params=params)
    try:
        return next((element for element in result.json()['models'] if element.get('name') == model_name), None)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return None

def list_model_info(self, model_id):

    if "Content-Type" in self.headers:
        del self.headers["Content-Type"]
    url = f"{self.base_url}/api/v2/models/{model_id}"
    response = requests.get(url, headers=self.headers)
    response.raise_for_status()
    return response.json()

def get_most_recent_model_version(registry_endpoint, model_id, token):
    headers = {'Authorization': 'Bearer ' + token,
           'Content-Type': 'application/json'}
    client = httpx.Client(headers=headers)
    url = registry_endpoint+'/api/v2/models/'+ model_id
    result = client.get(url)
    try:
        all_versions = result.json()
        num_versions = len(all_versions['model_versions'])
        return all_versions['model_versions'][num_versions-1]['version']
    except (json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None

def get_caii_domain(environmentName):
    command = ["cdp", "ml", "list-ml-serving-apps"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("cdp command failed.")
        print("Error:")
        print(result.stderr)
        return None

    try:
        aiisApps = json.loads(result.stdout)['apps']
        for app in aiisApps:
            if app['environmentName'] == environmentName:
                return app['cluster']['domainName']
        #return json.loads(result.stdout)['apps'][1]['cluster']['domainName']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return None

def deploy_model_to_caii(caii_domain, cdp_token, model_id, model_version, endpoint_name):
    # construct url
    deploy_url = f"https://{caii_domain}/api/v1alpha1/deployEndpoint"

    headers = {'Authorization': 'Bearer ' + cdp_token,
           'Content-Type': 'application/json'}

    client = httpx.Client(headers=headers)

    # Deploy the model endpoint. Note that "serving-default" is the only valid
    # namespace. Adjust resources and autoscaling parameters as you need. Also note
    # that we're not requesting a GPU for the model deployment. If your model requires GPUs,
    # you can add it to the "resources" section, e.g.
    # "resources": {
    #     "num_gpus": "2",
    #     "req_cpu": "4",
    #     "req_memory": "8Gi"
    #  }
    #

    deploy_payload = {
        "namespace": "serving-default",
        "name": f"{endpoint_name}",
        "instance_type": "m5.4xlarge",
        "task": "INFERENCE",
        "source": {
            "registry_source": {
                "model_id": f"{model_id}",
                "version": f"{model_version}"
            }
        },
        "resources": {
            "req_cpu": "4",
            "req_memory": "16Gi",
            "num_gpus": "0"
        },
        "autoscaling": {
            "min_replicas": "1",
            "max_replicas": "5"
        }
    }
    try:
        response = client.post(deploy_url, json=deploy_payload)
        response.raise_for_status()
        print(f"Deployed {endpoint_name} successfully!")
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Error deploying {endpoint_name}: {e}")

def endpoint_is_ready(caii_domain, cdp_token, endpoint_name):
    headers = {'Authorization': 'Bearer ' + cdp_token,
           'Content-Type': 'application/json'}
    url = f"https://{caii_domain}/api/v1alpha1/describeEndpoint"
    payload = {"namespace": "serving-default", "name": f"{endpoint_name}"}

    client = httpx.Client(headers=headers)

    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()['status']['active_model_state'] == 'Loaded'
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Error describing {endpoint_name}: {e}")

def base_url(url, target):
    parsed = urlparse(url)
    path = parsed.path

    target = target
    pos = path.find(target)

    if pos == -1:
        return url

    # Find the end position and strip everything after
    end_pos = pos + len(target)
    new_path = path[:end_pos]

    # Reconstruct the URL
    new_parsed = parsed._replace(path=new_path)
    return urlunparse(new_parsed)

def get_endpoint_base_url(caii_domain, cdp_token, endpoint_name):
    headers = {'Authorization': 'Bearer ' + cdp_token,
           'Content-Type': 'application/json'}
    url = f"https://{caii_domain}/api/v1alpha1/describeEndpoint"
    payload = {"namespace": "serving-default", "name": f"{endpoint_name}"}

    client = httpx.Client(headers=headers)

    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return base_url(response.json()['url'], endpoint_name)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Error describing {endpoint_name}: {e}")
