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

# Import required libraries
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch
import mlflow.onnx
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns

# Import inference-related libraries
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import time
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any, List

# Used while configuring CDP credentials config
import getpass

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")

class TritonBatchInference:
    """Class to handle Triton inference with dynamic batching"""

    def __init__(self, base_url: str, model_name: str, token: str):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        # Note: Uncomment these lines for actual inference
        self.httpx_client = httpx.Client(headers=self.headers)
        self.client = OpenInferenceClient(base_url=base_url, httpx_client=self.httpx_client)

    def get_triton_model_config(self) -> Optional[Dict[str, Any]]:
        """Get Triton model configuration including dynamic batching settings"""
        config_url = f"{self.base_url}/v2/models/{self.model_name}/config"

        try:
            response = self.httpx_client.get(config_url)
            response.raise_for_status()
            config = response.json()
            print("Model Configuration:")
            print(json.dumps(config, indent=2))

            # Extract dynamic batching info
            if 'dynamic_batching' in config:
                return config['dynamic_batching']
            else:
                return None

        except Exception as e:
            print(f"Error getting model config: {e}")
            return None

    def check_server_status(self) -> bool:
        """Check if the server is ready and get model metadata"""
        try:
            # Check server readiness
            self.client.check_server_readiness()
            print("✓ Server is ready")

            # Get model metadata
            metadata = self.client.read_model_metadata(self.model_name)
            metadata_dict = json.loads(metadata.json())
            print("Model Metadata:")
            print(json.dumps(metadata_dict, indent=2))

            return True
        except Exception as e:
            print(f"Error checking server status: {e}")
            return False
        return True

    def prepare_iris_data(self) -> tuple:
        """Load and prepare the iris dataset for inference"""
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split data (we'll use test set for batch inference)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scale the features (assuming model was trained with scaled features)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"Dataset prepared: {len(X_test_scaled)} samples for inference")
        print(f"Feature shape: {X_test_scaled.shape}")

        return X_test_scaled, y_test, iris.target_names

    def create_batches(self, data: np.ndarray, batch_size: int) -> List[np.ndarray]:
        """Create batches from the input data"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches

    def run_batch_inference_demo(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Run batch inference on the data"""
        batches = self.create_batches(data, batch_size)
        all_predictions = []

        print(f"Running inference on {len(batches)} batches of size {batch_size}")

        for i, batch in enumerate(batches):
            try:
                # Create inference request
                # Note: Adjust input/output names based on your model's specification
                inference_request = InferenceRequest(
                    inputs=[{
                        "name": "input",  # Adjust based on your model's input name
                        "shape": list(batch.shape),
                        "datatype": "FP32",
                        "data": batch.flatten().tolist()
                    }],
                )

                start_time = time.time()
                response = self.client.model_infer(self.model_name, request=inference_request)
                inference_time = time.time() - start_time

                # Extract predictions from response
                response_dict = json.loads(response.json())
                output_data = response_dict['outputs'][0]['data']

                # Reshape output to match batch size and number of classes
                output_array = np.array(output_data).reshape(batch.shape[0], -1)
                predictions = np.argmax(output_array, axis=1)
                all_predictions.extend(predictions)

                print(f"Batch {i+1}/{len(batches)} completed in {inference_time:.3f}s")

            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                # Fill with dummy predictions to maintain consistency
                dummy_predictions = [0] * len(batch)
                all_predictions.extend(dummy_predictions)

        return np.array(all_predictions)

    def evaluate_predictions(self, predictions: np.ndarray, y_true: np.ndarray,
                           class_names: List[str]) -> Dict[str, Any]:
        """Evaluate the predictions and return metrics"""
        accuracy = accuracy_score(y_true, predictions)
        report = classification_report(y_true, predictions, target_names=class_names)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'total_samples': len(y_true),
            'correct_predictions': np.sum(predictions == y_true)
        }

        return results

print("Inference client class defined!")
