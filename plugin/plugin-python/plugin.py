from concurrent import futures
import sys
import time
import json
from typing import Dict
from models import Model

import grpc

from proto import model_pb2
from proto import model_pb2_grpc

from grpc_health.v1.health import HealthServicer
from grpc_health.v1 import health_pb2, health_pb2_grpc

from models import CombinedNERModel, CnnNerExtractor

import argparse

model_dict: Dict[str, Model] = {
    "python_combined_ner_model": CombinedNERModel,
    "python_cnn_ner_model": CnnNerExtractor,
}


class ModelServicer(model_pb2_grpc.ModelServicer):
    """Implementation of Model service."""

    def __init__(self, model):
        self.model = model

    def Predict(self, request, context):
        sentence = request.sentence
        preds = self.model.predict(sentence).to_go()
        result = model_pb2.PredictResponse(entities=preds)
        return result


def serve(model_name: str, **kwargs):
    health = HealthServicer()
    health.set("plugin", health_pb2.HealthCheckResponse.ServingStatus.Value("SERVING"))

    # Create the model instance
    model_class = model_dict.get(model_name)
    if model_class is None:
        raise ValueError(f"Model {model_name} not found in model_dict.")

    model = model_class(**kwargs)

    # Start the server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(model), server)
    health_pb2_grpc.add_HealthServicer_to_server(health, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    # Output information using the dynamically assigned port
    hs = f"1|1|tcp|127.0.0.1:{port}|grpc\n"
    sys.__stdout__.write(hs)
    sys.__stdout__.flush()

    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
    except Exception as e:
        server.stop(1)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the NER model gRPC server.")
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model to serve."
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="{}",
        help="Additional keyword arguments for the model in JSON format.",
    )
    args = parser.parse_args()

    model_config = json.loads(args.model_config)

    serve(model_name=args.model_name, **model_config)
