from concurrent import futures
import sys
import time

import grpc

from proto import model_pb2
from proto import model_pb2_grpc

from grpc_health.v1.health import HealthServicer
from grpc_health.v1 import health_pb2, health_pb2_grpc

class ModelServicer(model_pb2_grpc.ModelServicer):
    """Implementation of Model service."""

    def Predict(self, request, context):
        sentence = request.sentence
        result = model_pb2.PredictResponse(entities=[{"label": "label", "text": "text", "start": 0, "end": 4}]*5000)
        return result

def serve():
    health = HealthServicer()
    health.set("plugin", health_pb2.HealthCheckResponse.ServingStatus.Value('SERVING'))

    # Start the server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
    health_pb2_grpc.add_HealthServicer_to_server(health, server)
    port = server.add_insecure_port('127.0.0.1:0') 
    server.start()

    # Output information using the dynamically assigned port
    print(f"1|1|tcp|127.0.0.1:{port}|grpc")
    sys.stdout.flush()


    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
    except Exception as e:
        server.stop(1)
        raise

if __name__ == '__main__':
    serve()