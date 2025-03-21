import grpc

import example_pb2
import example_pb2_grpc


def run():
    with grpc.insecure_channel('p-9d1122a3-grpc-server.tenant-cerebrium-prod.svc.cluster.local:50051') as channel:
        stub = example_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(example_pb2.HelloRequest(name="Alice"))
        print("Server response:", response.message)
