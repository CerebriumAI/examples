import litserve as ls
import numpy as np


class SimpleBatchedAPI(ls.LitAPI):
    def setup(self, device):
        print("setup")
        self.model = lambda x: x ** 2

    def decode_request(self, request):
        print("decode_request")
        return np.asarray(request["input"])

    def predict(self, x):
        print("predict")
        result = self.model(x)
        return result

    def encode_response(self, output):
        print("encode_response")
        return {"output": output}


api = SimpleBatchedAPI()
server = ls.LitServer(api, max_batch_size=3, batch_timeout=20)
app = server.app


#  health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    server.run(port=8000)
