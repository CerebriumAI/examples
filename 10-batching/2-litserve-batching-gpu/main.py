import litserve as ls
from transformers import pipeline


class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        print("setup")
        self.model = pipeline(
            "sentiment-analysis", model="stevhliu/my_awesome_model", device=device
        )

    def decode_request(self, request):
        print("decode_request")
        return request["text"]

    def predict(self, x):
        print("predict")
        result = self.model(x)
        return result

    def encode_response(self, output):
        print("encode_response")
        return output["label"]


api = TextClassificationAPI()
server = ls.LitServer(api, max_batch_size=4, batch_timeout=20)
app = server.app


#  health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    server.run(port=8000)
