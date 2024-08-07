import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoConfig

# Set up the paths to the model cache
APP_NAME = os.environ.get(
    "APP_NAME", "default"
)  # this gets the name of your deployment from the environment variables
private_model_path = f"/persistent-storage/.cache/tensorised-models/{APP_NAME}/{{org}}/{{model_name}}.tensors"


def fast_load(model_id, load_weights_func, faster=False):
    """A function that loads a model from the cache if it exists, or serialises it if it doesn't
    Args:
        model_id (str): The model id in the form of org/model_name
        load_weights_func (function): Some function where you load your model, send it to GPU and prep it for inference
        faster (bool, optional): Whether to use the faster method of loading the model. Defaults to False.
    """
    org, model_name = model_id.split("/")
    model_path = private_model_path.format(org=org, model_name=model_name)

    # Check if the model exists in the cache
    if os.path.isfile(model_path):
        print(
            f"Deserializing model from {model_path}", file=sys.stderr
        )  # printing to stderr prints the output immediately. Otherwise, it will be buffered.
        model = deserialize_saved_model(model_path, model_id, plaid=faster)
    else:
        # some function where you load your model, send it to GPU and prep it for inference
        model = load_weights_func()
        print(f"Serialising model to {model_path}", file=sys.stderr)
        serialise_model(model, model_path)
    return model


def deserialize_saved_model(model_path, model_id, plaid=True):
    """Deserialize the model from the model_path and load into GPU memory"""
    from tensorizer import TensorDeserializer
    from tensorizer.utils import no_init_or_tensor

    # create a config object that we can use to init an empty model
    config = AutoConfig.from_pretrained(model_id)

    # Init an empty model without loading weights into gpu. We'll load later.
    print("Initialising empty model", file=sys.stderr)
    start = time.time()
    with no_init_or_tensor():
        # Load your model here using whatever class you need to initialise an empty model from a config.
        # In this example, we're using a transformer for causal LM
        model = AutoModelForCausalLM.from_config(config)
    end_init = time.time() - start

    # Create the deserializer object
    #   Note: plaid_mode is a flag that does a much faster deserialization but isn't safe for training.
    #    -> only use it for inference.
    deserializer = TensorDeserializer(model_path, plaid_mode=True)

    # Deserialize the model straight into GPU (zero-copy)
    print("Loading model", file=sys.stderr)
    start = time.time()
    deserializer.load_into_module(model)
    end = time.time()
    deserializer.close()

    # Report on the timings.
    print(f"Initialising empty model took {end_init} seconds", file=sys.stderr)
    print(f"\nDeserializing model took {end - start} seconds\n", file=sys.stderr)

    return model


def serialise_model(model, model_path):
    """Serialise the model and saving the weights to the model_path"""
    from tensorizer import TensorSerializer

    try:
        serializer = TensorSerializer(model_path)
        start = time.time()
        serializer.write_module(model)
        end = time.time()
        print(f"Serialising model took {end - start} seconds", file=sys.stderr)
        serializer.close()
        return True
    except Exception as e:
        print("Serialisation failed with error: ", e, file=sys.stderr)
        return False
