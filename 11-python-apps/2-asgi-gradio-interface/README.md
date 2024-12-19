# Cerebrium ASGI Example - Gradio (BETA)

This example demonstrates how to deploy a Cerebrium app using an ASGI server and Gradio.

Warning: This functionality is in beta and may change in the future.

## Notes 

- This application uses Gradio as the frontend
- The model is hosted as a separate instance due to the differing hardware and uptime requirements between our model and the frontend. We'd like our Gradio frontend to be always-on and running on a CPU, and for our model to scale and start as necessary on GPU instances.

## Required changes

### cerebrium.toml
- The `port` should be set to the port on which the ASGI server will run. Note that your requests to cerebrium will still be made to port 443.
- The `entrypoint` should be set to the command that starts the ASGI server.
- The code lives in the `/cortex` directory and this is also the entrypoint workdir.
- The `entrypoint` should point to the ASGI server in the `main.py` file. For Uvicorn, the entrypoint should be
  `uvicorn main:app ...`

  ```toml
  [cerebrium.runtime.custom]
  port = 5000
  entrypoint = "uvicorn main:app --host 0.0.0.0 --port 5000"
  healthcheck_endpoint = "/health"
  ```
  
## Running your application
- Make the necessary changes listed in the _Required changes_ section above
- Deploy your application using `cerebrium deploy -y`
- In your browser, navigate to `https://dev-api.cortex.cerebrium.ai/v4/<your-project-id>/34-asgi-gradio-interface/` to interact with your frontend.
