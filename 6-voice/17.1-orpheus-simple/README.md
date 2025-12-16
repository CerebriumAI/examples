# Streaming Orpheus Text-to-Speech

## Deployment

Run "cerebrium deploy" to deploy this model. Feel free to adjust memory and request concurrency based on speed/cost trade-offs.

## Example Usage

Once deployed, you can convert text to speech using:
```
curl --location 'https://api.cloud.crusoe.prd.us-east-1.cerebrium.ai/v4/p-xxxxxx/17-orpheus-server/tts?prompt=Hello%20World&format=pcm%7Cwav'
```
