#!/bin/bash

dirs=(
  "1-first-cortex-deployment"
  "2-using-cerebrium-secrets"
  "3-using-webhook-endpoints"
#  "4-faster-inference-with-vllm"
  "5-loading-model-weights-faster"
  "6-multi-gpu-inference"
  "7-streaming-endpoint"
  "8-langchain-QA"
  "9-logo-controlnet"
  "10-sdxl"
  "11-whisper-transcription"
  "12-comfyui"
  "13-tool-calling-langsmith"
  "14-inferentia"
  "16-faster-image-generation"
  "17-winston"
  "18-realtime-voice-agent"
  "19-sdxl-refiner"
  "20-openai-compatible-vllm"
  "21-sdxl-lightning"
)

for dir in "${dirs[@]}"
do
  (
    cd "$dir" || exit
    echo "Running cerebrium deploy -y in $dir"
    cerebrium deploy -y &
  )
done

wait
echo "All deployments finished."