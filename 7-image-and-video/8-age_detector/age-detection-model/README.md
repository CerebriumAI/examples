# Age Detection Model API

This repository contains a FastAPI-based age detection model that can be deployed on Cerebrium.

## Model Details
The model uses ViT (Vision Transformer) to classify images into age ranges:
- 0-2 years
- 3-9 years  
- 10-19 years
- 20-29 years
- 30-39 years
- 40-49 years
- 50-59 years
- 60-69 years
- 70-79 years
- 80+ years

## Deployment Instructions

1. Install the Cerebrium CLI: `pip install --upgradecerebrium`
2. Login to Cerebrium: `cerebrium login`
3. Run the deployment command: `cerebrium deploy`
