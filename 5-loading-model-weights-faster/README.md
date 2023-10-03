# Faster loading of model weights into GPU
In this example we show you how to utilise a library called **Tensorizer** to load weights from disk into GPU memory faster.  
This is extremely useful with large models where the biggest bottlenecks to cold-start time is model weight loading.

## Requirements
- tensorizer

More information on tensorizer can be found in their github repo [here](
    <!-- TODO add tensorizer github repo -->
)

## Implementation
For implementation instructions, read the comments in the main.py of this folder. 