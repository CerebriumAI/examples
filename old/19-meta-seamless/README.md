Steps to deploy on Cerebrium

1. Git clone the seamless-communication repo into this folder using the command: git clone https://github.com/facebookresearch/seamless_communication.git
2. Make sure you have added your AWS access key and secret to your secrets page on Cerebrium under the names: aws-access-key, aws-secret-key and meta-seamless-bucket. This
is to upload the audio files to a s3 bucket
3. Deploy this model using the command: cerebrium deploy <NAME> --config-file ./config.yaml