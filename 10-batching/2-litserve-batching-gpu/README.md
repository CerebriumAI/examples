## Things to note

- torch is required

## How to call it

Positive sentiment

```bash
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/<your-app-name>/predict' \
--header 'Authorization: Bearer <your-api-key>' \
--header 'Content-Type: application/json' \
--data '{"text": "this product is great"}'
```

Negative sentiment

```bash
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/<your-app-name>/predict' \
--header 'Authorization: Bearer <your-api-key>' \
--header 'Content-Type: application/json' \
--data '{"text": "this thing is the worst"}'
```
