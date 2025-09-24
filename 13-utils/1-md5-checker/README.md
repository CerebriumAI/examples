# Example usage:

```bash
 curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/p-xxx/1-md5-checker/return_md5' \
 --header 'Authorization: Bearer ' \
 --header 'Content-Type: application/json' \
 --data '{
     "file_path": "/persistent-storage/file_name.ext"
 }'
 ```

# Example response:

```json
{
  "run_id": "44a0a684-e4ef-9077-8782-9b8f701511fe",
  "result": "607c4c95caf0a6c570368ae3f4ee01c1",
  "run_time_ms": 6472.740411758423
}
```