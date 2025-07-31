# PayPal MCP Voice Agent

A voice-enabled AI agent with PayPal MCP (Model Context Protocol) integration for handling PayPal transactions and invoicing through natural conversation.

## Setup

### 1. PayPal Configuration

1. Go to [PayPal Developer Dashboard](https://developer.paypal.com/)
2. Create a new app to get your Client ID and Client Secret
3. Set up your environment variables in a `.env` file:

```env
# PayPal Configuration
PAYPAL_CLIENT_ID=your_paypal_client_id_here
PAYPAL_CLIENT_SECRET=your_paypal_client_secret_here
PAYPAL_ACCESS_TOKEN=your_generated_access_token_here
PAYPAL_ENVIRONMENT=sandbox

# Voice Services
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key

# Daily.co for video calls
DAILY_TOKEN=your_daily_token

# OpenAI
OPENAI_API_KEY=your_openai_api_key
```

### 2. Generate PayPal Access Token

For server-side authentication (no browser required), run:

```bash
python get_paypal_token.py
```

This will generate an access token using your Client ID and Secret, and automatically add it to your `.env` file.

### 3. Install Dependencies

```bash
pip install pipecat-ai[silero, daily, openai, deepgram, cartesia, mcp] python-dotenv loguru aiohttp requests
npm install -g @paypal/mcp
```

### 4. Run the Agent

Uncomment the file
```bash
python pipecat_agent/main.py
```

## 5. Deploy on Cerebrium
Run
```bash
pip install cerebrium
cerebrium login
cerebrium deploy
```

It should return a deployment url that you can then hit in order to join a meeting room. The request should be something like:
```
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxx/pipecat-agent/start_bot' \
--header 'Authorization: Bearer <CEREBRIUM_API_TOKEN>' \
--header 'Content-Type: application/json' \
--data '{"paypal_access_token": <PAYPAL_ACCESS_TOKEN>, "paypal_environment": <PAYPAL_ENV>}'
```

## How It Works

The agent uses:
- **Pipecat** for the voice pipeline (speech-to-text, LLM, text-to-speech)
- **PayPal MCP** for transaction and invoice management
- **Daily.co** for WebRTC voice calls
- **Deepgram** for speech recognition
- **Cartesia** for text-to-speech
- **OpenAI GPT-4** for conversation
- **Cerebrium** for deploying and scaling

## Available PayPal Functions

Through MCP, the agent can:
- List transactions
- Create invoices
- Check payment status
- And more PayPal API functions


