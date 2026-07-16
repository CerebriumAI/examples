# Voice Web Agent (LiveKit + Cerebrium + Linkup)

A low-latency voice agent that chats normally and searches the live web when a
query needs fresh facts. Built on LiveKit Agents, served with a fast MoE LLM on
Cerebrium, and grounded with Linkup `fast` search.

- **STT:** Deepgram nova-3
- **LLM:** Qwen3.6-35B-A3B (SGLang + DFLASH on a Cerebrium B200)
- **TTS:** Cartesia sonic-2
- **Web search:** Linkup `fast`
- **Turn detection:** LiveKit `EnglishModel`

## Local dev

```bash
cd 6-voice/19-voice-webagent
cp .env.example .env          # fill in API keys
pip install -r requirements.txt
python main.py download-files # once: VAD + turn detector weights
python main.py dev            # join via LiveKit Agent Console
```

## Deploy to Cerebrium

The app runs from the included `Dockerfile` (see `cerebrium.toml`).

- Upload secrets (.env) to Cerebrium
```bash
cerebrium deploy
```

## Read more

https://cerebrium.ai/blog/a-low-latency-architecture-for-voice-agents-with-live-web-retrieval
