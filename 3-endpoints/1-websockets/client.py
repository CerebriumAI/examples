import websockets
import asyncio


async def main():
    async with websockets.connect("ws://localhost:5555/ws") as websocket:
        while True:
            await websocket.send("alive?")
            response = await websocket.recv()
            print(response)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
