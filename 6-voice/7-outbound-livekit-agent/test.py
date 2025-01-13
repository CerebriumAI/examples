import asyncio
from main import create_sip_participant

async def main(number: str):
    room_name = "Test SIP Room"
    
    result = await create_sip_participant(number, room_name)
    print(result)  

if __name__ == '__main__':
    asyncio.run(main("<phone number to call>"))
