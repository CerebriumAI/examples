from daily import Daily
import asyncio
import logging
from translator import TranslationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


async def run(room_url, target_language, user_name, user_id):
    """
    Run the translation service in a Daily room

    Args:
        room_url: URL of the Daily room to join
        target_language: Target language for translation
        user_name: Name of the user joining the room
        user_id: ID of the user joining the room

    Returns:
        dict: Status message
    """
    logger.info(f"Starting translation service for room: {room_url}")
    logger.info(f"Target language: {target_language}")
    logger.info(f"User ID: {user_id}")

    Daily.init()

    translation_service = TranslationService(
        room_url, target_language, user_name, user_id
    )

    translation_service.join(room_url)

    try:
        while translation_service.is_running():
            await asyncio.sleep(10)
            logger.debug("Service running...")
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        logger.exception("Stack trace:")
    finally:
        translation_service.leave()
        logger.info("Service stopped")

    return {"message": "Translation service has completed"}
