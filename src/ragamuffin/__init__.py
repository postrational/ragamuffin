import logging

from rich.logging import RichHandler

from ragamuffin import settings

if settings.get_settings().get("debug_mode"):
    logging.basicConfig(
        level="DEBUG", format="[%(name)s] %(message)s", datefmt="[%X]", handlers=[RichHandler(show_path=True)]
    )
else:
    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_path=False)])
    logging.getLogger("cassandra").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

logger.info("🐈 Starting Ragamuffin...")
logger.info("Herding our llamas: 🦙🦙🦙...\n")
