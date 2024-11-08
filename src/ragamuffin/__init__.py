import logging

from rich.logging import RichHandler
from rich.text import Text

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_path=False)])
logging.getLogger("cassandra").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

logger.info("🐈 Starting Ragamuffin...")
logger.info("Herding our llamas: 🦙🦙🦙...")
