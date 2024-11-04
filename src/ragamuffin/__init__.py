import logging

from rich.logging import RichHandler

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_path=False)])
logging.getLogger("cassandra").setLevel(logging.ERROR)
