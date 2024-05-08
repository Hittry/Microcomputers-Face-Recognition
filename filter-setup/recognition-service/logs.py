from loguru import logger

from const import BASE_DIR


def prepare_logging(log_dir_path: str) -> None:
    """Функция для подготовке логгера."""

    log_path = BASE_DIR / log_dir_path
    log_path.mkdir(exist_ok=True, parents=True)

    logger.add(
        log_path / "file_{time}.log",
        rotation="10 days",
        level="INFO",
        format="{time} {name} {level} {message}",
    )
