import json

from const import BASE_DIR


def get_service_config() -> dict:
    """Получить конфигурацию для сервиса."""

    config_path = BASE_DIR / "dist" / "config.json"
    with open(str(config_path), "r", encoding="UTF-8") as f:
        config = json.load(f)

    return config


def get_mapping_config() -> dict:
    """Получить конфигурацию для маппинга."""

    config_path = BASE_DIR / "dist" / "isolation_mapping.json"
    with open(str(config_path), "r", encoding="UTF-8") as f:
        config = json.load(f)

    return config
