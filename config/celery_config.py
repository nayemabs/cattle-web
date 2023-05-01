import os
from functools import lru_cache
from kombu import Queue
from dotenv import dotenv_values


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "celery"}


class BaseConfig:
    config = dotenv_values(".env")
    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL",config["CELERY_BROKER_URL"])
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", config["CELERY_RESULT_BACKEND"])
    FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

    CELERY_TASK_QUEUES: list = (
        # default queue
        Queue("celery"),
        # custom queue
        Queue("cattle"),
        
    )

    CELERY_TASK_ROUTES = (route_task,)


class DevelopmentConfig(BaseConfig):
    pass


@lru_cache()
def get_settings():
    config_cls_dict = {
        "development": DevelopmentConfig,
    }
    config_name = os.environ.get("CELERY_CONFIG", "development")
    config_cls = config_cls_dict[config_name]
    return config_cls()


settings = get_settings()
