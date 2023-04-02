import logging
from datetime import timedelta


class Config:
    # debug mode
    DEBUG = True
    # session expire time
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)


class DevelopmentConfig(Config):
    # development mode
    DEBUG = True
    LOG_LEVEL = logging.DEBUG


class ProductionConfig(Config):
    # production mode
    # close development
    DEBUG = False
    LOG_LEVEL = logging.ERROR


# config dict
config_dict = {
    'dev': DevelopmentConfig,
    'pro': ProductionConfig
}
