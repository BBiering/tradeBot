
import logging
import os
import sys


LOG_CONFIG = {
    'name': 'tradebot_logger',
    'level': logging.INFO,
    # 'stream_handler': logging.StreamHandler(sys.stdout),
    'file_handler': logging.FileHandler(filename='tradebot_log.txt'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%d-%b-%y %H:%M:%S'
}

POLLING_CONFIG = {
    'yahoo_interval': 30,
}

FINNHUB_CONFIG = {
    'api_key': os.environ['FINNHUB_API_KEY']
}

TRADING_CONFIG = {
    'user': os.environ[''],
    'password': os.environ[''],
    'api_key': os.environ['']
}