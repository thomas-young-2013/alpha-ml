import os
import json
import logging.config


def setup_logging():
    with open(os.path.join(os.path.dirname(__file__), "logging_configuration.json"), 'r') as logging_configuration_file:
        config_dict = json.load(logging_configuration_file)

        logging.config.dictConfig(config_dict)
