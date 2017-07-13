from __future__ import print_function

import json


class Config(object):
    @staticmethod
    def save_to_json(config):
        return {}

    def log_config(self, log_file):
        json.dump(self, log_file, default=self.save_to_json, indent=4)


if __name__ == '__main__':
    a = Config()
    a.log_config('1.json')
