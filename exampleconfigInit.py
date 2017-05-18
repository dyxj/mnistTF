#! python3
"""
exampleconfigInit.py
Initialize cofig data. ie: file paths etc
"""

import configparser


def createCfg():
    config = configparser.ConfigParser()
    config.add_section('main')
    config.set('main', 'key1', 'value1')
    config.set('main', 'key2', 'value2')
    config.set('main', 'key3', 'value3')
    config.set('main', 'key4', 'value4')

    config['DEFAULT'] = {'ServerAliveInterval': '',
                         'Compression': 'yes',
                         'CompressionLevel': '9'}

    config['bitbucket.org'] = {}
    config['bitbucket.org']['User'] = 'hg'

    with open('config.ini', 'w') as f:
        config.write(f)


def loadCfg():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print(config.get('main', 'key1'))
    print(config.get('main', 'key2'))
    print(config.get('main', 'key3'))
    print(config['bitbucket.org']['User'])
    print(config['bitbucket.org']['ServerAliveInterval'])
    print(float(config['bitbucket.org']['ServerAliveInterval']))
    print(config.getfloat('main', 'ServerAliveInterval'))
    # getfloat() raises an exception if the value is not a float
    # a_float = config.getfloat('main', 'a_float')
    # getint() and getboolean() also do this for their respective types
    # an_int = config.getint('main', 'an_int')


if __name__ == "__main__":
    createCfg()
    loadCfg()
