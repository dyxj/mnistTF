#! python3
"""
configInit.py
Initialize cofig data. ie: file paths etc
"""

import configparser


def createIni():
    """
    mlpath = machine learning data path of your choosing
    mnist = mnist dataset path in mlpath
    :return:
    """
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'mlpath': 'E:\mldata',
                         'mnist': 'mnist'}

    config.add_section('main')
    config.set('main', 'mlpath', 'E:\mldata')
    config.set('main', 'mnist', 'mnist')

    with open('config.ini', 'w') as f:
        config.write(f)


def loadIni():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


if __name__ == "__main__":
    createIni()
    config = loadIni()
    print(config.get('main', 'mlpath'))
    print(config.get('main', 'mlpath'), config.get('main', 'mnist'))
