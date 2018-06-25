import os


environment = os.getenv('ENV')

env_settings = {
    'LOCAL': {
        'SECRET_KEY': 'r_e42%@*n9a=2xx&y$szh-(c7qe=s6s8xxd%lbzp#s9ogp-dra',
        'DEBUG': True
    },
    'DEV': {
        # go and get from aws parameter store
        # 'SECRET_KEY': 'not set',
        'DEBUG': True
    },
    'STAGING': {
        # go and get from aws parameter store
        # 'SECRET_KEY': 'not set',
        'DEBUG': True
    },
    'PROD': {
        # go and get from aws parameter store
        # 'SECRET_KEY': 'not set',
        'DEBUG': False
    },
}


if environment not in env_settings.keys():
    raise EnvironmentError("{} is not setup yet are you sure you're in {}"
                           .format(environment, environment))
SETTINGS = env_settings.get(environment)
SECRET_KEY = SETTINGS['SECRET_KEY']
DEBUG = SETTINGS['DEBUG']


