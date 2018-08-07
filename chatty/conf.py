import os
import yaml


CONF_YAML_PATH = os.environ.get('CHATTY_CONF_PATH')

if not CONF_YAML_PATH:
    raise EnvironmentError("CHATTY_CONF_PATH should point to your conf.yml file")

def _load_configs():
    with open(CONF_YAML_PATH, 'rb') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def _parse_envs(configs):
    # Get what environment we're in from shell variable 'ENV'
    try:
        env = os.environ['ENV'].lower()
        if env == '':
            raise EnvironmentError("ENV environment variable should be set")
    except KeyError:
        raise EnvironmentError("ENV environment variable should be set")

    # Load all configs into dictionary
    env_configs = dict()
    for k, v in configs.items():
        new_value_conf = v[env]
        env_configs[k] = new_value_conf

    # if in production should be providing SECRET_KEY
    try:
        SECRET_KEY = os.environ['SECRET_KEY']
        if SECRET_KEY == '':
            raise EnvironmentError(
                "SECRET_KEY environment variable should be set\n"
                "to something more than an empty string"
                )
    except KeyError:
        raise EnvironmentError("SECRET_KEY environment variable should be set")
    env_configs['api']['SECRET_KEY'] = SECRET_KEY


    return env_configs 


conf = _parse_envs(_load_configs())
