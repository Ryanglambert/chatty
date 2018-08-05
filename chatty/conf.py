import os
import yaml


CONF_YAML_PATH = os.environ.get('CHATTY_CONF')

if not CONF_YAML_PATH:
    raise EnvironmentError("CHATTY_CONF should point to your conf.yml file")

def _load_configs():
    with open(CONF_YAML_PATH, 'rb') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def _parse_envs(configs):
    try:
        env = os.environ['ENV'].lower()
        if env == '':
            raise EnvironmentError("ENV environment variable should be set")
    except KeyError:
        raise EnvironmentError("ENV environment variable should be set")

    env_configs = dict()
    for k, v in configs.items():
        new_value_conf = v[env]
        env_configs[k] = new_value_conf

    return env_configs 
        

conf = _parse_envs(_load_configs())
