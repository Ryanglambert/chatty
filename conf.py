import os
import yaml


BASE = os.path.dirname(os.path.realpath(__file__))
FNAME = os.path.join(BASE, 'conf.yaml')

def _load_configs():
    with open(FNAME, 'rb') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def _parse_envs(configs):
    try:
        env = os.environ['ENV'].lower()
    except KeyError:
        print("You haven't set your \'ENV\' variable")
    env_configs = dict()
    for k, v in configs.items():
        new_value_conf = v[env]
        env_configs[k] = new_value_conf
    return env_configs 
        

conf = _parse_envs(_load_configs())
