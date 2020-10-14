import os
import json

from pathlib import Path
from typing import Dict
from data_layer import Redshift, Mongo

def get_config(name: str):
  environment_key = f'FABRICA_{name.upper()}_CONFIG_PATH'
  config_path = os.getenv(environment_key)
  if config_path is None:
    config_path = Path(__file__).parent.parent / 'config' / 'sql_config.json'
  with open(config_path) as f:
    return json.load(f)

def get_sql_config() -> Dict[str, any]:
  return get_config('sql')

def get_config_for_database(database_name: str) -> Dict[str, any]:
  return get_sql_config()[database_name]

def get_sql_for_database(database_name: str, configure: bool=True) -> any:
  database_config = get_config_for_database(database_name=database_name)
  if database_config['schema'] == 'redshift+psycopg2':
    SQL = Redshift
  elif database_config['schema'] == 'mongodb':
    SQL = Mongo
  else:
    raise ValueError('Unsupported database schema.', database_config['schema'])
  SQL.Layer.configure_connection(options=database_config if configure else None)

  return SQL