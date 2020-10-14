from __future__ import annotations

import os
import io
import click
import json
import difflib
import datacompy
import subprocess
import pandas as pd
import bson
import decimal

from .config import get_sql_for_database, get_config_for_database
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, OrderedDict as OrderedDictType, Optional
from datetime import date, datetime
from moda import style, log
from moda.user import UserInteractor, PythonShellType, MenuOption, Interaction
from functools import reduce

class ResultEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, date):
      return obj.isoformat()
    if isinstance(obj, datetime):
      return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
      return float(obj)
    # Let the base class default method raise the TypeError
    return json.JSONEncoder.default(self, obj)

def run_sql(SQL: any, query_text: str, escape_file_text: bool, format_parameters: Optional[Dict[str, any]], verbose: bool):
  if escape_file_text:
    query_text = SQL.Query.escaped_query_text(query_text=query_text)
  elif format_parameters is not None:
    query_text = SQL.Query.formatted_query_text(
      query_text=query_text,
      format_parameters=format_parameters
    )
  if verbose:
    log.log(f'...running query:\n{query_text}')
  query = SQL.Query(query_text)
  layer = SQL.Layer()
  layer.connect()
  cursor = query.run(sql_layer=layer)
  records = layer.fetch_all_records(cursor=cursor)
  layer.disconnect()
  return records

def write_json(file_path:str, results: List[OrderedDictType[str, any]]):
  with open(file_path, 'w', encoding='utf-8') as f:
    f.write('[\n')
    for result in results:
      f.write(f'{json.dumps(result, cls=ResultEncoder)}\n')
    f.write(']')

class Verifier:
  class Option(MenuOption):
    diff = 'v'
    quit = 'q'

    @property
    def option_text(self) -> str:
      if self is Verifier.Option.diff:
        return '(V)iew JSON diff'
      elif self is Verifier.Option.quit:
        return '(Q)uit'

    @property
    def styled(self) -> style.Styled:
      if self is Verifier.Option.diff:
        return style.CustomStyled(text=self.option_text, style=style.Format().blue())
      if self is Verifier.Option.quit:
        return style.CustomStyled(text=self.option_text, style=style.Format().red())

  class Verification:
    verification_date: datetime
    name_a: str
    name_b: str
    data_frame_a: pd.DataFrame
    data_frame_b: pd.DataFrame
    csv_path_a: Optional[str]
    csv_path_b: Optional[str]
    json_path_a: Optional[str]
    json_path_b: Optional[str]
    diff_path: Optional[str]
    success: bool

    def __init__(self, verification_date: datetime, name_a: str, name_b: str, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame, csv_path_a: Optional[str], csv_path_b: Optional[str], json_path_a: Optional[str], json_path_b: Optional[str], diff_path: Optional[str], success: bool):
      self.verification_date = verification_date
      self.name_a = name_a
      self.name_b = name_b
      self.data_frame_a = data_frame_a
      self.data_frame_b = data_frame_b
      self.json_path_a = json_path_a
      self.json_path_b = json_path_b
      self.csv_path_a = csv_path_a
      self.csv_path_b = csv_path_b
      self.diff_path = diff_path
      self.success = success

  database: str
  user: UserInteractor
  verbose: bool
  output_directory: str
  diff_command: str

  def __init__(self, database: str, interactive: bool=False, verbose: bool=False, output_directory: str=os.path.join('output', 'verify'), diff_command: str='vimdiff', python_shell_type: PythonShellType=PythonShellType.ipython):
    self.database = database
    self.user = UserInteractor(timeout=None, interactive=interactive, python_shell_type=python_shell_type)
    self.verbose = verbose
    self.output_directory = output_directory
    self.diff_command = diff_command

  def filter_columns(self, df: pd.DataFrame, columns: Optional[List[str]]=None, exclude_columns: Optional[List[str]]=None) -> pd.DataFrame:
    if columns is None and not exclude_columns:
      return df.copy()
    final_columns = [
      c
      for c in df.columns
      if (columns is None or c in columns)
      and (exclude_columns is None or c not in exclude_columns)
    ]
    if not final_columns:
      return pd.DataFrame()
    return df[final_columns]

  def get_data_frame(self, text: Optional[str], stream: Optional[str], database: Optional[str], csv: bool, escape: bool, columns: Optional[List[str]], exclude_columns: Optional[List[str]], format_parameters: Dict[str, any]) -> pd.DataFrame:
    SQL = get_sql_for_database(database_name=database if database else self.database)
    if csv:
      df = pd.read_csv(stream if stream is not None else io.StringIO(text))
    else:
      df = pd.DataFrame(run_sql(
        SQL=SQL,
        query_text=text if text is not None else stream.read(),
        escape_file_text=escape,
        format_parameters=format_parameters,
        verbose=self.verbose
      ))
    df = self.filter_columns(
      df=df,
      columns=columns,
      exclude_columns=exclude_columns
    )
    return df

  def combine_columns(self, *column_lists: List[Optional[List[str]]]) -> Optional[List[str]]:
    column_lists = list(filter(lambda l: l is not None, column_lists))
    return reduce(lambda l, m: l + [c for c in m if c not in l], column_lists, []) if column_lists else None

  def apply_script(self, script_path: str, database: str, data_frame: pd.DataFrame, other_database: Optional[str]=None, other_data_frame: Optional[pd.DataFrame]=None, context: Optional[Dict[str, any]]=None) -> List[pd.DataFrame]:
    user_interactive = self.user.interactive
    user_locals = self.user.locals
    user_script_directory_components = self.user.script_directory_components

    self.user.script_directory_components = list(Path(script_path).parent.parts)
    self.user.locals = {
      **self.user.locals,
      'pd': pd,
      'bson': bson,
      'dfs': [
        data_frame,
        other_data_frame,
      ],
      'database_configs': [
        get_config_for_database(database_name=database),
        get_config_for_database(database_name=other_database) if other_database else None,
      ],
      'SQLs': [
        get_sql_for_database(
          database_name=database,
          configure=False
        ),
        get_sql_for_database(
          database_name=other_database,
          configure=False
        ) if other_database else None
      ],
      'context': context if context is not None else {},
    }
    modified_data_frames = self.user.locals['dfs']
    self.user.interactive = False
    if self.verbose:
      script_text = Path(script_path).read_text()
      log.log(f'...running script:\n{script_text}')
    self.user.run_script(script_name=Path(script_path).stem)

    self.user.script_directory_components = user_script_directory_components
    self.user.locals = user_locals
    self.user.interactive = user_interactive
    return modified_data_frames

  def verify(self, name_a: str, name_b: str, text_a: Optional[str]=None, text_b: Optional[str]=None, stream_a: Optional[io.TextIOBase]=None, stream_b: Optional[io.TextIOBase]=None, script_path: Optional[str]=None, script_path_a: Optional[str]=None, script_path_b: Optional[str]=None, database_a: Optional[str]=None, database_b: Optional[str]=None, csv_a: bool=False, csv_b: bool=False, escape_a: bool=False, escape_b: bool=False, columns: Optional[List[str]]=None, columns_a: Optional[List[str]]=None, columns_b: Optional[List[str]]=None, exclude_columns: Optional[List[str]]=None, exclude_columns_a: Optional[List[str]]=None, exclude_columns_b: Optional[List[str]]=None, format_parameters: Dict[str, any]={}, absolute_tolerance: float=0, relative_tolerance: float=0) -> Verification:
    if self.verbose:
      detail_a = f'database: {database_a if database_a else self.database}' if not csv_a else 'file: csv'
      detail_b = f'database: {database_b if database_b else self.database}' if not csv_b else 'file: csv'
      log.log(f'Comparing:\na: {name_a} ({detail_a})\nb: {name_b} ({detail_b})\n')
    script_context = {
      'format_parameters': format_parameters
    }

    df_a = self.get_data_frame(
      text=text_a,
      stream=stream_a,
      database=database_a,
      csv=csv_a,
      escape=escape_a,
      columns=self.combine_columns(columns, columns_a),
      exclude_columns=self.combine_columns(exclude_columns, exclude_columns_a),
      format_parameters=format_parameters
    )
    if script_path_a:
      script_context['df_a'] = df_a
      df_a = self.apply_script(
        script_path=script_path_a,
        database=database_a if database_a else self.database,
        data_frame=df_a,
        context=script_context
      )[0]

    df_b = self.get_data_frame(
      text=text_b,
      stream=stream_b,
      database=database_b,
      csv=csv_b,
      escape=escape_b,
      columns=self.combine_columns(columns, columns_b),
      exclude_columns=self.combine_columns(exclude_columns, exclude_columns_b),
      format_parameters=format_parameters
    )
    if script_path_b:
      script_context['df_a'] = df_a
      script_context['df_b'] = df_b
      df_b = self.apply_script(
        script_path=script_path_b,
        database=database_b if database_b else self.database,
        data_frame=df_b,
        context=script_context
      )[0]

    if script_path:
      script_context['df_a'] = df_a
      script_context['df_b'] = df_b
      df_a, df_b = self.apply_script(
        script_path=script_path,
        database=database_a if database_a else self.database,
        data_frame=df_a,
        other_database=database_b if database_b else self.database,
        other_data_frame=df_b,
        context=script_context
      )
    verification_date = datetime.utcnow()
    results_a = df_a.to_dict(orient='records')
    results_b = df_b.to_dict(orient='records')

    compare = datacompy.Compare(
      df_a,
      df_b,
      on_index=True,
      df1_name=f'{name_a} [a]',
      df2_name=f'{name_b} [b]',
      # join_columns='acct_id',  #You can also specify a list of columns
      abs_tol=absolute_tolerance, #Optional, defaults to 0
      rel_tol=relative_tolerance #Optional, defaults to 0
      )
    if self.verbose:
      log.log(f'\n{compare.report()}')

    output_path_a = os.path.join(self.output_directory, f'comparison_a_{name_a}')
    output_path_b = os.path.join(self.output_directory, f'comparison_b_{name_b}')

    csv_path_a = f'{output_path_a}.csv'
    csv_path_b = f'{output_path_b}.csv'
    df_a.to_csv(csv_path_a)
    df_b.to_csv(csv_path_b)
    json_path_a = f'{output_path_a}.json'
    json_path_b = f'{output_path_b}.json'
    write_json(json_path_a, results_a)
    write_json(json_path_b, results_b)
    if self.verbose:
      log.log(f'CSV files written to\n{csv_path_a}\n{csv_path_b}\n')
      log.log(f'JSON result files written to\n{json_path_a}\n{json_path_b}\n')

    matched = compare.matches(ignore_extra_columns=False)
    if not matched:
      with open(json_path_a) as f:
        lines_a = f.readlines()
      with open(json_path_b) as f:
        lines_b = f.readlines()
      diff_lines = list(difflib.unified_diff(lines_a, lines_b))
      diff_path = os.path.join(self.output_directory, f'comparison_diff__a_{name_a}__b_{name_b}.txt')
      with open(diff_path, 'w') as f:
        f.write(''.join(diff_lines))
      log_diff_lines = [
        *diff_lines[:10],
        f'\n... ({len(diff_lines) - 20} lines not displayed)\n\n',
        *diff_lines[-10:],
      ] if len(diff_lines) > 24 else diff_lines
      log.log(f'JSON result file diff written to {diff_path}\n\n{"".join(log_diff_lines)}\n\n')

    self.user.present_message(message='The results appear to MATCH.' if matched else 'The results DO NOT APPEAR TO MATCH.')
    while True:
      option = self.user.present_menu(
        options=[
          Verifier.Option.diff,
          Interaction.python,
          Interaction.debugger,
          Verifier.Option.quit
        ],
        default_option=Verifier.Option.diff if self.user.interactive and not matched else Verifier.Option.quit
      )
      if option is Verifier.Option.diff:
        subprocess.run([self.diff_command, json_path_a, json_path_b])
      elif isinstance(option, Interaction):
        self.user.locals = {
          **self.user.python_locals,
          'df1': df_a,
          'df2': df_b,
        }
        self.user.interact(interaction=option)
        self.user.locals = {}
      elif option is Verifier.Option.quit:
        break

    verification = Verifier.Verification(
      verification_date=verification_date,
      name_a=name_a,
      name_b=name_b,
      data_frame_a=results_a,
      data_frame_b=results_b,
      csv_path_a=csv_path_a,
      csv_path_b=csv_path_b,
      json_path_a=json_path_a,
      json_path_b=json_path_b,
      diff_path=diff_path if not matched else None,
      success=matched
    )
    return verification

if __name__ == "__main__":
  run()
