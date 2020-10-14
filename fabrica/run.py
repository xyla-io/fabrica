import os
import io
import sys
import stat
import glob
import click
import shlex
import subprocess
import pandas as pd

from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

from fabrica import date_format, control_prefix, calculate_prefix, Operator, GenerateMode, EntityFactory, MMPFactory, CustomFactory, EvaluationContext, Verifier
from data_layer import Redshift as SQL
from .config import get_sql_config
from moda import style, log
from moda.command import invoke_subcommand
from subir import Uploader

def load_template(templates_path: str, template_name: str, template_dfs: Dict[str, pd.DataFrame]):
  if template_name in template_dfs.keys():
    return
  df = pd.read_csv(f'{templates_path}/{template_name}.csv', dtype='object')
  if f'{control_prefix}target_id_column' in df.columns:
    df[f'{control_prefix}target_id'] = df.apply(lambda r: r[r[f'{control_prefix}target_id_column']], axis=1)
  template_dfs[template_name] = df

def load_source_table(source_path: str, table_name: str, source_dfs: Dict[str, pd.DataFrame]):
  if table_name in source_dfs.keys():
    return
  source_file_paths = sorted([f for f in glob.glob(f'{source_path}/**/{table_name}.csv', recursive=True)], key=lambda p: len(p.split('/')))
  df = pd.DataFrame()
  for path in source_file_paths:
    df = df.append(pd.read_csv(path, dtype='object'), sort=False)
  print(f'Loaded {len(df)} rows from {len(source_file_paths)} source files for table {table_name} at\n' + '\n'.join(source_file_paths))
  source_dfs[table_name] = df

def load_query(queries_path: str, query_name: str, query_dfs: Dict[str, pd.DataFrame], format_parameters: Dict[str, str], data_layer: SQL.Layer):
  if query_name in query_dfs.keys():
    return
  query_file_paths = sorted([f for f in glob.glob(f'{queries_path}/**/{query_name}.sql', recursive=True)], key=lambda p: len(p.split('/')))
  df = pd.DataFrame()
  for path in query_file_paths:
    with open(path) as f:
      query = f.read()
    if format_parameters:
      query = query.format(**format_parameters)
    query_df = pd.read_sql_query(query, con=data_layer.connection)
    df = df.append(query_df, sort=False)
  print(f'Loaded {len(df)} rows from {len(query_file_paths)} query files for query {query_name} at\n' + '\n'.join(query_file_paths))
  query_dfs[query_name] = df

def quote_command(run_args: str):
  return ' '.join(shlex.quote(a) for a in run_args)

all_modes = [m.value for m in GenerateMode]

class Fabrica:
  database: str
  schema: str
  user: Optional[str]
  password: Optional[str]

  def __init__(self, database: str, schema: str, user: Optional[str], password: Optional[str]):
    self.database = database
    self.schema = schema
    self.user = user
    self.password = password

  @property
  def database_name(self) -> str:
    return get_sql_config()[self.database]['database']

  def configure_data_layer(self):
    sql_config = get_sql_config()
    for database in sql_config:
      if self.user is not None:
        sql_config[database]['user'] = self.user
      if self.password is not None:
        sql_config[database]['password'] = self.password
    sql_config = get_sql_config()
    database_options = sql_config[self.database]
    SQL.Layer.configure_connection(options=database_options)

@click.group(name='run')
@click.option('-db', '--database', 'database', type=str, default='default')
@click.option('-s', '--schema', 'schema', type=str, default='demo')
@click.option('-u', '--database-user', 'database_user', type=str)
@click.option('-p', '--database-password', 'database_password', type=str)
@click.pass_context
@invoke_subcommand()
def run(ctx: any, database: str, schema: str, database_user: Optional[str], database_password: Optional[str]):
  fabrica = Fabrica(
    database=database,
    schema=schema,
    user=database_user,
    password=database_password
  )
  fabrica.configure_data_layer()
  ctx.obj = fabrica

@run.command()
@click.option('-t', '--templates', 'templates_path', type=str, default='input/templates')
@click.option('-q', '--queries', 'queries_path', type=str, default='input/queries')
@click.option('-s', '--sources', 'source_path', type=str, default='output/csv')
@click.option('-r', '--results', 'results_path', type=str, default='output/csv')
@click.option('-p', '--parameters', 'parameters_path', type=str, default='input/parameters')
@click.option('-m', '--mode', 'mode_values', type=click.Choice(all_modes), multiple=True, default=all_modes)
@click.pass_obj
def generate(
  context: Fabrica,
  templates_path: str,
  queries_path: str,
  source_path: str,
  results_path: str,
  parameters_path: str,
  mode_values: List[str],
):
  template_dfs = {}

  for mode in GenerateMode:
    if mode.value not in mode_values:
      continue
    result_dfs = {}
    parameters_file_path = f'{parameters_path}/{mode.parameters_file_name}.csv' 
    parameters_df = pd.read_csv(parameters_file_path)
    if 'disabled' not in parameters_df.columns:
      parameters_df['disabled'] = False
    parameters_df.disabled = parameters_df.disabled.apply(lambda v: v and not pd.isna(v))
    log.log(f'Generating data for {len(parameters_df)} parameter rows.')
    for index, parameters in parameters_df.iterrows():
      if parameters.disabled:
        print(f'Skipping disabled parameter row {index}')
        continue
      parameters_df = parameters_df[parameters_df.disabled == False]
      load_template(
        templates_path=templates_path,
        template_name=parameters.template,
        template_dfs=template_dfs
      )
      template_df = template_dfs[parameters.template]
      template_df = template_df[template_df[f'{control_prefix}target_id'].str.match(str(parameters.target_id))]
      log.log(style.Format().green()(f'Generating data for paramerter row {index} ({len(template_df)} template rows)'))
      log.log(style.Format().cyan()(parameters))
      table = pd.DataFrame()
      source_templates = {}
      if parameters.source_templates and not pd.isna(parameters.source_templates):
        for source_template in parameters.source_templates.split(' '):
          load_template(
            templates_path=templates_path,
            template_name=source_template,
            template_dfs=source_templates
          )
      source_tables = {}
      if parameters.source_tables and not pd.isna(parameters.source_tables):
        for source_table in parameters.source_tables.split(' '):
          load_source_table(
            source_path=source_path,
            table_name=source_table,
            source_dfs=source_tables
          )
      source_queries = {}
      if parameters.source_queries and not pd.isna(parameters.source_queries):
        layer = SQL.Layer()
        layer.connect()
        for source_query in parameters.source_queries.split(' '):
          load_query(
            queries_path=queries_path,
            query_name=source_query,
            query_dfs=source_queries,
            format_parameters={'SCHEMA': f'{context.schema}.'},
            data_layer=layer
          )
        layer.disconnect()
      for _, template in template_df.iterrows():
        if mode is GenerateMode.custom:
          parameter_context = EvaluationContext(row=parameters)
          template_context = EvaluationContext(row=template)
          factory = CustomFactory(
            parameter_context=parameter_context,
            template_context=template_context,
            source_tables=source_tables,
            source_templates=source_templates,
            source_queries=source_queries,
            evaluate_directory='output/python/evaluate',
          )
          factory.source_context = factory.evaluate(
            context=parameter_context,
            column='source_context',
            default_value={
              'pd': pd,
              'factory': factory,
              'parameters': parameters,
              'template': template,
              'source_tables': source_tables,
              'source_templates': source_templates,
              'source_queries': source_queries,
            },
            require_repair=True
          )
          # TODO: assemble source from source templates, output files, and query sources
          factory.source = factory.evaluate(context=parameter_context, column='source', default_value=pd.DataFrame(), require_repair=True)
          factory.source_context['s'] = factory.source
          factory.source_iterator = factory.evaluate(context=parameter_context, column='source_iterator', default_value=[], require_repair=True)
          factory.source_context['source_iterator'] = factory.source_iterator
          factory.finish_repairs()
          results = pd.DataFrame()
          results = results.append(factory.generate_custom_results(
            table=results
          ))
          results.drop(columns=[k for k in results.columns if k.startswith(control_prefix)], inplace=True)
          table_name = parameters.table if not pd.isna(parameters.table) else parameters.template
        else:
          table_name = template[f'{control_prefix}table']
          start_date = datetime.strptime(parameters.start_date, date_format)
          end_date = datetime.strptime(parameters.end_date, date_format)
          days = (end_date - start_date).days + 1
          print(f'Generating data for {days} days for parameter row {index} with generated daily row counts:')
          for day in range(0, days):     
            result = template.copy()
            date_column = template[f'{control_prefix}date_column'] if f'{control_prefix}date_column' in template else None
            date = start_date + timedelta(days=day)
            if date_column:
              result[date_column] = date.strftime(date_format)   
      
            if mode is GenerateMode.entity:
              factory = EntityFactory()
              results = factory.generate_entity_results(
                result=result,
                day=day,
                parameters=parameters,
                template=template
              )
            elif mode is GenerateMode.mmp:
              factory = MMPFactory()
              load_template(
                templates_path=templates_path,
                template_name=parameters.entity_template,
                template_dfs=template_dfs
              )
              entity_template_df = template_dfs[parameters.entity_template]
              entity_template = entity_template_df.iloc[0]
              load_source_table(
                source_path=results_path, 
                table_name=entity_template[f'{control_prefix}table'],
                source_dfs=source_tables
              )
              source_df = source_tables[entity_template[f'{control_prefix}table']]
              source_df = source_df[source_df[entity_template[f'{control_prefix}date_column']] == date.strftime(date_format)]
              results = pd.DataFrame()
              for _, entity_result in source_df.iterrows():
                results = results.append(factory.generate_mmp_results(
                  result=result.copy(),
                  date=date,
                  parameters=parameters,
                  template=template,
                  entity_result=entity_result,
                  entity_template=entity_template
                ), sort=False)
              if not results.empty:
                results[template[f'{control_prefix}events_column']] = results[template[f'{control_prefix}events_column']].astype(pd.Int64Dtype())
                results[template[f'{control_prefix}day_column']] = results[template[f'{control_prefix}day_column']].astype(pd.Int64Dtype())

        calculated_columns = [c for c in template.keys() if c.startswith(calculate_prefix)]
        for calculated_column in calculated_columns:
          results[calculated_column[len(calculate_prefix):]] = results.apply(
            lambda r: factory.calculate_column(
              expression=str(template[calculated_column]),
              result=r,
              parameters=parameters,
              template=template
            ),
            axis=1
          )
        results.drop(columns=[k for k in results.columns if k.startswith(control_prefix)], inplace=True)
        table = table.append(results, sort=False)
        sys.stdout.write(f'{len(results)}.')
        sys.stdout.flush()

      if table_name not in result_dfs.keys():
        result_dfs[table_name] = pd.DataFrame()
      result_dfs[table_name] = result_dfs[table_name].append(table, sort=False)
      print(f'\n{table.iloc[-1] if len(table) else None}\n{len(table)} rows generated for row {index + 1} in {parameters_file_path}\nfor table {table_name} using template {parameters.template}')
      
    for table, df in result_dfs.items():
      path = f'{results_path}/{table}.csv'
      df.to_csv(path, index=False)
      print(f'{len(df)} total rows generated for table {table} at {path}')

@run.command()
@click.option('-r', '--results', 'results_path', type=str, default='output/csv')
@click.option('-m', '--merge', 'merge_columns', type=str, multiple=True)
@click.option('-t', '--table', 'tables', type=str, multiple=True)
@click.option('-d', '--drop', 'should_drop', is_flag=True)
@click.pass_obj
def upload(
  context: Fabrica,
  results_path: str,
  database_name: str,
  merge_columns: List[str],
  tables: List[str],
  should_drop: bool
):
  database_name = context.database_name
  schema = context.schema

  result_file_paths = {f: os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{results_path}/**/*.csv', recursive=True)}
  result_file_text = '\n'.join(f'{k} ——> {result_file_paths[k]}' for k in sorted(result_file_paths.keys()) if not tables or result_file_paths[k] in tables)
  confirm_action = 'drop and replace' if should_drop else 'append'
  if not click.prompt(
    f'Confirm {confirm_action} demo data to schema {schema} in database {database_name} from files\n{result_file_text}', 
    type=click.Choice(['y', 'n']), 
    confirmation_prompt=True
  ) == 'y':
    return

  uploader = Uploader()
  action = 'Truncating table and uploading' if should_drop else 'Adding'
  for path in sorted(result_file_paths.keys()):
    table_name = result_file_paths[path]
    if tables and table_name not in tables:
      log.log(f'Skipping table {table_name}')
      continue  
    df = pd.read_csv(path, dtype='object')
    print(f'{action} {len(df)} rows to {schema}.{table_name} in database {database_name}')
    uploader.upload_data_frame(
      schema_name=schema,
      table_name=table_name,
      merge_column_names=merge_columns,
      data_frame=df,
      column_type_transform_dictionary={},
      replace=should_drop
    )

@run.command()
@click.option('-db', '--database', 'database', type=str)
@click.option('-db1', '--database-1', 'database_a', type=str)
@click.option('-db2', '--database-2', 'database_b', type=str)
@click.option('-csv1', '--csv-1', 'csv_a', is_flag=True)
@click.option('-csv2', '--csv-2', 'csv_b', is_flag=True)
@click.option('-e1/-E1', '--escape-1/--no-escape-1', 'escape_a', is_flag=True, default=True)
@click.option('-e2/-E2', '--escape-2/--no-escape-2', 'escape_b', is_flag=True, default=True)
@click.option('-s', '--script', 'script_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('-s1', '--script-1', 'script_path_a', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('-s2', '--script-2', 'script_path_b', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('-c', '--column', 'columns', type=str, multiple=True)
@click.option('-xc', '--exclude-column', 'exclude_columns', type=str, multiple=True)
@click.option('-c1', '--column-1', 'columns_a', type=str, multiple=True)
@click.option('-xc1', '--exclude-column-1', 'exclude_columns_a', type=str, multiple=True)
@click.option('-c2', '--column-2', 'columns_b', type=str, multiple=True)
@click.option('-xc2', '--exclude-column-2', 'exclude_columns_b', type=str, multiple=True)
@click.option('-i/-I', '--interactive/--no-interactive', 'interactive', is_flag=True, default=True)
@click.option('-v/-V', '--verbose/--no-verbose', 'verbose', is_flag=True, default=True)
@click.option('-o', '--output', 'output_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('-in', '--input-directory', 'input_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.option('-df', '--diff-tool', 'diff_tool', type=str, default='vimdiff')
@click.option('-at', '--absolute-tolerance', type=float, default=0)
@click.option('-rt', '--relative-tolerance', type=float, default=0)
@click.option('-fn', '--format-name', 'format_names', multiple=True)
@click.option('-fv', '--format-value', 'format_values', multiple=True)
@click.argument('path_a')
@click.argument('path_b')
@click.pass_obj
def verify(context: Fabrica, database: Optional[str], database_a: Optional[str], database_b: Optional[str], csv_a: bool, csv_b: bool, escape_a: bool, escape_b: bool, script_path: Optional[str], script_path_a: Optional[str], script_path_b: Optional[str], columns: Tuple[str], columns_a: Tuple[str], columns_b: Tuple[str], exclude_columns: Tuple[str], exclude_columns_a: Tuple[str], exclude_columns_b: Tuple[str], interactive: bool, verbose: bool, output_directory: Optional[str], input_directory: Optional[str], diff_tool: str, absolute_tolerance: float, relative_tolerance: float, format_names: Tuple[str], format_values: Tuple[str], path_a: str, path_b: str):
  assert len(format_names) == len(format_values)
  file_a = click.File()(os.path.join(input_directory, path_a) if input_directory else path_a)
  file_b = click.File()(os.path.join(input_directory, path_b) if input_directory else path_b)

  verifier = Verifier(
    database=database if database is not None else context.database,
    interactive=interactive,
    verbose=verbose,
    output_directory=output_directory if output_directory else os.path.join('output', 'verify'),
    diff_command=diff_tool
  )
  verification = verifier.verify(
    name_a=os.path.splitext(os.path.basename(file_a.name))[0],
    name_b=os.path.splitext(os.path.basename(file_b.name))[0],
    stream_a=file_a,
    stream_b=file_b,
    script_path=script_path,
    script_path_a=script_path_a,
    script_path_b=script_path_b,
    database_a=database_a,
    database_b=database_b,
    csv_a=csv_a,
    csv_b=csv_b,
    escape_a=escape_a,
    escape_b=escape_b,
    columns=list(columns) if columns else None,
    columns_a=list(columns_a) if columns_a else None,
    columns_b=list(columns_b) if columns_b else None,
    exclude_columns=list(exclude_columns) if exclude_columns else None,
    exclude_columns_a=list(exclude_columns_a) if exclude_columns_a else None,
    exclude_columns_b=list(exclude_columns_b) if exclude_columns_b else None,
    format_parameters={
      'SCHEMA': context.schema,
      **dict(zip(format_names, format_values)),
    },
    absolute_tolerance=absolute_tolerance,
    relative_tolerance=relative_tolerance
  )

  if verification.success:
    print('Verification succeeded.')
  else:
    print('Verification FAILED.')
    if not interactive:
      raise click.ClickException('Non-interactive verification failed.')

@run.group()
def docker():
  pass

@docker.command()
@click.option('-c/-C' '--cache/-no-cache', 'cache', is_flag=True, default=True)
def build(cache: bool=True):
  run_args = [
    'docker',
    'build',
    *(['--no-cache'] if not cache else []),
    '-t', 'xyla/fabrica',
    '-f', 'Dockerfile',
    '..',
  ]
  print(quote_command(run_args=run_args))
  subprocess.call(args=run_args)

@docker.command(name='run')
@click.option('-l', '--link', 'link_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True, readable=True))
@click.argument('fabrica_args', nargs=-1)
def docker_run(link_directory: Optional[str], fabrica_args: List[str]):
  symlink_args = ['-v', f'{os.path.realpath(link_directory)}:/host/fabrica'] if link_directory is not None else []
  output_args = ['-o', '/host/fabrica'] if link_directory else []
  run_args = [
    'docker',
    'run',
    *symlink_args,
    '-it',
    '--rm',
    '--name', 'fabrica',
    'xyla/fabrica',
    *output_args,
    *fabrica_args,
  ]
  print(quote_command(run_args=run_args))
  subprocess.call(args=run_args)

@docker.command()
@click.option('--host-path', 'host_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True))
def setup(host_path: Optional[str]):
  if host_path is None:
    print('''Welcome to fabrica on docker
To continue setup, please run the following command:

mkdir fabrica && cd fabrica && docker run -v \"`pwd`\":/host/fabrica -it --rm --name fabrica xyla/fabrica docker setup --host-path /host/fabrica
'''
    )
    return

  user = click.prompt('Please enter a database user name')
  password = click.prompt('Please enter the database user\'s password', hide_input=True)

  script_path = os.path.join(host_path, 'verify.sh')
  open(script_path, 'a').close()
  os.chmod(script_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
  with open(script_path, 'w') as f:
    backslash = '\\'
    f.write(f'''
#!/bin/bash

cd "$( dirname "$0" )"
SCRIPT_DIR=$(pwd)

docker run -v "$SCRIPT_DIR":/host/fabrica -it --rm --name fabrica xyla/fabrica -u "{user.replace('"', f'{backslash}"')}" -p "{password.replace('"', f'{backslash}"')}" verify -in /host/fabrica -o /host/fabrica "$@"
'''
    )
  with open(os.path.join(host_path, 'x.sql'), 'w') as f:
    f.write('select 0;')
  with open(os.path.join(host_path, 'y.sql'), 'w') as f:
    f.write('select 1;')
  
# mkdir verify && cd verify && echo 'select 0;' > x.sql && echo 'select 1;' > y.sql && touch verify.sh && chmod 700 verify.sh && echo "docker run -v \"`pwd`\":/host/fabrica -it --rm --name fabrica xyla/fabrica -u \"{user}\" -p \"{password}\" verify -in /host/fabrica -o /host/fabrica \\\"\\$@\\\"" > verify.sh

  print('''You can now run a comparison by running the command:

./verify.sh x.sql y.sql

You can see all available options by running:

./verify.sh --help'''
  )
  