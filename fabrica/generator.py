import os
import sys
import pdb
import click
import secrets
import difflib
import subprocess
import pandas as pd

from datetime import datetime
from typing import List, Dict, Optional, Iterable
from math import sin, exp
from enum import Enum
from moda.user import MenuOption, UserInteractor
from moda.style import Styled, CustomStyled, Format

from .base import date_format, Operator, control_prefix, calculate_prefix, evaluate_prefix

class GenerateMode(Enum):
  entity = 'entity'
  mmp = 'mmp'
  custom = 'custom'

  @property
  def parameters_file_name(self) -> str:
    if self is GenerateMode.entity:
      return 'entity_parameters'
    elif self is GenerateMode.mmp:
      return 'mmp_parameters'
    elif self is GenerateMode.custom:
      return 'custom_parameters'

class Generator(Enum):
  identity = 'nan'
  sin = 'sin'
  exp = 'exp'
  polynomial_roots = 'pr'
  polynomial_coefficients = 'pc'

  def generate(self, *args: List[any]) -> any:
    if self is Generator.identity:
      return args[0]
    elif self is Generator.sin:
      return sin(args[0])
    elif self is Generator.exp:
      return exp(args[0])
    elif self is Generator.polynomial_roots:
      if len(args) <= 1:
        return 0
      x = args[0]
      f = 1
      for root in args[1:]:
        f = f * (x - float(root))
      return f
    elif self is Generator.polynomial_coefficients:
      x = args[0]
      f = 0
      for power, coefficient in enumerate(args[1:]):
        f = f + float(coefficient) * x**power
      return f

class ControlOption(MenuOption):
  quit = 'q'

  @property
  def option_text(self) -> str:
    if self is ControlOption.quit:
      return '(Q)uit'

  @property
  def styled(self) -> Styled:
    return CustomStyled(text=self.option_text, style=Format().red())

class RepairOption(MenuOption):
  post_mortem = 'p'
  debugger = 'd'
  open = 'o'
  edit = 'e'
  load = 'l'

  @property
  def option_text(self) -> str:
    if self is RepairOption.post_mortem:
      return '(P)ost mortem'
    elif self is RepairOption.debugger:
      return '(D)ebugger'
    elif self is RepairOption.open:
      return '(O)pen Python file'
    elif self is RepairOption.edit:
      return '(E)dit Python file'
    elif self is RepairOption.load:
      return '(L)oad Python file changes and retry'

  @property
  def styled(self) -> Styled:
    return CustomStyled(text=self.option_text, style=Format().yellow())

class FinishRepairOption(MenuOption):
  review = 'r'
  persist = 'p'
  abandon = 'a'

  @property
  def option_text(self) -> str:
    if self is FinishRepairOption.review:
      return '(R)eview repairs'
    elif self is FinishRepairOption.persist:
      return '(P)ersist repairs to source CSV files'
    elif self is FinishRepairOption.abandon:
      return '(A)bandon repairs without saving them'

  @property
  def styled(self) -> Styled:
    return CustomStyled(text=self.option_text, style=Format().yellow())

class EvaluationContext:
  row: pd.Series

  def __init__(self, row: pd.Series):
    self.row = row

class Repair:
  path: str
  context: EvaluationContext
  column: str
  error: Exception
  trace_back: any
  original_code: str

  def __init__(self, path: str, context: EvaluationContext, column: str, error: Exception, trace_back: any):
    self.path = path
    self.context = context
    self.column = column
    self.error = error
    self.trace_back = trace_back
    self.original_code = self.context.row[self.column]

class Factory:
  date_format = date_format

  def random_multiplier(self, percentage: float):
    return 1 + (secrets.randbelow(2001) - 1000) / 1000 * percentage / 100 if percentage != 0 else 1

  def calculate_column(self, expression: str, result: pd.Series, parameters: pd.Series, template: pd.Series):
    parts = expression.split(' ', 2)
    if parts[0] in result.keys():
      operand = result[parts[0]]
    elif parts[0] in parameters.keys():
      operand = parameters[parts[0]]
    elif parts[0] in template.keys():
      operand = template[parts[0]]
    else:
      operand = parts[0]
    if len(parts) == 1:
      return operand
    assert len(parts) == 3
    operator = Operator(parts[1])
    return operator.operate(
      operand,
      self.calculate_column(
        expression=parts[2], 
        result=result, 
        parameters=parameters,
        template=template
      )
    )

class EntityFactory(Factory):
  def generate_entity_results(self, result: pd.Series, day: int, parameters: pd.Series, template: pd.Series) -> pd.Series:
    generator_parameters = str(parameters.generator).split(' ')
    generator = Generator(generator_parameters[0])
    spend_column = template[f'{control_prefix}spend_column'] if f'{control_prefix}spend_column' in template else None
    impressions_column = template[f'{control_prefix}impressions_column'] if f'{control_prefix}impressions_column' in template else None
    clicks_column = template[f'{control_prefix}clicks_column'] if f'{control_prefix}clicks_column' in template else None
    conversions_column = template[f'{control_prefix}conversions_column'] if f'{control_prefix}conversions_column' in template else None
    if spend_column:
      result[spend_column] = (parameters.multiplier * generator.generate(parameters.date_multiplier * (parameters.date_offset + day), *generator_parameters[1:]) + parameters.offset) * self.random_multiplier(parameters.random)
      if impressions_column:
        result[impressions_column] = int(result[spend_column] * parameters.impressions_multiplier * self.random_multiplier(parameters.impressions_random))
        if clicks_column:
          result[clicks_column] = int(result[impressions_column] * parameters.clicks_multiplier * self.random_multiplier(parameters.clicks_random))
          if conversions_column:
            result[conversions_column] = int(result[clicks_column] * parameters.conversions_multiplier * self.random_multiplier(parameters.conversions_random))
    return pd.DataFrame([result])

class MMPFactory(Factory):
  def generate_mmp_results(self, result: pd.Series, date: datetime, parameters: pd.Series, template: pd.Series, entity_result: pd.Series, entity_template: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame()
    users_generator_parameters = str(parameters.users_generator).split(' ')
    users_generator = Generator(users_generator_parameters[0])
    users = (parameters.users_multiplier * users_generator.generate(float(entity_result[parameters.entity_users_column]), *users_generator_parameters[1:]) + parameters.users_offset) * self.random_multiplier(parameters.users_random)

    date_day = (date - datetime.strptime(parameters.start_date, date_format)).days
    date_generator_parameters = str(parameters.date_generator).split(' ')
    date_generator = Generator(date_generator_parameters[0])
    date_factor = (parameters.date_multiplier * date_generator.generate(date_day, *date_generator_parameters[1:]) + parameters.date_offset) * self.random_multiplier(parameters.date_random)
    users = users * date_factor

    start_day = int(parameters.start_day)
    end_day = int(parameters.end_day)
    generator_parameters = str(parameters.generator).split(' ')
    generator = Generator(generator_parameters[0])
    day_column = template[f'{control_prefix}day_column']
    events_column = template[f'{control_prefix}events_column']
    revenue_column = template[f'{control_prefix}revenue_column']
    event_fraction = 0
    first_day = start_day
    if not pd.isna(parameters['start_day_date']):
      start_day_date = datetime.strptime(parameters['start_day_date'], date_format)
      first_day = max(start_day, (start_day_date - date).days)
    if not pd.isna(parameters['end_day_date']):
      end_day_date = datetime.strptime(parameters['end_day_date'], date_format)
      end_day = min(end_day, (end_day_date - date).days)
    
    for day in range(start_day, end_day + 1):
      if users <= 0:
        return df
      events = users * (parameters.multiplier * generator.generate(parameters.day_multiplier * (parameters.day_offset + day), *generator_parameters[1:]) + parameters.offset) * self.random_multiplier(parameters.random) + event_fraction
      if events < 1:
        event_fraction += events
        continue
      events = int(events)
      event_fraction = 0
      repeat_users = (events * parameters.repeat_multiplier + parameters.repeat_offset) * self.random_multiplier(parameters.repeat_random)
      users -= events - repeat_users
      if day < first_day:
        continue
      day_result = result.copy()
      day_result[day_column] = day
      day_result[events_column] = events
      day_result[revenue_column] = max((events * parameters.revenue_multiplier + parameters.revenue_offset) * self.random_multiplier(parameters.revenue_random), 0)
      df = df.append(day_result, sort=False)
    return df

class CustomFactory(Factory):
  parameter_context: EvaluationContext
  template_context: EvaluationContext
  parameters: pd.Series
  template: pd.Series
  source_tables: Dict[str, pd.DataFrame]
  source_templates: Dict[str, pd.DataFrame]
  source_queries: Dict[str, pd.DataFrame]
  source: pd.DataFrame
  source_context: Dict[str, any]
  source_iterator: Iterable
  evaluate_directory: str
  evaluate_cache: Dict[str, str]
  repairs: List[Repair]
  user: UserInteractor

  def __init__(self, parameter_context: EvaluationContext, template_context: EvaluationContext, source_tables: Dict[str, pd.DataFrame], source_templates: Dict[str, pd.DataFrame], source_queries: Dict[str, pd.DataFrame], evaluate_directory: str, source_context: Dict[str, any]={}, source: Optional[pd.DataFrame]=None, source_iterator: Optional[Iterable]=None):
    self.parameter_context = parameter_context
    self.parameters = self.parameter_context.row
    self.template_context = template_context
    self.template = self.template_context.row
    self.sources_tables = source_tables
    self.source_templates = source_templates
    self.source_queries = source_queries
    self.evaluate_directory = evaluate_directory
    self.source_context = {**source_context}
    self.source = source if source is not None else pd.DataFrame()
    self.source_iterator = source_iterator if source_iterator is not None else []
    self.evaluate_cache = {}
    self.repairs = []
    self.user = UserInteractor(locals=self.source_context, timeout=None)

  def clear_evaluate_cache(self):
    self.evaluate_cache = {}    

  def evaluate(self, context: EvaluationContext, column: str, default_value: Optional[any]=None, allow_repair: bool=True, require_repair: bool=False) -> Optional[any]:
    if not context.row[column] or pd.isna(context.row[column]):
      return default_value
    while True:
      code = context.row[column]
      path = None
      path = f'{self.evaluate_directory}/{column}.py'
      if column not in self.evaluate_cache or self.evaluate_cache[column] != code:
        with open(path, 'w') as f:
          f.write(code)
      compiled = compile(code, path, 'exec')
      self.source_context['v'] = default_value
      try:
        exec(compiled, self.source_context, self.source_context)
        if 'v' in self.source_context:
          value = self.source_context['v']
          del self.source_context['v']
        else:
          value = None
        break
      except KeyboardInterrupt:
        raise
      except Exception as e:
        _, _, trace_back = sys.exc_info()
        if allow_repair:
          self.user.present_error(error=e)
          repair = Repair(
            path=path,
            context=context,
            column=column,
            error=e,
            trace_back=trace_back
          )
          self.try_repair(repair=repair)
        if not require_repair:
            raise
          
    return value

  def try_repair(self, repair: Repair):
    self.repairs.append(repair)
    while True:
      choice = self.user.present_menu(
        options=list(RepairOption) + list(ControlOption), 
        default_option=RepairOption.post_mortem,
        message=Format().magenta().bold()(f'An error occurred in {repair.path}\n')
      )
      if choice is RepairOption.post_mortem:
        pdb.post_mortem(t=repair.trace_back)
      elif choice is RepairOption.debugger:
        pdb.set_trace()
      elif choice is RepairOption.open:
        subprocess.call(['open', repair.path])
      elif choice is RepairOption.edit:
        editor = os.environ.get('EDITOR','vi')
        subprocess.call([editor, repair.path])
      elif choice is RepairOption.load:
        with open(repair.path, 'r') as f:
          code = f.read()
          repair.context.row[repair.column] = code
          self.evaluate_cache[repair.column] = code
        break
      elif choice is ControlOption.quit:
        raise click.Abort

  def finish_repairs(self):
    if not self.repairs:
      return
    paths = []
    for repair in self.repairs:
      if repair.path in paths:
        continue
      paths.append(repair.path)
    paths_text = '\n'.join(paths)
    self.user.present_message(f'These files were successfully repaired:\n{paths_text}')
    while True:
      choice = self.user.present_menu(options= list(FinishRepairOption) + list(ControlOption), default_option=FinishRepairOption.review)
      if choice is FinishRepairOption.review:
        paths = []
        for repair in self.repairs:
          if repair.path not in paths:
            paths.append(repair.path)
        for path in paths:
          path_repairs = list(filter(lambda r: r.path == path, self.repairs))
          self.user.present_message(f'{len(path_repairs)} repairs made to \n{paths_text}')
          original_lines = [l + '\n' for l in path_repairs[0].original_code.split('\n')]
          final_lines = [l + '\n' for l in path_repairs[-1].context.row[path_repairs[-1].column].split('\n')]
          diff_text = ''.join(difflib.unified_diff(original_lines, final_lines))
          self.user.present_message(f'{diff_text}\n\n')
      elif choice is FinishRepairOption.abandon:
        break
      elif choice is ControlOption.quit:
        raise click.Abort
      else:
        # TODO: implement remaining repair options
        self.user.present_message(Format().red()('Sorry, this option is not yet implemented'))
    self.repairs = []

  def generate_custom_results(self, table: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame()
    self.source_context['table'] = table
    self.source_context['results'] = results
    evaluate_columns = [c for c in self.template.keys() if c.startswith(evaluate_prefix)]
    for source_index in self.source_iterator:
      result = self.template.copy()
      self.source_context['i'] = source_index
      self.source_context['r'] = result
      evaluated_order = []
      allow_repair = False
      while True:
        evaluated = []
        for column in evaluate_columns:
          try:
            result[column[len(evaluate_prefix):]] = self.evaluate(context=self.template_context, column=column, allow_repair=allow_repair)
            allow_repair = False
            evaluated.append(column)
          except (KeyboardInterrupt, click.Abort):
            raise
          except Exception:
            pass
        if evaluated:
          evaluated_order += evaluated
          evaluate_columns = list(filter(lambda c: c not in evaluated, evaluate_columns))
        elif evaluate_columns:
          allow_repair = True
        else:
          break
      evaluate_columns = evaluated_order
      results = results.append([result])
      self.finish_repairs()
    return results
