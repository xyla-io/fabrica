from enum import Enum
from datetime import datetime, timedelta

date_format = '%Y-%m-%d'
control_prefix = 'flx_'
calculate_prefix = f'{control_prefix}calculate_'
evaluate_prefix = f'{control_prefix}eval_'

class Operator(Enum):
  additon = '+'
  subtraction = '-'
  multiplication = '*'
  division = '/'
  dateadd = 'dateadd'

  def operate(self, a: any, b: any) -> any:
    if self is Operator.additon:
      return float(a) + float(b)
    elif self is Operator.subtraction:
      return float(a) - float(b)
    elif self is Operator.multiplication:
      return float(a) * float(b)
    elif self is Operator.division:
      return float('nan') if float(b) == 0 else float(a) / float(b)
    elif self is Operator.dateadd:
      return datetime.strptime(a, date_format) + timedelta(days=float(b))
