"""
Task orchestration.
"""

def prepare():
  # TODO: Add prepare function docstring
  
  data_train = []
  data_val = []
  data_test = []
  print('Data preparation for this model.')
  return data_train, data_val, data_test

class B2:
  
  def __init__(self, **kwargs):
    pass
  
  def __str__(self):
    print('The B2 model.')
  
  def train(self, params):
    """Training process for this model.

    Args:
        params (dict): Specifications for a given testing 'run' of this model.
    """
    self.params = params
    print('Training B2 model with these params: {0}.'.format(self.params))
  
  def test(self, params):
    """Testing process for this model.

    Args:
        params (dict): Specifications for a given training 'run' of this model.
    """
    self.params = params
    print('Testing B2 model with these params: {0}.'.format(self.params))
  
  