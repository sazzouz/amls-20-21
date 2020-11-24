"""
Task orchestration.
"""
import os
import shutil
from .preprocessing import load_data, process_data, gen_data

def prepare(task_dir, celeba_img_dir, celeba_df):
  # TODO: Add prepare function docstring
  # # Create a dataframe for the celeba labels.csv
  gender_train_df, gender_test_df = load_data(celeba_df)
  print('\n>>> Generating Training Dataset In Temporary Directory\n')
  train_is_valid, train_is_not_valid = process_data(task_dir, celeba_img_dir, gender_train_df, 'train', 'gender')
  print('\n>>> Generating Testing Dataset In Temporary Directory\n')
  test_is_valid, test_is_not_valid = process_data(task_dir, celeba_img_dir, gender_test_df, 'test', 'gender')
  return gen_data(task_dir)

def cleanup(task_dir):
  print('\n>>> Removing Temporary Train/Test Image Directories\n')
  for folder in ['train', 'test']:
    if os.path.exists(os.path.join(task_dir, folder)) and os.path.isdir(os.path.join(task_dir, folder)):
      try:
          shutil.rmtree(os.path.join(task_dir, folder))
      except OSError:
          print ("Failed To Remove Directory %s" % current_gender_output_path)

class A1:
  
  def __init__(self, **kwargs):
    pass
  
  def __str__(self):
    print('The A1 Orchestration Class.')
  
  def train(self, params):
    """Training process for this model.

    Args:
        params (dict): Specifications for a given testing 'run' of this model.
    """
    self.params = params
    print('Training A1 model with these params: {0}.'.format(self.params))
    acc_A1_train = 'ACC %'
    return acc_A1_train
  
  def test(self, params):
    """Testing process for this model.

    Args:
        params (dict): Specifications for a given training 'run' of this model.
    """
    self.params = params
    print('Testing A1 model with these params: {0}.'.format(self.params))
  
  