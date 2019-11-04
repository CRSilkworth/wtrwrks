from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
  long_description = f.read()

class CustomDependencyInstallCommand(install):
  """Customized setuptools install command - install all dependencies."""

  @staticmethod
  def _define_dependencies():
    """
    Returns a list of dependency repo urls for pip to download
    Any version updates or additional dependencies should be added here
    """
    dependency_links = [
        'nltk==3.3'
    ]

    return dependency_links

  def run(self):
    """
    Override original run function of install command,
    with actions to install dependencies
    """
    # go through all dependencies defined by user above
    for dep in self._define_dependencies():
      # install them with pip
      subprocess.call('pip install ' + dep, shell=True)

    install.run(self)

    # MIGHT NOT RAISE EXCEPTION IF IT FAILS
    # Install some nltk stuff
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')


setup(
  name='wtrwrks',
  description='A system for creating reversible data transformations',
  long_description=long_description,
  long_description_content_type='text/markdown',
  version='2.0.5',
  author='Christopher Silkworth',
  author_email='crsilkworth@gmail.com',
  packages=find_packages(exclude=['unit_test', 'unitTest', '*.unitTest',
                                  'unitTest.*', '*.unitTest.*']),
  install_requires=[
    'numpy~=1.14',
    'pandas~=0.23',
    'matplotlib~=2.2',
    'scipy~=1.1',
    'mpu~=0.14',
    'tables~=3.4',
    'tensorflow~=1.14',
    'nltk~=3.4',
    'tinysegmenter~=0.4',
    'chop~=1.0',
    'konlpy~=0.5',
    'simplejson~=3.16',
    'gensim~=3.6',
    'dill~=0.3',
    'sqlalchemy~=1.3',
    'pathos~=0.2'
  ],
  cmdclass={
      'install': CustomDependencyInstallCommand
  }
)
