from setuptools import setup,find_packages

dependencies = ['numpy','torch','torchvision','matplotlib']

setup(
  name='torch_lego',
  version='1.0',
  url='https://github.com/ma3oun/torch-lego.git',
  package_dir={"":"src"},
  include_package_data=True,
  install_requires=dependencies,
  description='Build torch modules from yaml',
)
