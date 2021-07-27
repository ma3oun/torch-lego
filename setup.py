from setuptools import setup,find_packages

dependencies = ['numpy','torch','torchvision','matplotlib']

setup(
  name='torch_lego',
  version='1.0',
  url='https://github.com/ma3oun/torch-lego.git',
  packages=find_packages(),
  include_package_data=False,
  install_requires=dependencies,
  description='Build torch modules from yaml',
  zip_safe=True
)
