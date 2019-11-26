
from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'dgaintel',         # How you named your package folder (MyLib)
  packages = ['dgaintel'],   # Chose the same as "name"
  version = '0.9',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Extremely fast and accurate predictions of whether a domain name is genuine or DGA with deep learning.',   # Give a short description about your library
  long_description = long_description,
  long_description_content_type='text/markdown',
  author = 'Rushil Mallarapu',                   # Type in your name
  author_email = 'rushil.mallarapu@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/sudo-rushil/dgaintel',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/sudo-rushil/dgaintel/archive/v0.9.tar.gz',    # I explain this later on
  keywords = ['DGA', 'Domain', 'Domain Generation', 'Domain Classifier', 'Deep Learning', 'AI', 'RNN', 'LSTM', 'CNN-LSTM', 'CNN'],
  include_package_data=True,
  install_requires=[            # I get to this in a second
          'numpy',
          'tensorflow',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ])