import os

if not os.getcwd().endswith('mob2crime'):
    os.chdir('..')
os.getcwd()

from src.mex_helper import *

tower_vor()
