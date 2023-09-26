"""
Collection of functions for manipulating files, saving data, etc.
"""

import os
import argparse

def maybe_mkdir(fp, force=True):
    if not os.path.exists(fp):
        os.mkdir(fp)
    elif not force:
        x = input('{} already exists. Hit enter to continue and overwrite. Q to exit.'.format(fp))
        if x.lower() == 'q':
            exit(0)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
