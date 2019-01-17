""" Hello World.
Prints "Hello, world."
"""

from __future__ import print_function

import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mlops', type=int, dest='use_mlops', default=0)
    args = parser.parse_args()

    if args.use_mlops > 0:
        ## MLOps start
        from parallelm.mlops import mlops
        from parallelm.mlops.stats_category import StatCategory
        
        # Initialize the mlops library
        mlops.init()
        
        # 
        mlops.set_stat("Testing set_stat", 50)
        
        # Release mlops resources
        mlops.done()
        ## MLOps end
    
    print("Hello, world")

if __name__ == "__main__":
    main()
