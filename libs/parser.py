import argparse

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('data', metavar = 'DATA', default='/home/shared/sagnik/datasets/MNIST', \
                            help='path to the dataset')
        parser.add_argument('no_class', metavar = 'CLASS', default=10, type=int, help='number of output classes')
        self.parser=parser

    def get_parser(self):
        return self.parser