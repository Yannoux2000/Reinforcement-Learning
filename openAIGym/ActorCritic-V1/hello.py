import argparse
parser = argparse.ArgumentParser()

parser.add_argument("lr",help="Learning Rate")
parser.add_argument("il",help="Intermediate_layer")
parser.add_argument("gamma",help="Gamma also called discount rate")
parser.add_argument("beta",help="Representing additionnal exploration")

args = parser.parse_args()
print args.lr, args.il, args.gamma, args.beta