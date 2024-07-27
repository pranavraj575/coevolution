import argparse, os, sys

DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

data_folder = os.path.join(DIR, 'data', 'pyquaticus_coevolution')

print(DIR)
PARSER = argparse.ArgumentParser()
