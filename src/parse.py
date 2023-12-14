import argparse
from parser.parser import create

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group measurement data by cell type')
    parser.add_argument('-p', '--src_path', type=str, required=True)
    parser.add_argument('-i', '--info_path', type=str, required=True)
    parser.add_argument('-d', '--dest_path', type=str, required=True)
    parser.add_argument('-c','--cell_types', nargs='+', required=True)
    
    args = parser.parse_args()
    create(args.src_path, args.info_path, args.dest_path, args.cell_types)