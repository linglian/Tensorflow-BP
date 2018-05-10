from tools import create_label_list

file = open('list.txt', 'w')

def make_label(file_path):
    if file_path is None:
        raise ValueError('Please Check File\'s Path Not Be None', file_path)
    
    id_val = create_label_list.getFloderOfFile(file_path)
    file.write('{} {}\n'.format(file_path, id_val))

if __name__ == '__main__':
    import sys
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
    path = None
    for op, value in opts:
        if op == '-f':
           path = value 
    if path is None:
        raise ValueError('Must Enter Path Use -f', path)
        
    create_label_list.traverse_floder(path, make_label, 'py')
