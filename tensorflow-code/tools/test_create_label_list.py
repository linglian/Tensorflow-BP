import create_label_list

if __name__ == '__main__':
    import sys
    import getopt
    path = None
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
    for op, value in opts:
        if op == '-f':
            path = value
    
    def display(things):
        print create_label_list.getFileName(things)
    if path is not None:
        create_label_list.traverse_floder(path, display, '.py')
    else:
        print('Please Enter Path')