
import sys
from get_terminal_size import get_terminal_size
import numpy

class ConsoleError(Exception):
    pass

colours = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'}

def colour(m, colour_name):
    return "\033[" + colours[colour_name] + "m" + m + "\033[0m"

def bold(m):
    return u'\033[1m%s\033[0m' % m

def pretty_print_times(times, lengths, functions, highlight=None):

    if highlight == 'min':
        highlight_positions = numpy.argmin(times, 0)
    elif highlight == 'max':
        highlight_positions = numpy.argmax(times, 0)
    else:
        highlight_positions = None

    if highlight_positions is not None:
        highlighter = lambda m: colour(m, 'green')
    else:
        highlighter = None

    t_width, t_height = get_terminal_size()

    func_str_length = 0
    for each_function in functions:
        if len(each_function) > func_str_length:
            func_str_length = len(each_function)

    func_str_length += 1

    number_length = max(len('%1.3g' % (1/3.0*1e-6)), 
            max(len(str(length)) for length in lengths)) + 1

    sub_table_width = min(
        len(lengths),
        (t_width - func_str_length)//number_length)
    sub_table_entries = [sub_table_width]

    if sub_table_width == 0:
        raise ConsoleError('Console is too narrow to display the results')

    while sub_table_entries[-1] < len(lengths):
        sub_table_width = min(
            (t_width - func_str_length)//number_length, 
            len(lengths) - sub_table_entries[-1])

        sub_table_entries.append(sub_table_entries[-1] + sub_table_width)

    def print_sub_table(start_idx, end_idx):
        
        sys.stdout.write(' '.ljust(func_str_length))

        for each_length in lengths[start_idx:end_idx]:
            sys.stdout.write(colour(str(each_length).rjust(number_length), 
                'cyan'))

        sys.stdout.write('\n')

        for n, (each_function, each_timeset) in enumerate(zip(functions, times)):
            sys.stdout.write(bold(each_function.ljust(func_str_length)))
            for idx in range(start_idx,end_idx):
                each_time = each_timeset[idx]
                time_str = '%1.3g' % each_time
                if highlighter is not None and n == highlight_positions[idx]:
                    print_str = highlighter(time_str.rjust(number_length))
                else:
                    print_str = time_str.rjust(number_length)

                sys.stdout.write(print_str)

            sys.stdout.write('\n')

    print_sub_table(0, sub_table_entries[0])
    for n in range(1, len(sub_table_entries)):
        sys.stdout.write('\n')        
        print_sub_table(sub_table_entries[n-1], sub_table_entries[n])

    sys.stdout.write('\n')

