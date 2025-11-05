#!/usr/bin/python

from nostril import nonsense


def clean_string(input_string):
    # input_string =input_string.strip()

    # Define the pattern to remove unwanted characters
    # pattern2 = r'\([^)]*\)'

    # input_string = re.sub(pattern2, '', input_string)

    # pattern = r'[^a-zA-Z0-9\s]'
    pattern = '\!@#$%^&*'

    # Use the pattern to remove unwanted characters
    if any(c in pattern for c in input_string):
        return True
    else:
        return False


def detect(string):
    value = True

    if string.startswith('[Reg') or string.startswith('[T'):
        value = False
    elif clean_string(string):
        value = False
    elif nonsense(string):
        value = False
    else:
        value = True
    #print(value)
    return value


print(detect(' STYLEREF  "Heading 2"  \* CHARFORMAT APPENDICES TO ANNEX XIII'))
