def lower(string):
    return string.lower()


def strip_nextline_return(string):
    string = string.replace('\n', ' ').replace('\r', ' ').strip()
    return string
