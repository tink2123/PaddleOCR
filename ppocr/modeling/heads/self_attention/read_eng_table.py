def get_map():
    file = 'table_eng_95_val'
    f = open(file)
    map = {}
    for line in f.readlines():
        ret = line.split()
        try:
            map[ret[1]]=ret[2]
        except:
            print(ret)
    return map
