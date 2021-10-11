import json

class configData():
    def __init__(self, inpath):
        infile = open(inpath, 'r')
        injson = json.load(infile)
        infile.close()

        # member from input json
        for k, v in injson.items():
            setattr(self, k, v)

