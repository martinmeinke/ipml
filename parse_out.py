import sys

def parseparams(line):
    return [float(p.split("=")[1].strip()) for p in line.split() if "=" in p]

def parsescore(line):
    return 1.0 - float(line.split("=")[1].strip())

def parsefile(fname):
    content = None
    with open(fname, 'r') as f:
        content = f.readlines()
    if not content:
        print "No data in %s" % fname
        return []
    alldata = []
    curdata = None
    for l in content:
        if l.startswith("Run"):
            if curdata:
                if 'validation' in curdata and 'test' in curdata and 'params' in curdata:
                    curdata["score"] = (curdata["validation"] + curdata["test"])*0.5
                    alldata.append(curdata)
                else:
                    print "Skipping incomplete data: ", curdata
            curdata = dict(params=parseparams(l))
        elif l.startswith("Train"):
            curdata["training"] = parsescore(l)
        elif l.startswith("Valid"):
            curdata["validation"] = parsescore(l)
        elif l.startswith("Test"):
            curdata["test"] = parsescore(l)
    return alldata

def main():
    if len(sys.argv) < 2:
        sys.exit("Provide filenames")

    data = []
    for f in sys.argv[1:]:
        data += parsefile(f)

    print data

    print "Best data: "
    print max(data, key=lambda x: x["score"])

if __name__ == "__main__":
    main()
