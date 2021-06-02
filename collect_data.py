import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--directory", dest="directory",
                    help="Directory of log files", metavar="DIR")

args = parser.parse_args()

directory = args.directory

if args.directory is None:
    raise FileNotFoundError("Please use the -d flag to specify a directory with the linear probing logs.")

all_results = {}

for filename in os.listdir(directory):
    sline = None
    resultline = None
    searchfile = open(directory + filename, "r")
    for line in searchfile:
        if "Namespace" in line:
            sline = line
        if line.startswith("["):
            resultline = line

    s = sline[11:-2]
    sdict = {}

    for item in s.split(','):
        try:
            k, v = item.split("=")
            sdict[k.strip()] = eval(v.strip())
        except (ValueError, SyntaxError, NameError):
            pass

    result = max(eval(resultline))

    model_name = os.path.dirname(sdict.get("pretrained"))
    checkpoint = sdict.get("pretrained").split("/")[-1].split(".")[-3].replace("checkpoint_", "")
    epoch = eval(checkpoint.split("_")[0].lstrip("0") or "0")
    try:
        batch = eval(checkpoint.split("_")[1].lstrip("0") or "0")
        checkpoint = (epoch, batch)
    except:
        checkpoint = epoch

    if model_name not in all_results:
        all_results[model_name] = [(result, checkpoint)]
    else:
        all_results.get(model_name).append((result, checkpoint))

for experiment, accuracy in all_results.items():
    print(experiment, sorted(accuracy, key=lambda x: x[1]))
