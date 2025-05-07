from auto_exp import *
import argparse

def parse_config(config):
    # Parse the arguments and store them in a dictionary
    parsed_arguments = {}
    for arg in config:
        key_value = arg.split('=')
        if len(key_value) != 2:
            raise ValueError(f"Invalid argument: {arg}. Must be of the form 'key=value'")
        key, value = key_value
        parsed_arguments[key] = int(value)

    return alg_config(**parsed_arguments)

def parse_exp(exp : str) -> Union[None, exp_config]:
    try:
        elems = exp.split()
        if len(elems) != 6:
            # print(f"Invalid line: {exp}. Must be of the form 'alg_goal exp_type alg_name filename source iterations'")
            return None
        exp_algs = {'alg_goal' : elems[0]}
        if elems[1] != 'time': return None
        exp_algs['exp_type'] = elems[1]
        exp_algs['alg_name'] = elems[2]
        exp_algs['filename'] = elems[3]
        exp_algs['source'] = int(elems[4])
        exp_algs['reps'] = int(elems[5])

        return exp_config(**exp_algs)
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', help='The name of the experiment')
    parser.add_argument('query', help='The filename of the query, or the explicit query')
    parser.add_argument('config', nargs='*', help='Configuration parameters of the form "arg1=val1 arg2=val2 arg3=val3 ..."')
    parser.add_argument('-t', '--timeout', type=float, help='Set the timeout in seconds for any execution.')
    args = parser.parse_args()

    # Get the filename
    query = args.query

    conf_alg = parse_config(args.config)

    # Read the query file and split it into lines. Ignore the
    # lines where the first non-space character is '#', and
    # ignore the empty lines

    conf_algs = []
    conf_exps = []

    parsed_exp = parse_exp(query)

    if parsed_exp is not None:
        conf_algs.append(conf_alg)
        conf_exps.append(parsed_exp)

    else:
        # Check whether the query file exists
        if not os.path.exists(query):
            raise ValueError(f"Query file {query} does not exist and it cannot be interpreted as query string")

        with open(query, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != '' and line.strip()[0] != '#']

        for line in lines:
            parsed_exp = parse_exp(line)
            if parsed_exp is None: continue
            conf_algs.append(conf_alg)
            conf_exps.append(parsed_exp)


    set_of_exps(args.expname, conf_algs, conf_exps, timeout=args.timeout)

        

if __name__ == "__main__":
    main()
