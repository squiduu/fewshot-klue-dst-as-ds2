import argparse
import os
import json
from tqdm import tqdm


def convert_data_format(input_dir: str, run_type: str) -> list:
    """Convert the KLUEWOS 1.1 format to MultiWOZ 2.1 format for DS2 and return it.
    Reference paper:
        https://arxiv.org/pdf/2203.01552.pdf (https://github.com/jshin49/ds2)

    Args:
        input_dir (str): a directory of the KLUEWOS 1.1 dataset.

    Returns:
        dataset (list): a modified KLUEWOS 1.1 dataset.
    """
    with open(file=input_dir, mode="r", encoding="utf-8") as f:
        rawdata = json.load(f)

    dataset = []
    for i in tqdm(range(len(rawdata)), desc=f"Convert KLUEWOS 1.1 {run_type} data format", total=len(rawdata)):
        data = {}
        turns = []
        dial_id = rawdata[i]["guid"]
        domains = rawdata[i]["domains"]
        dialogs = rawdata[i]["dialogue"]
        for j in range(len(dialogs)):
            state = {}
            turn = {}
            if j == 0:
                turn["system"] = "none"
                turn["user"] = str.strip(dialogs[j]["text"])
                turn_state = dialogs[j]["state"]
                slot_values = {}
                if turn_state != []:
                    for ts in turn_state:
                        ts_slot = str.split(ts, sep="-")[:-1]
                        ts_slot = ts_slot[0] + "-" + ts_slot[1]
                        ts_value = str.split(ts, sep="-")[-1]
                        slot_values[ts_slot] = ts_value
                        state["active_intent"] = "none"
                        state["slot_values"] = slot_values
                else:
                    state = {"active_intent": "none", "slot_values": {}}
                turn["state"] = state

            elif j > 0 and j < len(dialogs) / 2:
                turn["system"] = str.strip(dialogs[2 * j - 1]["text"])
                turn["user"] = str.strip(dialogs[2 * j]["text"])
                turn_state = dialogs[2 * j]["state"]
                slot_values = {}
                if turn_state != []:
                    for ts in turn_state:
                        ts_slot = str.split(ts, sep="-")[:-1]
                        ts_slot = ts_slot[0] + "-" + ts_slot[1]
                        ts_value = str.split(ts, sep="-")[-1]
                        slot_values[ts_slot] = ts_value
                        state["active_intent"] = "none"
                        state["slot_values"] = slot_values
                else:
                    state = {"active_intent": "none", "slot_values": {}}
                turn["state"] = state

            if turn != {}:
                turns.append(turn)

            data["dial_id"] = dial_id
            data["domains"] = domains
            data["turns"] = turns

        dataset.append(data)

    return dataset


def main(args):
    for run_type in ["train", "dev"]:
        data_dir = os.path.join(args.input_dir, f"wos-v1.1_{run_type}.json")
        modified_data = convert_data_format(input_dir=data_dir, run_type=run_type)

        with open(file=os.path.join(args.output_dir, f"{run_type}.json"), mode="w", encoding="utf-8") as f:
            json.dump(obj=modified_data, fp=f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./kluewos11/")
    parser.add_argument("--output_dir", type=str, default="./kluewos11/")
    args = parser.parse_args()

    main(args)
