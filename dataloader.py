import json
from pyclbr import Function
import random
from collections import defaultdict
from typing import Dict, List

from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from few_shot import get_balanced_fewshot_samples, get_filtered_fewshot_samples, get_final_states
from fix_label import has_or_character
from heuristic_converter import get_converter, KluewosConverter

EXPERIMENT_DOMAINS = set(["관광", "숙소", "식당", "지하철", "택시"])
EXCLUDE_DOMAINS = set([])


class CustomDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data: list, args):
        """Reads source and target sequences from .txt files.
        Part that pre-processes the dataset originally.
        """
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair from the dataset i.e. source and target."""
        item_info = self.data[index]

        return item_info

    def __len__(self):
        """Length of the dataset, i.e. the total number of data samples."""
        return len(self.data)


def read_data(args, data_path: str, slots: list, tokenizer: AutoTokenizer, run_type: str) -> list:
    # generate domain-dependent slot list
    if args["only_domain"]:
        eval_slots = [slot for slot in slots if str.startswith(slot, args["only_domain"])]
    elif args["except_domain"]:
        eval_slots = [slot for slot in slots if not str.startswith(slot, args["except_domain"])]
    else:
        eval_slots = slots

    print(f"Reading all files from {data_path}")

    data = []
    # set converter for state-to-summary or for summary-to-state
    converter = get_converter(args["state_converter"])
    # set domain_counter (dict): {key: value}, the default value is 0 unless explicitly specified
    domain_counter = defaultdict(int)
    # read files
    with open(data_path) as f:
        dialogs = json.load(f)

        for single_dialog in dialogs:
            dialog_history = ""

            # skip if the domain is in `EXCLUDE_DOMAINS` as they are not in test set
            if single_dialog["domains"][0] in EXCLUDE_DOMAINS:
                continue

            # count domains
            for domain in single_dialog["domains"]:
                if domain in EXPERIMENT_DOMAINS:
                    # domain_counter (dict): {domain: the number of dialogs}
                    domain_counter[domain] += 1

            # dialog-level filtering
            if args["only_domain"]:
                if args["only_domain"] not in single_dialog["domains"]:
                    continue
            elif args["except_domain"]:

                # there are two options to filter dialog samples when pre-training model with a given `except_domain`
                # max: filter out every dialog that contains the `except_domain` context
                # min: filter out a dialog only if `except_domain` is the one and only domain that it has
                if args["dialog_filter"] == "max" and args["except_domain"] in single_dialog["domains"]:
                    continue
                elif args["dialog_filter"] == "min" and [args["except_domain"]] == single_dialog["domains"]:
                    continue

            # reading data
            for turn_id, turn in enumerate(single_dialog["turns"]):
                # accumulate dialog uttr
                dialog_history += (
                    " <system> " + turn["system"] + " </system> " + " <user> " + turn["user"] + " </user> "
                )
                # get slot-values as dict format
                slot_values = turn["state"]["slot_values"]
                slot_values = {k: v for k, v in dict.items(slot_values) if v != "none"}

                if run_type in ["train", "dev"] and has_or_character(slot_values):
                    continue

                if args["except_domain"] and any([str.startswith(k, args["except_domain"]) for k in slot_values]):
                    continue

                if args["model_name"] == "ke-t5-base":
                    # ke-t5-base is pre-trained without `eos_token`
                    input_text = f"summarize:{dialog_history.lower()}"
                else:
                    input_text = f"{tokenizer.bos_token} {dialog_history.lower()} {tokenizer.eos_token}"

                # get slots for evaluation per dialog
                eval_slots_per_dialog = set(
                    s for s in eval_slots if str.split(s, sep="-")[0] in single_dialog["domains"]
                )

                data_detail = {
                    "ID": single_dialog["dial_id"],
                    "domains": single_dialog["domains"],
                    "turn_id": turn_id,
                    "dialog_history": dialog_history,
                    "input_text": input_text,
                    "slot_values": slot_values,
                    "eval_slots": eval_slots_per_dialog,
                }
                if run_type in ["dev", "test"]:
                    output_text = converter.convert_state_to_summary(dialog_state=slot_values)
                    data_detail["output_text"] = output_text

                data.append(data_detail)

    print("Domain counter: ", domain_counter)

    return data


def get_slot_information(ontology: Dict[str, List[str]]) -> List:
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    # SLOTS = [key.replace(" ", "") for key in ontology_domains.keys()]
    SLOTS = [key.replace(" ", " ") for key in ontology_domains.keys()]

    return SLOTS


def collate_fn(tokenizer: AutoTokenizer, converter: KluewosConverter) -> Function:
    """Tie the sequences if the length of the sequences vary from indices in the dataset.

    Args:
        tokenizer (AutoTokenizer): a pre-trained tokenizer from HuggingFace.
        converter (KluewosConverter): a heuristic converter for state-to-summary or summary-to-state.

    Returns:
        function: a customized collate function for dataloader. 
    """

    def _collate(batch) -> dict:
        batch_data = {}
        for key in batch[0]:
            # batch_data (dict): {key: [data[key]]}
            # key (str): ID, domains, turn_id, dialog_history, input_text, slot_values, eval_slots
            # data[key] (list): data corresponding to the above key for each turn per single batch
            batch_data[key] = [single_batch[key] for single_batch in batch]

        input_batch = tokenizer(
            text=batch_data["input_text"],
            padding=True,  # whether to apply padding based on the longest sequence in the batch
            return_tensors="pt",
            add_special_tokens=True,
            verbose=False,  # whether to return additional output or warning, etc
            truncation=True,
            max_length=1000,
        )

        if "output_text" not in batch_data:
            batch_data["output_text"] = [converter.convert_state_to_summary(x) for x in batch_data["slot_values"]]

        batch_data["encoder_input"] = input_batch["input_ids"]
        batch_data["attention_mask"] = input_batch["attention_mask"]
        batch_data["decoder_output"] = tokenizer(
            batch_data["output_text"],
            padding=True,
            return_tensors="pt",  # non-padded return List[List[Int]]
            return_attention_mask=False,
            truncation=True,
            max_length=200,
        )["input_ids"]

        return batch_data

    return _collate


def prepare_data(args, tokenizer: AutoTokenizer) -> tuple((dict, dict)):
    data_paths = {run_type: f"./kluewos11/{run_type}.json" for run_type in ["train", "dev"]}
    ontology = json.load(open("./kluewos11/ontology.json", "r"))
    ALL_SLOTS = get_slot_information(ontology)

    datasets = {
        run_type: CustomDataset(
            read_data(args=args, data_path=data_path, slots=ALL_SLOTS, tokenizer=tokenizer, run_type=run_type), args,
        )
        for run_type, data_path in data_paths.items()
    }

    if 0.0 < args["fewshot"] < 1.0:
        num_train_dialogs = len(set(x["ID"] for x in datasets["train"]))
        # apply auto rounding off with int()
        num_fewshot_dialogs = int(num_train_dialogs * args["fewshot"])
        # get dialog state of the final turn
        final_states = get_final_states(datasets["train"])

        # get invalid value-filtered final states
        if args["filtered_sampling"]:
            final_states = get_filtered_fewshot_samples(final_states)

        # sample the dialogs balanced manner or randomly
        if args["balanced_sampling"]:
            sampled_ids = get_balanced_fewshot_samples(
                final_states=final_states, num_target_samples=num_fewshot_dialogs, all_slots=ALL_SLOTS
            )
        else:
            sampled_ids = random.sample(population=final_states.keys(), k=num_fewshot_dialogs)

        # update the dataset to sampled data
        datasets["train"].data = [x for x in datasets["train"].data if x["ID"] in sampled_ids]

        domain_counter = defaultdict(int)
        multi_domain_counter = defaultdict(int)
        for d in datasets["train"].data:
            domains = d["domains"]
            multi_domain_counter[tuple(sorted(d["domains"]))] += 1
            for domain in domains:
                if domain in EXPERIMENT_DOMAINS:
                    domain_counter[domain] += 1
        print("num_train_diags: ", num_train_dialogs, len(sampled_ids))
        print("domain_counter: ", domain_counter)
        print("multi_domain_counter: ", multi_domain_counter)

    print(
        f'dontcare occurence: {sum(["dontcare" in x["slot_values"].values() for x in datasets["train"].data]) / len(datasets["train"].data)}'
    )

    if args["debug_code"]:
        datasets["train"] = datasets["train"][:50]
        datasets["dev"] = datasets["dev"][:50]
        datasets["test"] = datasets["test"][:50]

    # get dataloader
    dataloaders = {
        run_type: DataLoader(
            dataset=dataset,
            batch_size=args[f"{run_type}_batch_size"],
            shuffle=True if run_type == "train" else False,
            collate_fn=collate_fn(tokenizer=tokenizer, converter=get_converter(args["state_converter"])),
        )
        for run_type, dataset in datasets.items()
    }
    domain_data = {}

    return dataloaders, domain_data
