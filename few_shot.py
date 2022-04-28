import random
from collections import defaultdict

from torch.utils.data import Dataset


def get_balanced_fewshot_samples(final_states: dict, num_target_samples: int, all_slots: list) -> list:
    """Sample the slots and domains in a balanced manner, and return the sampled `dial_id`.

    Args:
        final_states (dict): {`dial_id`: `dialog_state` of the final turn for each `dial_id`}
        num_target_samples (int): the number of dialogs to sample. 
        all_slots (list): a list of all slots.
    """
    # initialize sampled slots count and sampled ids
    sampled_slots_count = {_slot_name: 0 for _slot_name in all_slots}
    sampled_ids = []

    # domains (dict): {dial_id: sorted domains as tuple(str)}
    domains = {
        dial_id: tuple(sorted(set(str.split(_slot, sep="-")[0] for _slot in dialog_state)))
        for dial_id, dialog_state in final_states.items()
    }
    # initialize sampled domain count
    sampled_domains_count = {_domain: 0 for _domain in set(domains.values())}
    # choose only a single domain `dial_id` randomly
    sampled = random.choice([_dial_id for _dial_id, _domain in domains.items() if len(_domain) == 1])
    sampled_ids.append(sampled)
    # delete sampled dialog for balancing
    del domains[sampled], final_states[sampled]

    while len(sampled_ids) < num_target_samples:
        # get the least sampled slots for balancing
        target_slot = get_argmin(sampled_slots_count)
        candidates = [_dial_id for _dial_id, _dialog_state in final_states.items() if target_slot in _dialog_state]

        if len(candidates) == 0:
            del sampled_slots_count[target_slot]
            continue

        # choose only a single domain randomly from the least sampled domains
        sampled = random.choice(candidates)
        sampled_ids.append(sampled)

        # update sample counters
        for _slot in final_states[sampled]:
            if _slot in sampled_slots_count:
                sampled_slots_count[_slot] += 1
        sampled_domains_count[domains[sampled]] += 1
        # delete sampled dialog for balancing
        del final_states[sampled], domains[sampled]

        if len(sampled_ids) >= num_target_samples:
            break

        # get the least sampled domains for balancing
        target_domain = get_argmin(sampled_domains_count)
        candidates = [_dial_id for _dial_id, _domain in domains.items() if _domain == target_domain]
        if len(candidates) == 0:
            del sampled_domains_count[target_domain]
            continue
        sampled = random.choice(candidates)
        sampled_ids.append(sampled)

        # update sample counters
        for _slot in final_states[sampled]:
            if _slot in sampled_slots_count:
                sampled_slots_count[_slot] += 1
        sampled_domains_count[domains[sampled]] += 1

        # delete sampled dialog for balancing
        del domains[sampled], final_states[sampled]

    print(sampled_slots_count.values())
    print(sampled_domains_count.values())

    return sampled_ids


def get_filtered_fewshot_samples(final_states: dict) -> dict:
    """
    Args:
        final_states (dict): {`dial_id`: `dialog_state` of the final turn for each `dial_id`}
    """
    return {dial_id: dialog_state for dial_id, dialog_state in final_states.items()}


def get_final_states(dataset: Dataset) -> dict:
    """Return the dialog state of the final turn for each `dial_id`.

    Args:
        dataset (DSTDataset): a customized dataset.
    """
    final_turn_ids = defaultdict(int)
    final_turn_states = {}
    for x in dataset:
        dial_id, turn_id = x["ID"], x["turn_id"]
        if final_turn_ids[dial_id] < turn_id:
            final_turn_ids[dial_id] = turn_id
            final_turn_states[dial_id] = x["slot_values"]

    return final_turn_states


def get_argmin(sample_counter: dict):
    """Return the least sampled domains.

    Args:
        sample_counter (dict): a dict representing  the number of samples per domain in a dataset.
    """
    min_value = min(sample_counter.values())
    for k, v in sample_counter.items():
        if v == min_value:

            return k
