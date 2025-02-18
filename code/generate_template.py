import json
import os
import random
from collections import defaultdict

import fire


def generate_template(
    template_id,
    topics_file,
    n_session_min,
    n_session_max,
    proportion_of_session_with_pivot_min,
    proportion_of_session_with_pivot_max,
    n_pivot_only_session_min,
    n_pivot_only_session_max,
    n_pivot_per_pivot_session_min,
    n_pivot_per_pivot_session_max,
    pivot_update_rate_min,
    pivot_update_rate_max,
    max_update_per_filler,
    filler_instruction_rate_min,
    filler_instruction_rate_max,
    filler_instruction_update_rate_min,
    filler_instruction_update_rate_max,
    output_dir,
):
    """Generates a template for a conversation.

    Args:
        template_id: Template id
        topics_file: File containing the list of topics for pivots and fillers
        n_session_min: Minimum number of sessions
        n_session_max: Maximum number of sessions
        proportion_of_session_with_pivot_min: Min proportion of sessions containing pivots (0-1)
        proportion_of_session_with_pivot_max: Max proportion of sessions containing pivots (0-1)
        n_pivot_only_session_min: Minimum number of sessions containing only pivots (i.e no fillers)
        n_pivot_only_session_max: Maximum number of sessions containing only pivots (i.e no fillers)
        n_pivot_per_pivot_session_min: Minimum number of pivots per session that contains pivots
        n_pivot_per_pivot_session_max: Maximum number of pivots per session that contains pivots
        pivot_update_rate_min: Minimum rate of pivot updates (0-1)
        pivot_update_rate_max: Maximum rate of pivot updates (0-1)
        max_update_per_filler: Maximum number of times a filler can be updated
        filler_instruction_rate_min: Minimum rate of filler instruction updates (0-1)
        filler_instruction_rate_max: Maximum rate of filler instruction updates (0-1)
        filler_instruction_update_rate_min: Minimum rate of filler instruction updates (0-1)
        filler_instruction_update_rate_max: Maximum rate of filler instruction updates (0-1)
        output_dir: Output directory to save the generated template
    """
    with open(topics_file, "r") as f:
        topics = json.load(f)

    SESSION_LENGTH = ["short", "medium", "long"]
    assert proportion_of_session_with_pivot_min > 0
    assert proportion_of_session_with_pivot_max <= 1

    ### Sessions
    n_session = random.randint(n_session_min, n_session_max)
    session_length = random.choices(SESSION_LENGTH, k=n_session)

    ### Pivots
    proportion_of_session_with_pivot = random.uniform(
        proportion_of_session_with_pivot_min, proportion_of_session_with_pivot_max
    )
    n_session_with_pivot = int(n_session * proportion_of_session_with_pivot)
    pivot_positions = random.sample(range(n_session), n_session_with_pivot)
    n_pivot_only_session = random.randint(n_pivot_only_session_min, n_pivot_only_session_max)
    pivot_positions_pivot_only_sessions = random.sample(pivot_positions, n_pivot_only_session)
    n_pivot_per_session = {
        p: random.randint(n_pivot_per_pivot_session_min, n_pivot_per_pivot_session_max) for p in pivot_positions
    }
    pivot_update_rate = random.uniform(pivot_update_rate_min, pivot_update_rate_max)
    # Sample pivots
    pivot_template, pivot_type, pivot_text, pivot_regex, cumulative_pivot_regex, pivot_eval, cumulative_pivot_eval = (
        sample_pivots(topics, n_session, pivot_positions, n_pivot_per_session, pivot_update_rate)
    )

    ### Fillers
    filler_instruction_rate = random.uniform(filler_instruction_rate_min, filler_instruction_rate_max)
    filler_instruction_update_rate = random.uniform(
        filler_instruction_update_rate_min, filler_instruction_update_rate_max
    )
    # Sample fillers
    filler_template, filler_type, filler_text = sample_fillers(
        topics,
        n_session,
        max_update_per_filler,
        n_pivot_only_session,
        pivot_positions_pivot_only_sessions,
        filler_instruction_rate,
        filler_instruction_update_rate,
    )

    ### Combine pivots and fillers
    sessions = []
    for session_idx in range(n_session):
        types, texts, regexs, cumulative_labels, evals, cumulative_evals = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # fillers
        if filler_template[session_idx] != -1:
            types.append(filler_type[session_idx])
            texts.append(filler_text[session_idx])
        # pivots
        if session_idx in pivot_positions:
            types += pivot_type[session_idx]
            texts += pivot_text[session_idx]
        cumulative_labels += cumulative_pivot_regex[session_idx]
        regexs += pivot_regex[session_idx]
        evals += pivot_eval[session_idx]
        cumulative_evals += cumulative_pivot_eval[session_idx]

        session = {
            "type": types,
            "topic": texts,
            "session_regex": regexs,
            "cumulative_regex": cumulative_labels,
            "session_eval_query": evals,
            "cumulative_eval_query": cumulative_evals,
            "session_length": session_length[session_idx],
        }
        sessions.append(session)

    ### Contexts and personas
    context = random.choice(topics["contexts"])
    context["mentor_persona"] = random.choice(topics["personas"]["mentor"])
    context["mentee_persona"] = random.choice(topics["personas"]["mentee"])

    ### Final template
    template = {
        "context": context,
        "pivots": pivot_template,
        "fillers": filler_template,
        "sessions": sessions,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"conversation_{template_id}.json")
    with open(output_file, "w") as f:
        json.dump(template, f, indent=1)


def sample_pivots(topics, n_session, pivot_positions, n_pivot_per_session, pivot_update_rate):
    """Sample pivots for a conversation."""
    pivot_topics = {pivot["id"]: pivot for pivot in topics["pivots"]}
    pivot_topics_by_type = defaultdict(list)
    for pivot_id, pivot in pivot_topics.items():
        pivot_type = pivot["regex"][0][0]
        pivot_topics_by_type[pivot_type].append(pivot_id)
    pivot_types = list(pivot_topics_by_type.keys())
    # pivots that can be updated
    pivot_ids_with_updates = list(range(6, 16))
    # Combine text and regex
    for pivot_id in pivot_topics.keys():
        curr_pivot = pivot_topics[pivot_id]
        curr_pivot["text_regex"] = [
            (text_id, text, regex) for text_id, (text, regex) in enumerate(zip(curr_pivot["text"], curr_pivot["regex"]))
        ]

    pivot_template = []
    pivot_type = []
    pivot_text = []
    pivot_regex = []
    cumulative_pivot_regex = []
    latest_cumulative_pivot_regex = {}
    pivot_eval = []
    cumulative_pivot_eval = []

    current_cumulative_eval = set()
    # previously introduced pivots that now can be updated
    updatable_pivot_ids = set()

    for session_idx in range(n_session):
        if session_idx in pivot_positions:
            n_pivot = n_pivot_per_session[session_idx]
            # Prevent update of the same pivot in the same session
            seen_pivots_session = set()
            pivot_update_ids, texts, regexs, session_eval, type = [], [], {}, [], []
            for _ in range(n_pivot):
                # Sample object type first before sampling pivots
                selected_pivot_ids_per_type = set()
                for t in pivot_types:
                    pivot_ids = [p for p in pivot_topics_by_type[t] if p in pivot_topics.keys()]
                    if len(pivot_ids) > 0:
                        selected_pivot_id = random.choice(pivot_ids)
                        selected_pivot_ids_per_type.add(selected_pivot_id)

                # Insertion or update
                insertion_pivot_ids = list(selected_pivot_ids_per_type - updatable_pivot_ids - seen_pivots_session)
                update_pivot_ids = list(updatable_pivot_ids - seen_pivots_session)
                if random.random() < pivot_update_rate or len(insertion_pivot_ids) == 0:
                    if len(update_pivot_ids) > 0:
                        pivot_id = random.choice(update_pivot_ids)
                        type.append("pivot-update")
                    else:
                        # Introduce a pivot that can be updated if not introduced yet
                        multi_update_insertion_pivot_ids = [
                            p
                            for p in pivot_ids_with_updates
                            if p in pivot_topics.keys() and p not in updatable_pivot_ids
                        ]
                        pivot_id = random.choice(multi_update_insertion_pivot_ids)
                        updatable_pivot_ids.add(pivot_id)
                        type.append("pivot-add")
                else:
                    pivot_id = random.choice(insertion_pivot_ids)
                    if pivot_id in pivot_ids_with_updates:
                        updatable_pivot_ids.add(pivot_id)
                    type.append("pivot-add")
                seen_pivots_session.add(pivot_id)

                # shuffle updates or not
                text_regex = pivot_topics[pivot_id]["text_regex"]
                if pivot_topics[pivot_id]["shuffle_updates"]:
                    selected_update = random.randint(0, len(text_regex) - 1)
                else:
                    selected_update = 0
                update_id, text, regex = text_regex.pop(selected_update)
                pivot_update_ids.append([pivot_id, update_id])
                texts.append(text)
                regexs[pivot_id] = regex
                session_eval.append(pivot_topics[pivot_id]["eval_query"])

                # Remove pivots that cannot be updated anymore and add eval
                if len(pivot_topics[pivot_id]["text_regex"]) == 0:
                    pivot_topics.pop(pivot_id)
                    updatable_pivot_ids.discard(pivot_id)

            pivot_template.append(pivot_update_ids)
            pivot_text.append(texts)
            pivot_regex.append(list(regexs.values()))
            latest_cumulative_pivot_regex.update(regexs)
            cumulative_pivot_regex.append(list(latest_cumulative_pivot_regex.values()))
            pivot_type.append(type)

            pivot_eval.append(session_eval)
            current_cumulative_eval.update(session_eval)
            cumulative_pivot_eval.append(list(current_cumulative_eval))

        else:
            pivot_template.append([-1])
            pivot_text.append([])
            pivot_regex.append([])
            pivot_type.append([])
            cumulative_pivot_regex.append(list(latest_cumulative_pivot_regex.values()))
            cumulative_pivot_eval.append(list(current_cumulative_eval))
            pivot_eval.append([])

    return (
        pivot_template,
        pivot_type,
        pivot_text,
        pivot_regex,
        cumulative_pivot_regex,
        pivot_eval,
        cumulative_pivot_eval,
    )


def sample_fillers(
    topics,
    n_session,
    max_update_per_filler,
    n_pivot_only_session,
    pivot_positions_pivot_only_sessions,
    filler_instruction_rate,
    filler_instruction_update_rate,
):
    """Sample fillers for a conversation."""
    fillers = random.choices(topics["fillers"], k=n_session - n_pivot_only_session)
    filler_n_update = {filler["id"]: max_update_per_filler for filler in fillers}
    fillers = {filler["id"]: filler for filler in fillers}
    filler_template = []
    filler_type = []
    filler_text = []
    seen_filler_ids = set()
    fillers_instruction = {fi["id"]: fi for fi in topics["fillers_instruction"]}
    updatable_filler_instruction_ids = set()
    for session_idx in range(n_session):
        seen_fillers_instruction_session = set()
        if session_idx in pivot_positions_pivot_only_sessions:
            filler_template.append(-1)
            filler_text.append(None)
            filler_type.append(None)
        else:
            if random.random() < filler_instruction_rate:
                filler_id = random.choice(list(filler_n_update.keys()))
                filler_template.append(filler_id)
                filler_text.append(fillers[filler_id]["text"])
                filler_n_update[filler_id] -= 1
                if filler_n_update[filler_id] == 0:
                    filler_n_update.pop(filler_id)
                # Get filler type: add or update
                if filler_id in seen_filler_ids:
                    filler_type.append("filler-update")
                else:
                    filler_type.append("filler-add")
                    seen_filler_ids.add(filler_id)
            else:
                insertion_filler_instruction_ids = list(fillers_instruction.keys() - seen_fillers_instruction_session)
                update_filler_instruction_ids = list(
                    updatable_filler_instruction_ids - seen_fillers_instruction_session
                )
                if len(update_filler_instruction_ids) > 0 and random.random() < filler_instruction_update_rate:
                    filler_instruction_id = random.choice(update_filler_instruction_ids)
                    filler_type.append("filler-instruction-update")
                else:
                    filler_instruction_id = random.choice(insertion_filler_instruction_ids)
                    updatable_filler_instruction_ids.add(filler_instruction_id)
                    filler_type.append("filler-instruction-add")
                seen_fillers_instruction_session.add(filler_instruction_id)
                filler_template.append(filler_instruction_id)
                filler_text.append(fillers_instruction[filler_instruction_id]["text"].pop())

                # Remove fillers instruction that cannot be updated anymore and add eval
                if len(fillers_instruction[filler_instruction_id]["text"]) == 0:
                    fillers_instruction.pop(filler_instruction_id)
                    updatable_filler_instruction_ids.discard(filler_instruction_id)
    return filler_template, filler_type, filler_text


if __name__ == "__main__":
    fire.Fire(generate_template)
