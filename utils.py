def pad_outputs(beam_outputs, num_beams=10):
    """ Pad beam outputs to num_beams. """

    if len(beam_outputs) < num_beams:
        beam_outputs += [''] * (num_beams - len(beam_outputs))
    return beam_outputs


def get_rank(beam_outputs, targets, num_beams=10):
    """ Get rank of the target word(s) in the beam outputs. """

    for i, beam_output in enumerate(beam_outputs):
        if any([x in beam_output for x in targets]):
            return i + 1
    return num_beams


def get_acc(beam_outputs, targets):
    """ Return 1 if any target word are in the beam outputs. """

    for i, beam_output in enumerate(beam_outputs):
        if any([x in beam_output for x in targets]):
            return 1
    return 0


def mean_reciprocal_rank(rank_list):
    """ Compute mean reciprocal rank of the rank list. """

    return sum([1 / x for x in rank_list]) / len(rank_list)


def calc_metrics(outputs, targets, num_beams=10):
    """ Calculate metrics. """

    rank_list, acc_list = [], []
    for beam_outputs, target in zip(outputs, targets):
        rank_list.append(get_rank(beam_outputs, target, num_beams=num_beams))
        acc_list.append(get_acc(beam_outputs, target))

    mrr = mean_reciprocal_rank(rank_list)
    acc = sum(acc_list) / len(acc_list)
    
    return {'mrr': round(mrr*100, 3), 'acc': round(acc*100, 3)}