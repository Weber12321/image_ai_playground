import logging


def prf_metric(pred_ids, label_ids, processor):
    # https://github.com/microsoft/unilm/blob/master/trocr/scoring.py
    pred_list, label_list = \
        get_decode_str(pred_ids, label_ids, processor)
    

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for pred, label in zip(pred_list, label_list):
        n_gt_words, n_detected_words, n_match_words = calculate_match_words(
            pred, label
        )
        p, r, f = scoring(n_gt_words, n_detected_words, n_match_words)
        precision += p
        recall += r
        f1 += f
    length = len(pred_list)
    return precision / length, recall / length, f1 / length


def scoring(n_gt_words, n_detected_words, n_match_words):
    if n_detected_words == 0:
        precision = 0
    else:        
        precision = n_match_words / n_detected_words
    if n_gt_words == 0:
        recall = 0
    else:
        recall = n_match_words / n_gt_words

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def calculate_match_words(pred_str, label_str):
    n_match_words = 0

    pred_words = list(pred_str.split())
    ref_words = list(label_str.split())
    n_gt_words = len(ref_words)
    n_detected_words = len(pred_words)

    for pred_w in pred_words:
        if pred_w in ref_words:
            n_match_words += 1
            ref_words.remove(pred_w)

    return n_gt_words, n_detected_words, n_match_words


def get_decode_str(pred_ids, label_ids, processor):
    pred_list = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_list = processor.batch_decode(label_ids, skip_special_tokens=True)

    return pred_list, label_list