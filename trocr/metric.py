
# def prf(pred, true, processor):
#     precision = 0.0
#     recall = 0.0
#     f1 = 0.0
#     for pred_ids, label_ids in zip(pred, true):
#         p, r, f = prf_metric(pred_ids, label_ids, processor)
#         precision += p
#         recall += r
#         f1 += f
#
#     return precision / len(pred) *100, recall / len(pred) *100, f1 / len(pred) *100


def prf_metric(pred_ids, label_ids, processor):
    # https://github.com/microsoft/unilm/blob/master/trocr/scoring.py
    pred_str, label_str = \
        get_decode_str(pred_ids, label_ids, processor)

    n_gt_words, n_detected_words, n_match_words = calculate_match_words(pred_str, label_str)

    return scoring(n_gt_words, n_detected_words, n_match_words)


def scoring(n_gt_words, n_detected_words, n_match_words):
    precision = n_match_words / float(n_detected_words)
    recall = n_match_words / float(n_gt_words)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def calculate_match_words(pred_str, label_str):
    n_gt_words = 0
    n_detected_words = 0
    n_match_words = 0

    pred_words = list(pred_str.split())
    ref_words = list(label_str.split())
    n_gt_words += len(ref_words)
    n_detected_words += len(pred_words)
    for pred_w in pred_words:
        if pred_w in ref_words:
            n_match_words += 1
            ref_words.remove(pred_w)

    return n_gt_words, n_detected_words, n_match_words


def get_decode_str(pred_ids, label_ids, processor):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    return pred_str, label_str