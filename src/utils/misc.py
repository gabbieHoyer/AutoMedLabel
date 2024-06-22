
# ------------------ Misc utils ------------------
def remap_labels(labels_dict:dict, target_label_id=[]):
    # Ensure target_label_id is a list for uniform processing
    if target_label_id is None:
        target_label_id = []
    elif isinstance(target_label_id, int):
        target_label_id = [target_label_id]

    # Create a new labels dictionary excluding the target_label_id(s)
    filtered_labels_dict = {key: val for key, val in labels_dict.items() if key not in target_label_id}
    return filtered_labels_dict