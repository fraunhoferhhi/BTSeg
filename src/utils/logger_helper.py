from typing import Dict, List


def add_member_variables_to_hparams_dict(
    hparams_dict: Dict,
    dict_params: Dict,
    key_prefix: str = None,
    dict_additional_params: Dict = None,
    list_params_to_ignore: List = [],
) -> None:
    # remove params to ignore
    for param in list_params_to_ignore:
        dict_params.pop(param, None)

    # add the additional params from dict_additional_params
    if dict_additional_params is not None:
        dict_params.update(dict_additional_params)

    if key_prefix is not None:
        dict_params = {f"{key_prefix}{key}": val for key, val in dict_params.items()}

    hparams_dict.update(dict_params)
