from typing import TypedDict


class Target(TypedDict):
    """Target data type for the `FlowersDataset`.

    This class represents the target data type used in the `FlowersDataset` class.
    It is used to store the target data for each sample in the dataset.

    Keys:
        class_name:
            Name of the class.
        label:
            Label of the class.
        id:
            ID of the sample.
    """

    class_name: str
    label: int
    id: int
