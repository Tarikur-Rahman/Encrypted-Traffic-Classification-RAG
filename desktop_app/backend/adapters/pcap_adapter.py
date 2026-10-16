from backend.adapters.input_adapter import validate

def inspect_capture(path, dataset):
    # Uses the registry's exact USTC extractor; CIC capture inference stays disabled.
    return validate(path, dataset)
