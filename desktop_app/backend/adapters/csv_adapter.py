from backend.adapters.input_adapter import validate

def inspect_csv(path, dataset):
    return validate(path, dataset)
