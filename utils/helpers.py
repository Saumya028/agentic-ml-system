import json

def save_memory(data, path="memory/memory_store.json"):
    try:
        with open(path, "r") as f:
            memory = json.load(f)
    except:
        memory = []

    memory.append(data)

    with open(path, "w") as f:
        json.dump(memory, f, indent=4)