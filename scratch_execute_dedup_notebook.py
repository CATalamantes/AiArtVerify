import time

import nbformat as nbf
from nbclient import NotebookClient

path = "/Users/prabesharyal/ai4all/notebooks/04_dedup_group_split.ipynb"
nb = nbf.read(path, as_version=4)

client = NotebookClient(
    nb,
    timeout=1800,
    kernel_name="ai4all-venv",
    resources={"metadata": {"path": "/Users/prabesharyal/ai4all/notebooks"}},
)

t0 = time.time()
client.execute()
print(f"Executed in {time.time() - t0:.1f}s")

nbf.write(nb, path)
print("Saved executed notebook.")
