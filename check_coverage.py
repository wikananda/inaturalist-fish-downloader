from pathlib import Path

def load_file(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

species_all = load_file(Path("species.txt"))
train = load_file(Path("train.txt"))
val = load_file(Path("val.txt"))
test = load_file(Path("test.txt"))

covered = train | val | test
missing = species_all - covered

print(f"Total species in species.txt: {len(species_all)}")
print(f"Total covered in splits: {len(covered)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nMissing species:")
    for s in sorted(missing):
        print(s)
else:
    print("\nAll species are covered!")
