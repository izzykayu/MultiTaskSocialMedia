from rxspace import ClassificationTsvReader
reader = ClassificationTsvReader()
dataset = list(reader.read("./data/Task-DataForParticipants/train.tsv"))

print("type of its first element: ", type(dataset[0]))
print("size of dataset: ", len(dataset))