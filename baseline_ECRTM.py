# ============================================================
# CELL 7: In-memory GPU dataset wrapper for TopMost's BasicTrainer
# L4 has ~22.5GB usable memory; train BoW as float32 is ~6.76GB,
# so keeping the whole tensor resident on GPU is safe here.
# ============================================================
import topmost
from torch.utils.data import DataLoader

BATCH_SIZE = 200  # can raise (e.g. 512) on L4 if training is stable

class InMemoryBoWDataset:
    def __init__(self, bow_train, bow_test, vocab, device="cuda", batch_size=200):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        train_dense = bow_train.toarray().astype("float32")
        test_dense = bow_test.toarray().astype("float32")

        self.train_data = torch.from_numpy(train_dense).to(device)
        self.test_data = torch.from_numpy(test_dense).to(device)

        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

print(f"Building in-memory GPU dataset "
      f"({bow_train.shape[0]*bow_train.shape[1]*4/1e9:.2f} GB train tensor)...")
tm_dataset = InMemoryBoWDataset(bow_train, bow_test, vocab, device=device, batch_size=BATCH_SIZE)
print(f"train_data on: {tm_dataset.train_data.device}, shape={tm_dataset.train_data.shape}")
print("Dataset ready.")
