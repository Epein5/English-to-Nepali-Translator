import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import random
import pandas as pd

# Hyperparameters
INPUT_SIZE = 10000  # Size of the English vocabulary
OUTPUT_SIZE = 10000  # Size of the Nepali vocabulary
EMBED_SIZE = 256
HIDDEN_SIZE = 512
N_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
import pandas as pd
from torch.utils.data import Dataset
from sentencepiece import SentencePieceProcessor

class TranslationDataset(Dataset):
    def __init__(self, cleaned_file_path):
        self.pairs = self.load_data(cleaned_file_path)

    def load_data(self, cleaned_file_path):
        df = pd.read_excel(cleaned_file_path)
        df = df.dropna()
        english_sentences = df['english_sent'].tolist()
        nepali_sentences = df['nepali_sent'].tolist()

        english_tokenizer = SentencePieceProcessor(model_file='english_sp.model')
        nepali_tokenizer = SentencePieceProcessor(model_file='nepali_sp.model')

        pairs = []
        for english_sentence, nepali_sentence in zip(english_sentences, nepali_sentences):
            english_indices = self.process_sentence(english_sentence, english_tokenizer)
            nepali_indices = self.process_sentence(nepali_sentence, nepali_tokenizer)
            pairs.append((english_indices, nepali_indices))
        
        return pairs

    def process_sentence(self, sentence, tokenizer):
        tokens = tokenizer.encode(sentence, out_type=int)
        return [1] + tokens + [2]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
# Padding function
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(src) for src in src_batch]
    trg_lens = [len(trg) for trg in trg_batch]
    src_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in src_batch], padding_value=0)
    trg_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in trg_batch], padding_value=0)
    return src_padded, trg_padded, src_lens, trg_lens

# Seq2Seq components (Encoder, Decoder, Seq2Seq classes)
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    # def forward(self, src, hidden=None):
    #     embedded = self.embed(src)
    #     outputs, hidden = self.gru(embedded, hidden)
    #     outputs = outputs[:, :, :self.gru.hidden_size // 2] + outputs[:, :, self.gru.hidden_size // 2:]
    #     return outputs, hidden
    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # Correctly sum bidirectional outputs
        outputs = (outputs[:, :, :self.gru.hidden_size] + 
                outputs[:, :, self.gru.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return torch.softmax(energy.squeeze(1), dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embed(input).unsqueeze(0)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = self.out(torch.cat((output.squeeze(0), context.squeeze(0)), 1))
        return torch.log_softmax(output, dim=1), hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.size(1)
        trg_len = trg.size(0)
        vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(DEVICE)
        encoder_outputs, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.gru.num_layers]
        output = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            output = trg[t] if teacher_force else output.argmax(1)
        return outputs

# Initialize model
encoder = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
def train_model(dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for src, trg, _, _ in dataloader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, trg, TEACHER_FORCING_RATIO)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

# Example dataset (replace with real data)
# dummy_pairs = [
#     ([1, 2, 3, 4], [5, 6, 7, 8]),
#     ([9, 10, 11], [12, 13, 14]),
# ]
dataset = TranslationDataset("Dataset/english-nepali-cleaned.xlsx")
train_model(dataset)
