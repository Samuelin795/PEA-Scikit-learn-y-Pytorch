import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets

# Definir campos para los datos
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

# Cargar dataset de ejemplo (preetiquetado)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Construir el vocabulario
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# Crear iteradores de datos
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), batch_size=64, device=torch.device('cpu')
)

# Definir una red neuronal simple
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Instanciar y entrenar el modelo
model = SentimentRNN(len(TEXT.vocab), 100, 256, 1)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Código de entrenamiento y evaluación...
