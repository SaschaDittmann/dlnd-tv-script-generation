import helper
import numpy as np
import problem_unittests as tests
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummaryX import summary
import argparse

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()

print("Torch version:", torch.__version__)

#################################################
## Get Command-Line Arguments
#################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/Seinfeld_Scripts.txt',
                    help='data directory')
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='output directory')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,
                    help='number of words in a sequence')
parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help='learning rate')
parser.add_argument('--sequence_length', type=int, default=10,
                    help='number of words in a sequence')
parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=400,
                    help='hidden dimension')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of RNN layers')
args = parser.parse_args()

#################################################
## Get the Data
#################################################
data_dir = args.data_dir
text = helper.load_data(data_dir)

#################################################
## Explore the Data
#################################################
view_line_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

#################################################
## Implement Pre-processing Functions
## - Lookup Table
## - Tokenize Punctuation
#################################################
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # return tuple
    return (vocab_to_int, int_to_vocab)
tests.test_create_lookup_tables(create_lookup_tables)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    tokens = dict()
    tokens['.'] = '<PERIOD>'
    tokens[','] = '<COMMA>'
    tokens['"'] = '<QUOTATION_MARK>'
    tokens[';'] = '<SEMICOLON>'
    tokens['!'] = '<EXCLAMATION_MARK>'
    tokens['?'] = '<QUESTION_MARK>'
    tokens['('] = '<LEFT_PAREN>'
    tokens[')'] = '<RIGHT_PAREN>'
    tokens['?'] = '<QUESTION_MARK>'
    tokens['-'] = '<HYPHEN>'
    tokens['\n'] = '<NEW_LINE>'
    #tokens[':'] = '<COLON>'
    return tokens
tests.test_tokenize(token_lookup)

#################################################
## Pre-process all the data and save it
#################################################

# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

#################################################
## Check Point
#################################################
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#################################################
## Build the Neural Network
#################################################

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
if train_on_gpu:
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size]
       
    x, y = [], []
    for idx in range(0, len(words) - sequence_length):
        x.append(words[idx:idx+sequence_length])
        y.append(words[idx+sequence_length])
    feature_tensors, target_tensors = torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
    data = TensorDataset(feature_tensors, target_tensors)
    data_loader = DataLoader(data, batch_size = batch_size, shuffle = False)
    # return a dataloader
    return data_loader

# test dataloader
test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        self.dropout = dropout
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        # batch_size equals the input's first dimension
        batch_size = nn_input.size(0)
        nn_input = nn_input.long()
        
        # Embedded_output and LTSM 
        embedded_output = self.embedding(nn_input)
        r_output, hidden = self.lstm(embedded_output, hidden)
        
        # Adding another LSTM output layer
        r_output = r_output.contiguous().view(-1, self.hidden_dim)
        
        # Dropout and fully-connected layers
        r_output = self.dropout(r_output)
        r_output = self.fc(r_output)
        
        # reshape into (batch_size, seq_length, output_size)
        r_output = r_output.view(batch_size, -1, self.output_size)
        
        # Final_output equals the last batch
        final_output = r_output[:, -1]

        # return one batch of output word scores and the hidden state
        return final_output, hidden
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        # initialize hidden state with zero weights, and move to GPU if available
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

tests.test_rnn(RNN, train_on_gpu)

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    if (train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
    # move data to GPU, if available
    # perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])
    # clear accumulated gradients
    rnn.zero_grad()
    # obtain the rnn's output
    output, hidden = rnn(inp, hidden)
    # loss calculation and backprop
    loss = criterion(output, target)
    loss.backward()
    # clipping + optimization
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    # mu_loss = average loss over the batch
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)

#################################################
## Neural Network Training
#################################################

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses, epoch_losses, train_losses = [], [], []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)
            epoch_losses.append(loss)
            train_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                # log the train loss to AML run
                run.log('batch_loss', np.average(batch_losses))
                batch_losses = []

        # log the train loss to AML run
        run.log('epoch_loss', np.average(epoch_losses))
        epoch_losses = []

    # log the train loss to AML run
    run.log('train_loss', np.average(train_losses))

    # returns a trained rnn
    return rnn

#################################################
## Hyperparameters
#################################################

# Data params
# Sequence Length
sequence_length = args.sequence_length # of words in a sequence
# Batch Size
batch_size = args.batch_size
# data loader
train_loader = batch_data(int_text, sequence_length, batch_size)

# Training parameters
# Number of Epochs
num_epochs = args.num_epochs
# Learning Rate
learning_rate = args.learning_rate

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = args.embedding_dim
# Hidden Dimension
hidden_dim = args.hidden_dim
# Number of RNN Layers
n_layers = args.num_layers

# Show stats for every n number of batches
show_every_n_batches = 500

#################################################
## Log the Hyperparameter Metrics to the AML Run
#################################################

run.log('epochs', np.int(num_epochs))
run.log('batch_size', np.int(batch_size))
run.log('learning_rate', np.float(learning_rate))

run.log('sequence_length', np.int(sequence_length))
run.log('embedding_dim', np.int(embedding_dim))
run.log('hidden_dim', np.int(hidden_dim))
run.log('layers', np.int(n_layers))

#################################################
## Visualize Model
#################################################

test_rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

test_inputs = torch.zeros((sequence_length, batch_size), dtype=torch.long)
test_hidden = torch.zeros((n_layers, sequence_length, hidden_dim), dtype=torch.float)
summary(test_rnn, test_inputs, hidden=(test_hidden, test_hidden))

#################################################
## Train Model
#################################################

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
save_filename = os.path.join(args.output_dir, "trained_rnn.pt")
torch.save(trained_rnn, save_filename)
print('Model Trained and Saved')
