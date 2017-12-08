import numpy as np
import dynet_config
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy
from os import path

dy.__version__

import sys
sys.path.append("..")
import utils as u

# change this string to match the path on your computer
path_to_root = "/Users/mcapizzi/Github/dynet_tutorial/"

train_tokens, train_labels, _, _, test_tokens, test_labels = u.import_pos(path_to_root)

train_tokens[0][:5], train_labels[0][:5]



l2i = u.labels_to_index_map(train_labels)
l2i.items()

train_labels = [[l2i[l] for l in sent] for sent in train_labels]
train_labels[0][:5]



test_labels = [[l2i[l] for l in sent] for sent in test_labels]
test_labels[0][:5]

i2l = dict((v,k) for k,v in l2i.items())
i2l.items()

RNN_model = dy.ParameterCollection()    # used to be called dy.Model()
RNN_model

################
# HYPERPARAMETER
################
# size of word embedding (if using "random", otherwise, dependent on the loaded embeddings)
embedding_size = 300

################
# HYPERPARAMETER
################
# size of hidden layer of `RNN`
hidden_size = 200

################
# HYPERPARAMETER
################
# number of layers in `RNN`
num_layers = 1

w2i_random = u.build_w2i_lookup(train_tokens)
w2i_random["the"]



###### CHOOSE HERE which approach you want to use. ######
# embedding_approach, embedding_dim = "pretrained", emb_matrix_pretrained.shape[1]
embedding_approach, embedding_dim = "random", embedding_size



if embedding_approach == "pretrained":
    embedding_parameters = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)
    w2i = w2i_pretrained    # ensure we use the correct lookup table
elif embedding_approach == "random":
    embedding_parameters = RNN_model.add_lookup_parameters((len(w2i_random)+1, embedding_dim))
    w2i = w2i_random        # ensure we use the correct lookup table
else:
    raise Exception("you chose poorly...")
dy.parameter(embedding_parameters).npvalue().shape



###### CHOOSE HERE which approach you want to use. ######
# RNN_unit = dy.LSTMBuilder(num_layers, embedding_dim, hidden_size, RNN_model)
RNN_unit = dy.GRUBuilder(num_layers, embedding_dim, hidden_size, RNN_model)
RNN_unit

# W (hidden x num_labels) 
pW = RNN_model.add_parameters(
        (hidden_size, len(list(l2i.keys())))
)
dy.parameter(pW).npvalue().shape



# b (1 x num_labels)
pb = RNN_model.add_parameters(
        (len(list(l2i.keys())))        
)
# note: we are just giving one dimension (ignoring the "1" dimension)
# this makes manipulating the shapes in forward_pass() below easier 
dy.parameter(pb).npvalue().shape

def words2indexes(seq_of_words, w2i_lookup):
    """
    This function converts our sentence into a sequence of indexes that correspond to the rows in our embedding matrix
    :param seq_of_words: the document as a <list> of words
    :param w2i_lookup: the lookup table of {word:index} that we built earlier
    """
    seq_of_idxs = []
    for w in seq_of_words:
        w = w.lower()            # lowercase
        i = w2i_lookup.get(w, 0) # we use the .get() method to allow for default return value if the word is not found
                                 # we've reserved the 0th row of embedding matrix for out-of-vocabulary words
        seq_of_idxs.append(i)
    return seq_of_idxs
    


sample_idxs = words2indexes(["I", "like", "armadillos"], w2i)
sample_idxs

def forward_pass(x):
    """
    This function will wrap all the steps needed to feed one sentence through the RNN
    :param x: a <list> of indices
    """
    # convert sequence of ints to sequence of embeddings
    input_seq = [embedding_parameters[i] for i in x]   # embedding_parameters can be used like <dict>
    # convert Parameters to Expressions
    W = dy.parameter(pW)
    b = dy.parameter(pb)
    # initialize the RNN unit
    rnn_seq = RNN_unit.initial_state()
    # run each timestep through the RNN
    rnn_hidden_outs = rnn_seq.transduce(input_seq)
    # project each timestep's hidden output to size of labels
    rnn_outputs = [dy.transpose(W) * h + b for h in rnn_hidden_outs]
    return rnn_outputs
    


sample_sentence = "i own 7 armadillos .".split()
sample = forward_pass(words2indexes(sample_sentence, w2i))
sample[0].npvalue()

def predict(list_of_outputs):
    """
    This function will convert the outputs from forward_pass() to a <list> of label indexes
    """
    # take the softmax of each timestep
    # note: this step isn't actually necessary as the argmax of the raw outputs will come out the same
    # but the softmax is more "interpretable" if needed for debugging
    pred_probs = [dy.softmax(o) for o in list_of_outputs]     
    # convert each timestep's output to a numpy array
    pred_probs_np = [o.npvalue() for o in pred_probs]
    # take the argmax for each step
    pred_probs_idx = [np.argmax(o) for o in pred_probs_np]
    return pred_probs_idx
    
sample_predict = predict(sample)
sample_predict

sample_predict_labels = [i2l[p] for p in sample_predict]
print(list(zip(sample_sentence, sample_predict_labels)))



################
# HYPERPARAMETER
################
trainer = dy.SimpleSGDTrainer(
    m=RNN_model,
    learning_rate=0.01
)

################
# HYPERPARAMETER
################
batch_size = 256
num_batches_training = int(np.ceil(len(train_tokens) / batch_size))
num_batches_testing = int(np.ceil(len(test_tokens) / batch_size))
num_batches_training, num_batches_testing

# iterate through the first 3 batches of training data (~1500 sentences)

# j = batch index
# k = sentence index (inside batch j)
# l = token index (inside sentence k)
# Note: we are reserving `i` as an index over epochs

for j in range(3):
    # begin a clean computational graph
    dy.renew_cg()
    # build the batch
    batch_tokens = train_tokens[j*batch_size:(j+1)*batch_size]
    batch_labels = train_labels[j*batch_size:(j+1)*batch_size]
    # iterate through the batch
    for k in range(len(batch_tokens)):
        # prepare input: words to indexes
        seq_of_idxs = words2indexes(batch_tokens[k], w2i)
        # make a forward pass
        preds = forward_pass(seq_of_idxs)
        # calculate loss for each token in each example
        loss = [dy.pickneglogsoftmax(preds[l], batch_labels[k][l]) for l in range(len(preds))]
        # sum the loss for each token
        sent_loss = dy.esum(loss)
        # backpropogate the loss for the sentence
        sent_loss.backward()
        trainer.update()
    if j % 5 == 0:
        print("batch {}".format(j+1))
        sample = forward_pass(words2indexes(sample_sentence, w2i))
        predictions = [i2l[p] for p in predict(sample)]
        print(list(zip(sample_sentence, predictions)))
        
def check_score(pred, true_y):
    return 1 if pred == true_y else 0

def check_sentence_score(sentence_scores):
    return 0 if 0 in sentence_scores else 1

def get_accuracy(flat_list_of_scores):
    return float(sum(flat_list_of_scores) / len(flat_list_of_scores))
    
def evaluate(nested_preds, nested_true):
    flat_scores = []
    sentence_scores = []
    for i in range(len(nested_true)):
        scores = []
        pred = nested_preds[i]
        true = nested_true[i]
        for p,t in zip(pred,true):
            score = check_score(p,t)
            scores.append(score)
        sentence_scores.append(check_sentence_score(scores))
        flat_scores.extend(scores)
    overall_accuracy = get_accuracy(flat_scores)
    sentence_accuracy = get_accuracy(sentence_scores)
    return overall_accuracy, sentence_accuracy
    
def test():
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)
    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()
        # build the batch
        batch_tokens = test_tokens[j*batch_size:(j+1)*batch_size]
        batch_labels = test_tokens[j*batch_size:(j+1)*batch_size]
        # iterate through the batch
        for k in range(len(batch_tokens)):
            # prepare input: words to indexes
            seq_of_idxs = words2indexes(batch_tokens[k], w2i)
            # make a forward pass
            preds = forward_pass(seq_of_idxs)
            label_preds = predict(preds)
            all_predictions.append(label_preds)
    return all_predictions
    
final_predictions = test()

overall_accuracy, sentence_accuracy = evaluate(final_predictions, test_labels)
print("overall accuracy: {}".format(overall_accuracy))
print("sentence accuracy (all tags in sentence correct): {}".format(sentence_accuracy))

################
# HYPERPARAMETER
################
num_epochs = 5


def train():

    # i = epoch index
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)

    epoch_losses = []
    overall_accuracies = []
    sentence_accuracies = []
    
    for i in range(num_epochs):
        epoch_loss = []
        for j in range(num_batches_training):
            # begin a clean computational graph
            dy.renew_cg()
            # build the batch
            batch_tokens = train_tokens[j*batch_size:(j+1)*batch_size]
            batch_labels = train_labels[j*batch_size:(j+1)*batch_size]
            # iterate through the batch
            for k in range(len(batch_tokens)):
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(batch_tokens[k], w2i)
                # make a forward pass
                preds = forward_pass(seq_of_idxs)
                # calculate loss for each token in each example
                loss = [dy.pickneglogsoftmax(preds[l], batch_labels[k][l]) for l in range(len(preds))]
                # sum the loss for each token
                sent_loss = dy.esum(loss)
                # backpropogate the loss for the sentence
                sent_loss.backward()
                trainer.update()
                epoch_loss.append(sent_loss.npvalue())
            # check prediction of sample sentence
            if j % 250 == 0:
                print("epoch {}, batch {}".format(i+1, j+1))
                sample = forward_pass(words2indexes(sample_sentence, w2i))
                predictions = [i2l[p] for p in predict(sample)]
                print(list(zip(sample_sentence, predictions)))
        # record epoch loss
        epoch_losses.append(np.sum(epoch_loss))
        # get accuracy on test set
        print("testing after epoch {}".format(i+1))
        epoch_predictions = test()
        epoch_overall_accuracy, epoch_sentence_accuracy = evaluate(epoch_predictions, test_labels)
        overall_accuracies.append(epoch_overall_accuracy)
        sentence_accuracies.append(epoch_sentence_accuracy)
        
    return epoch_losses, overall_accuracies, sentence_accuracies
    
losses, overall_accs, sentence_accs = train()