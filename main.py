import os
import sys
import numpy as np
from model.models import LSTM_attentModel
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe
import torch
import torch.nn.functional
import torch.optim 
from torch.autograd import Variable

def preprocess(test_sen=None):
    tokenize = lambda x: x.split()
    SENTENCE = data.Field(sequential=True, tokenize=tokenize, lower=True,
                      include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.Field(is_target=True, unk_token=None)

    train_data, test_data = datasets.IMDB.splits(SENTENCE, LABEL)
    SENTENCE.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = SENTENCE.vocab.vectors
    print("Length of Textual Vocabulary: " + str(len(SENTENCE.vocab)))
    print("Vector size of Textual Vocabulary: ", SENTENCE.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    # Further splitting of training_data to create new training_data & validation_data
    train_data, valid_data = train_data.split()
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(SENTENCE.vocab)

    return SENTENCE, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


SENTENCE, vocab_size, wghts, iteration, iteration_valid, test_iter = preprocess()


def gradient(model,clip):
    grad_val=list(filter(lambda x: x.grad is not None,model.parameters()))
    for i in grad_val:
        i.grad.data.clamp_(-clip,clip)

def trainer(model,iteration,epoch):
    total_loss=0
    total_accuracy=0
    step=0
    #model.cuda()
    optimize=torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    model.train()
    for index,batch in  enumerate(iteration):
        sentence=batch.text[0]
        label=batch.label
        label=torch.autograd.Variable(label).long()
        if torch.cuda.is_available():
            sentence=sentence.cuda()
            label=label.cuda()
        if(sentence.size()[0] is not 32):
            continue
        optimize.zero_grad()
        pred = model(sentence)
        loss=loss_fn(pred,label)
        correct_res=(torch.max(pred, dim=1)[1].view(label.size()).data == label.data).float().sum()
        accuracy = 100.0 * correct_res/len(batch)
        loss.backward()
        gradient(model,1e-1)
        optimize.step()
        step=step+1

        if step % 100 ==0:
            print(f"epoch:{epoch+1},Index:{index+1},Accuracy:{accuracy.item(): .2f},Loss:{loss.item(): .4f}%")
        total_loss+=loss.item()
        total_accuracy += accuracy.item()

        return total_loss/len(iteration), total_accuracy/len(iteration)

def model_evaluate(model,iteration_valid):
    total_loss = 0
    total_accuracy = 0
    model.cuda()
    optimize = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(iteration_valid):
            sentence = batch.text[0]
            label = batch.label
            label = torch.autograd.Variable(label).long()
            if torch.cuda.is_available():
                sentence = sentence.cuda()
                label = label.cuda()
            if(sentence.size()[0] is not 32):
                continue
            optimize.zero_grad()
            pred = model(sentence)
            loss = loss_fn(pred, label)
            correct_res = (torch.max(pred, dim=1)[1].view(label.size()).data == label.data).float().sum()
            accuracy = 100.0 * correct_res/len(batch)
            total_loss += loss.item()
            total_accuracy += accuracy.item()

    return total_loss/len(iteration_valid), total_accuracy/len(iteration_valid)


learn_rate=2e-5
batch_size=32
op_size=2
hidden_size=256
embed_len=300

model = LSTM_attentModel(vocab_size, op_size, hidden_size,batch_size, wghts, embed_len)
loss_fn=torch.nn.functional.cross_entropy

for epoch in range(10):
    t_loss,t_accuracy = trainer(model,iteration,epoch)
    v_loss,v_accuracy = model_evaluate(model,iteration_valid)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

test_loss, test_accuracy= model_evaluate(model,test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')



