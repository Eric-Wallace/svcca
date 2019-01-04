import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
from models.CNN import CNN
import cca_core
from copy import deepcopy

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

positive_dictionary = []
with open('positive.txt','r') as f:
    for line in f:
        positive_dictionary.append(line.strip().lower())
negative_dictionary = []
with open('negative.txt','r') as f:
    for line in f:
        negative_dictionary.append(line.strip().lower())

def train_model(model, optim, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(device)
    model.train()    
    steps = 0    
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        #target = torch.autograd.Variable(target).long()        
        text = text.to(device)
        target = target.to(device)
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction, _ = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 10000 == 0:
            print("Epoch: ", epoch+1)
            print("Idx: ", idx+1)
            print("Training Loss: ", loss.item())
            print("Training Accuracy: ", acc.item())
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            #target = torch.autograd.Variable(target).long()
            text = text.to(device)
            target = target.to(device)            
            prediction, all_out = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def svcca_two_models(model1, model2, val_iter):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            target = batch.label
            if idx == 0:
                all_text = text
                all_target = target
            else:
                all_text = torch.cat([all_text,text], 0)
                all_target = torch.cat([all_target,target], 0)
        all_text = all_text.to(device)
        all_target = all_target.to(device)        
        prediction, model1_allout = model1(all_text)
        prediction, model2_allout = model2(all_text)
        results = cca_core.get_cca_similarity(model2_allout.t().to("cpu").numpy(), model1_allout.t().to("cpu").numpy(), verbose=True)
        print(np.mean(results['cca_coef1']))

def svcca_no_sentiment(model1, val_iter):
    model1.eval()    
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            target = batch.label
            if idx == 0:
                all_text = text
                all_target = target
            else:
                all_text = torch.cat([all_text,text], 0)
                all_target = torch.cat([all_target,target], 0)
        all_text = all_text.to(device)
        all_target = all_target.to(device)            
        prediction, model1_allout = model1(all_text)        

        all_text_no_sentiment = deepcopy(all_text)
        for example_ind, ex in enumerate(all_text):
            for word_ind, word in enumerate(ex):
                if (TEXT.vocab.itos[word] in positive_dictionary):
                    all_text_no_sentiment[example_ind][word_ind] = 1
                if (TEXT.vocab.itos[word] in negative_dictionary):
                    all_text_no_sentiment[example_ind][word_ind] = 1

        prediction, model1_allout_no_sentiment = model1(all_text_no_sentiment)        
        # print(model1_allout.shape)
        # print(model1_allout_no_sentiment.shape)
        results = cca_core.get_cca_similarity(model1.allout.t().to("cpu").numpy(), model1_allout_no_sentiment.t().to("cpu").numpy(), verbose=True)        
        print(np.mean(results['cca_coef1']))

learning_rate = 2e-5
batch_size = 32
output_size = 5
hidden_size = 256
embedding_length = 300

# CNN hyperparameters
# in_channels = 1
# out_channels = 200
# kernel_heights = (3,4,5)
# stride = 1
# padding = 0
# keep_probab = 0.3

# model = CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, word_embeddings)    
model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)


loss_fn = F.cross_entropy
# optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001, momentum=0.9)
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

epochs = 30
for epoch in range(epochs):
    val_loss, val_acc = eval_model(model, valid_iter)
    print("Val loss", val_loss)
    print("Val acc", val_acc)
    train_loss, train_acc = train_model(model, optim, train_iter, epoch)
    #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

svcca_no_sentiment(model, valid_iter)    
#test_loss, test_acc = eval_model(model, test_iter)
#print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Positive")
# else:
#     print ("Sentiment: Negative")

