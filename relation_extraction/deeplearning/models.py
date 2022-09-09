from torch import nn
import torch.nn.functional as F
import torch

class MLP_PretrainedVecs_1(nn.Module):
    def __init__(self,embed_dim,num_class,hidden_size):
        super(MLP_PretrainedVecs_1,self).__init__()

        self.fc1= nn.Linear(embed_dim,hidden_size)
        self.fc2= nn.Linear(hidden_size,num_class)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        h1=self.relu(self.fc1(x))
        z=self.fc2(h1)
        return z
    
class MLP_PretrainedVecs_2(nn.Module):
    def __init__(self,embed_dim,num_class,hidden_size):
        super(MLP_PretrainedVecs_2,self).__init__()
        
#         self.embedding=nn.Embedding(vocab_size,embed_dim)
#         self.embedding.weight=nn.Parameter(word_embeddings,requires_grad=False)
        
        self.fc1= nn.Linear(embed_dim,hidden_size)
        self.fc2= nn.Linear(hidden_size,hidden_size//2)
        self.fc3= nn.Linear(hidden_size//2,num_class)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,x):
#         emb_sent=self.embedding.permute(1,0,2)
#         print (x.shape)
        h1=self.relu(self.fc1(x))
        h2=self.relu(self.fc2(h1))
        z=self.fc3(h2)
#         print (z.shape)
        return self.softmax(z)
    
class RNN_PretrainedVecs_1(nn.Module):
    def __init__(self,embed_dim,num_class,hidden_size):
        super(RNN_PretrainedVecs_1,self).__init__()
        
        self.rnn=nn.RNN(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1= nn.Linear(hidden_size,hidden_size//2)
        self.fc2= nn.Linear(hidden_size//2,num_class)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,x):
        out,h_n=self.rnn(x)
#         print (out.shape,h_n[-1].shape)
        h1=self.relu(self.fc1(h_n[-1]))
        z=self.fc2(h1)
#         print (z.shape)
        return z

class RNN_2(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_class,hidden_size,padding_index=0):
        super(RNN_2,self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
        self.rnn=nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1= nn.Linear(hidden_size,num_class)
        
    def forward(self,x):
#         print ([[idx2word[i] for i in j.detach().numpy()] for j in x])
        x=self.emb(x)
        out,h_n=self.rnn(x)
#         print (out[:,-1,:].shape,h_n[-1].shape)
        h1=self.fc1(out[:,-1,:])
#         h1=self.fc1(h_n[-1].squeeze(0))
#         print (z.shape)
        return h1


class LSTM_1(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_class,hidden_size,padding_index=0):
        super(LSTM_1,self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
        self.rnn=nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1= nn.Linear(hidden_size,num_class)
        
    def forward(self,x):
#         print ([[idx2word[i] for i in j.detach().numpy()] for j in x])
        x=self.emb(x)
        out,h_n=self.rnn(x)
        h1=self.fc1(out[:,-1,:])
#         print (out[:,-1,:].shape,h_n[-1].shape)
        return out[:,-1,:],h1
    
class MLP_PretrainedVecs_3(nn.Module):
    def __init__(self,embed_dim,num_class,hidden_size):
        super(MLP_PretrainedVecs_3,self).__init__()

        self.fc1= nn.Linear(embed_dim,hidden_size)
        self.fc2= nn.Linear(hidden_size,num_class)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        h1=self.relu(self.fc1(x))
        z=self.fc2(h1)
        return h1,z


class RNN_PretrainedVecs_4(nn.Module):
    def __init__(self,vocab_size,embedding_matrix,embed_dim,num_class,hidden_size,padding_index=0,freeze=True):
        super(RNN_PretrainedVecs_4,self).__init__()
        self.emb =  nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float(), freeze)
        self.rnn=nn.RNN(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1= nn.Linear(hidden_size,num_class)
        
    def forward(self,x):
#         print ([[idx2word[i] for i in j.detach().numpy()] for j in x])
        x=self.emb(x)
        out,h_n=self.rnn(x)
        h1=self.fc1(out[:,-1,:])
#         print (out[:,-1,:].shape,h_n[-1].shape)
        return out[:,-1,:],h1
    

class CNN(nn.ModuleList):

    def __init__(self, vocab_size,embed_dim,num_class,padding_index=0):
        super(CNN, self).__init__()

        self.dropout = nn.Dropout(0.25)
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5
        
        self.stride = 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)

        self.conv_1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim//2, kernel_size=self.kernel_1, stride=self.stride)
        self.conv_2 = nn.Conv1d(embed_dim, embed_dim//2, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(embed_dim, embed_dim//2, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(embed_dim, embed_dim//2, self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.dropout = nn.Dropout(0.6)
        
        self.fc = nn.Linear(embed_dim//2*4,num_class)
    
    def forward(self, x):

        x = self.embedding(x).transpose(1, 2)
        
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = F.max_pool1d(x1,kernel_size=x1.shape[2])

        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = F.max_pool1d(x2,kernel_size=x2.shape[2])

        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = F.max_pool1d(x3,kernel_size=x3.shape[2])

        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = F.max_pool1d(x4,kernel_size=x4.shape[2])

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)
        out = self.dropout(union)
        out = self.fc(union)

        return union,out.squeeze()