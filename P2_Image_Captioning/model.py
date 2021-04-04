import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        
        #Embedding.
        self.word_embedding=nn.Embedding(vocab_size,embed_size)
        
        #LSTM layer.
        self.lstm_layer=nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        
        #Fully connected layer.
        self.fc_layer=nn.Linear(hidden_size,vocab_size)
    
# Do not set hidden size as batch size is not input to the decoder parameter.  
        #self.hidden_size=hidden_size
        #self.hidden= self.init_hidden()
        
#    def init_hidden(self):
#        return (torch.zeros(1, 1, self.hidden_size),
#            torch.zeros(1, 1, self.hidden_size))
        
    
    def forward(self, features, captions):
       
        # The below step is necessary as we need not input <end> token. If this step is not done, then outputs.shape[1]!=captions.shape[1]    
        captions=captions[:,:-1]
        
        word_embedding=self.word_embedding(captions)
        #print("Word embedding",word_embedding.shape)
        
        #Adjust dimensions of features so that it can be concatenated.
        features=features.unsqueeze(1)
        
        word_embedding=torch.cat((features,word_embedding),1)
        #print("Word embedding after cat",word_embedding.shape)
    
        #lstm_out,self.hidden=self.lstm(word_embedding[:,:-1],self.hidden)

        lstm_out,hidden_state=self.lstm_layer(word_embedding) #Hidden states default to 0 (h0,c0).
        
        output=self.fc_layer(lstm_out)
        
        #Do not return a tuple.
        return output
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        #Implementing a greedy search where it adds the max scored word to the list.
        
        #Empty list.
        predicted_sentence=[]
        
        while True:
            #Pass through the LSTM layer and generate the score. 
            lstm_output,states=self.lstm_layer(inputs,states)
            scores=self.fc_layer(lstm_output)
         
            #Find the max score.
            scores=scores.squeeze(1)
            max_score,indices=torch.max(scores,dim=1)
            # If it is <end> token then break.
            if indices==1:
                break
            #Append the word index to the list.    
            predicted_sentence.append(indices.item())
            #Create word embedding for the next time sequence.
            word_embed=self.word_embedding(indices)
            #Adjust the dimensions.
            inputs=word_embed.unsqueeze(1)
                
        
        return predicted_sentence
       # pass