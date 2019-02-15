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
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM(in, out)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Linear layer to map hidden_size to vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        # Initialize the hidden state
        self.hidden = (torch.zeros(1, batch_size, self.hidden_size).to(self.device), torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        
        captions = captions[:, :-1] 
        embeddings = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        out_lstm, self.hidden = self.lstm(embeddings, self.hidden)
        
        outputs = self.linear(out_lstm)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        captions = []
        
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            
            outputs = self.linear(outputs.squeeze(1))
            
            outputs = outputs.squeeze(0)
            output = outputs.max(0)[1]            
            
            # Convert tensor to variable and append to captions list
            captions.append(output.item())
            
            # Input for the next time 't+1' is the output from current time 't'
            inputs = self.word_embeddings(output).unsqueeze(0).unsqueeze(0)
        
        return captions