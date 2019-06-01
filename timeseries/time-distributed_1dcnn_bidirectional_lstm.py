class TDCNNBidirLSTM(gluon.HybridBlock):
    def __init__(self, time=10, **kwargs):
        super(TDCNNBidirLSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv1D(64, 3, activation="relu") # NCW
            self.conv2 = nn.Conv1D(64, 3, activation="relu")
            self.maxpool = nn.MaxPool1D(pool_size=2)
            self.drop1 = nn.Dropout(0.5)
            self.drop2 = nn.Dropout(0.5)
            self.flatten = nn.Flatten()
            self.lstm = rnn.LSTM(100, num_layers=2, bidirectional=True, layout='NTC')
            self.dense1 = nn.Dense(100, activation='relu')
            self.dense2 = nn.Dense(1)
        self.time = time
        
    def hybrid_forward(self, F, x, state):
        # inputs shape: (batch, t, c, w)
        merged_time_output_list = []
        
        for i in range(self.time):
            # apply CNN layer each time
            sliced = F.slice_axis(x, axis=1, begin=i, end=i+1)
            conv1 = self.conv1(F.squeeze(sliced, axis=1))# N, C, W
            conv2 = self.conv2(conv1)
            drop1 = self.drop1(conv2)
            max1 = self.maxpool(drop1)
            fout = self.flatten(max1)
            
            # for concating, expand dims
            merged_time_output_list.append(F.expand_dims(fout, axis=1))
            
        # mxnet cannot feed list but tuple
        merged_time_output = F.concat(*merged_time_output_list, dim=1)
        lstm_out, state = self.lstm(merged_time_output, state) # N, T, C
        drop2 = self.drop2(lstm_out)
        dense1 = self.dense1(drop2)
        dense2 = self.dense2(dense1) # output
        out = F.sigmoid(dense2)
        
        return out
    
    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)
