class BidirAttentionConv1DLSTM(gluon.HybridBlock):
    def __init__(self, time, input_shape, output_channel, r, d, i2h_kernel=4, h2h_kernel=3, output_dim=1, **kwargs):
        super(BidirAttentionConv1DLSTM, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.output_channel = output_channel
        self.time = time
        self.r = r
        self.d = d
        self.conv_by_i2h_kernel = input_shape[1] - i2h_kernel + 1
            
        with self.name_scope():
            self.drop1 = nn.Dropout(0.5)
            self.bi_rnn = rnn.BidirectionalCell(
                 gluon.contrib.rnn.Conv1DLSTMCell(
                     input_shape, output_channel//2, i2h_kernel, h2h_kernel, conv_layout='NCW'), 
                 gluon.contrib.rnn.Conv1DLSTMCell(
                     input_shape, output_channel//2, i2h_kernel, h2h_kernel, conv_layout='NCW')
            )
            self.w_1 = nn.Dense(self.d, use_bias = False)
            self.w_2 = nn.Dense(self.r, use_bias = False)
            self.dense1 = nn.Dense(64, activation='relu')
            self.dense2 = nn.Dense(output_dim)
        
    def hybrid_forward(self, F, x, state):
        # inputs shape: (batch, t, c, w)
        h, _ = self.bi_rnn.unroll(
            length = self.time, inputs=x, layout='NTC', merge_outputs=True, begin_state=state)
        # output shape: (batch, t, output_channel(oc), conved_by_i2h_kernel(ck))
        # output is 2-dimension unlike normal lstm
        
        _h = F.reshape(h, shape=(-1, self.conv_by_i2h_kernel*self.output_channel)) # (batch * time, ck * oc)
        _w = F.tanh(self.w_1(_h))
        w = self.w_2(_w) # transform r dimension, # (batch * time, r)
        _att = F.reshape(w, shape=(-1, self.time, self.r))  # (batch, time, r)
        att = F.softmax(_att, axis=1) # softmax for time direction
        # reshape h to att.shape
        h_reshaped = F.reshape(h, shape=(-1, self.time, self.conv_by_i2h_kernel*self.output_channel)) # (batch, time, ck * oc)
        # calclate attention weight * h
        attend_out = F.linalg.gemm2(att, h_reshaped, transpose_a=True)
        
        dense1 = self.dense1(attend_out)
        drop1 = self.drop1(dense1)
        dense2 = self.dense2(drop1)
        out = F.sigmoid(dense2)
        
        return out
    
    def begin_state(self, *args, **kwargs):
        return self.bi_rnn.begin_state(*args, **kwargs)
