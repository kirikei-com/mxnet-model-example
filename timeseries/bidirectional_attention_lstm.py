class BidirAttentionLSTM(gluon.HybridBlock):
    def __init__(self, time, hidden_dim, r, d, output_dim=1, **kwargs):
        super(BidirAttentionLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.time = time
        self.r = r
        self.d = d
            
        with self.name_scope():
            self.drop1 = nn.Dropout(0.5)
            self.bi_rnn = rnn.BidirectionalCell(
                 rnn.LSTMCell(hidden_size=self.hidden_dim // 2),  #mx.rnn.LSTMCell doesnot work
                 rnn.LSTMCell(hidden_size=self.hidden_dim // 2)
            )
            self.w_1 = nn.Dense(self.d, use_bias = False)
            self.w_2 = nn.Dense(self.r, use_bias = False)
            self.dense1 = nn.Dense(64, activation='relu')
            self.dense2 = nn.Dense(output_dim)
        
    def hybrid_forward(self, F, x, state):
        # inputs shape: (batch, t, c, w)
        h, _ = self.bi_rnn.unroll(
            length = self.time, inputs=x, layout='NTC', merge_outputs=True, begin_state=state)
       
        _h = F.reshape(h, shape=(-1, self.hidden_dim))
        _w = F.tanh(self.w_1(_h))
        w = self.w_2(_w)
        _att = F.reshape(w, shape=(-1, self.time, self.r)) # Batch * Timestep * r
        att = F.softmax(_att, axis=1)
        attend_out = F.linalg.gemm2(att, h, transpose_a=True)
        dense1 = self.dense1(attend_out)
        drop1 = self.drop1(dense1)
        dense2 = self.dense2(drop1)
        out = F.sigmoid(dense2)
        
        return out
