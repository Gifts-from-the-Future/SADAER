"""
Some code is adapted from prior implementations of related work,
which are licensed under the MIT License.

Original authorship and specific references will be added
after the peer-review process.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import numpy as np
class DiffusionGraphConv2(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        super(DiffusionGraphConv2, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)
    @staticmethod
    def _build_sparse_matrix(L):
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1)).to(inputs.device)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size
        x0 = inputs_and_state
        x = torch.unsqueeze(x0, dim=1)
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.matmul(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2)
                    x1, x0 = x2, x1
        num_matrices = len(supports) * \
            self._max_diffusion_step + 1
        x = torch.transpose(x, dim0=1, dim1=2)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
class DCGRUCell2(nn.Module):
    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True,
            attention_num=2):
        super(DCGRUCell2, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":
            self._num_supports = 1
        elif filter_type == "random_walk":
            self._num_supports = 1
        elif filter_type == "dual_random_walk":
            self._num_supports = 2
        elif filter_type == "attention":
            self._num_supports = attention_num
        else:
            self._num_supports = 1
        self.dconv_gate = DiffusionGraphConv2(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv2(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type)
    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size
    def forward(self, adjs, inputs, state):
        state = state.to(inputs.device)
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(adjs, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(
                output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units)).to(inputs.device)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units)).to(inputs.device)
        c = self.dconv_candidate(adjs, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        return output, new_state
    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)
    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass
    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='attention',
                 device=None,attention_num=2):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        encoding_cells.append(
            DCGRUCell2(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type,
                attention_num=attention_num))
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell2(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type,
                    attention_num=attention_num))
        self.encoding_cells = nn.ModuleList(encoding_cells)
    def forward(self, inputs, initial_hidden_state, adjs, time_step):
        seq_leng_step = inputs.shape[0]
        batch = inputs.shape[1]
        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_leng_step):
                _, hidden_state = self.encoding_cells[i_layer](
                    adjs[int(t / time_step)], current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)
        return current_inputs, hidden_state, output_hidden
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        return torch.stack(init_states, dim=0)

class Multi_DCRNNEncoder(nn.Module):
    def __init__(self, shared_layers,num_tasks,input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='attention',
                 device=None,attention_num=2):
        super(Multi_DCRNNEncoder, self).__init__()
        self.shared_layers = max(shared_layers-2,0)
        self.num_tasks = num_tasks
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        self.shared_cells = nn.ModuleList()
        for i in range(self.shared_layers):
            if i==0:
                cell = DCGRUCell2(
                    input_dim=input_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type,
                    attention_num=attention_num)
            else:
                cell = DCGRUCell2(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type,
                    attention_num=attention_num)
            self.shared_cells.append(cell)

        self.task_cells = nn.ModuleList()
        for _ in range(num_tasks):
            task_layers = nn.ModuleList()
            for i in range(self.shared_layers,num_rnn_layers):
                if i==0:
                    cell = DCGRUCell2(
                        input_dim=input_dim,
                        num_units=hid_dim,
                        max_diffusion_step=max_diffusion_step,
                        num_nodes=num_nodes,
                        nonlinearity=dcgru_activation,
                        filter_type=filter_type,
                        attention_num=attention_num)
                else:
                    cell = DCGRUCell2(
                        input_dim=hid_dim,
                        num_units=hid_dim,
                        max_diffusion_step=max_diffusion_step,
                        num_nodes=num_nodes,
                        nonlinearity=dcgru_activation,
                        filter_type=filter_type,
                        attention_num=attention_num)
                task_layers.append(cell)
            self.task_cells.append(task_layers)
    def forward(self, inputs, shared_init_states,task_init_states, adjs, time_step):
        seq_leng_step = inputs.shape[0]
        batch = inputs.shape[1]
        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.shared_layers):
            hidden_state = shared_init_states[i_layer]
            output_inner = []
            for t in range(seq_leng_step):
                _, hidden_state = self.shared_cells[i_layer](
                    adjs[int(t / time_step)], current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)
        if self.shared_layers > 0:
            final_shared_statue = hidden_state
        else:
            final_shared_statue = None

        final_hidden_state = []
        for task_id in range(self.num_tasks):
            task_current_inputs = current_inputs
            init_hidden_state = task_init_states[task_id]
            for i_layer in range(self.num_rnn_layers-self.shared_layers):
                hidden_state = init_hidden_state[i_layer]
                output_inner = []
                for t in range(seq_leng_step):
                    _, hidden_state = self.task_cells[task_id][i_layer](
                        adjs[task_id][int(t / time_step)], task_current_inputs[t, ...], hidden_state)
                    output_inner.append(hidden_state)
                output_hidden.append(hidden_state)
                task_current_inputs = torch.stack(output_inner, dim=0).to(
                    self._device)
                final_task_state = hidden_state
            final_hidden_state.append(final_task_state)
        final_hidden_state = torch.stack(final_hidden_state, dim=0) if final_hidden_state else None

        return final_shared_statue,final_hidden_state
    def init_hidden(self, batch_size,device):
        shared_init_states = []
        task_init_states = []
        for i in range(self.shared_layers):
            shared_init_states.append(self.shared_cells[i].init_hidden(batch_size).to(device))
        for i in range(self.num_tasks):
            task_status = []
            for j in range(self.num_rnn_layers-self.shared_layers):
                task_status.append(self.task_cells[i][j].init_hidden(batch_size).to(device))
            task_init_states.append(torch.stack(task_status, dim=0))
        shared_init_states = torch.stack(shared_init_states, dim=0) if shared_init_states else None
        task_init_states = torch.stack(task_init_states, dim=0) if task_init_states else None
        return shared_init_states, task_init_states
def convent_to_coo(adj_mats):
    batch = adj_mats.shape[0]
    edge_index_list = []
    edge_attr_list = []
    for i in range(batch):
        adj_mat = adj_mats[i]
        edge_index, edge_attr = dense_to_sparse(adj_mat)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
    return edge_index_list,edge_attr_list
class GADN(nn.Module):
    def __init__(self, in_channels=62*1, classes=1, heads=2, concat=True,num_rnn_layers=2,max_diffusion_step=2,gat1_channels=64,gat2_channels=32,hid_dim=64,attention_num=2):
        super(GADN, self).__init__()

        self.encoder = DCRNNEncoder(input_dim=gat1_channels*heads,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=hid_dim, num_nodes=19,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation='tanh',
                                    filter_type='attention',
                                    attention_num=attention_num)
        self.conv = GATv2Conv(in_channels, gat1_channels, heads=heads, concat=concat,edge_dim=1)
        self.conv2 = GATv2Conv(gat1_channels*heads, gat2_channels, heads=heads, concat=concat, edge_dim=1)
        self.fc = nn.Linear(hid_dim, classes)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x,adj_mats):
        batch = x.shape[0]
        num_graph = x.shape[1]
        time_step = x.shape[2]
        num_nodes = x.shape[3]
        x_copy = x.clone()
        x = x.permute(1,0,3,2,4)
        x = x.reshape(num_graph,batch,num_nodes,-1)
        adj_mats = adj_mats.permute(1,0,2,3)
        all_output = []
        all_alpha = []
        for i in range(num_graph):
            edge_index, edge_attr = convent_to_coo(adj_mats[i])
            data = x[i]
            output = []
            alpha = []
            for j in range(batch):
                out1 = self.conv(x=data[j],edge_index=edge_index[j],edge_attr=edge_attr[j].float())
                out,attention_weight = self.conv2(x=out1,edge_index=edge_index[j],edge_attr=edge_attr[j].float(),return_attention_weights=True)
                H = attention_weight[1].size(1)
                dense_matrix = torch.zeros((H, 19, 19))
                indices = attention_weight[0]
                values = attention_weight[1]
                dense_matrix = torch.zeros((H, 19, 19), device=x.device)
                dense_matrix[:, indices[0, :], indices[1, :]] = values.t()
                alpha.append(dense_matrix)
                output.append(out1)
            all_output.append(torch.stack(output,dim=0))
            all_alpha.append(torch.stack(alpha,dim=0))
        all_output = torch.stack(all_output,dim=0)
        all_alpha = torch.stack(all_alpha,dim=0)

        all_alpha = all_alpha.permute(0,2,1,3,4).to(x.device)
        all_output = all_output.reshape(num_graph, batch, -1)
        init_hidden_state = self.encoder.init_hidden(
            batch).to(x.device)

        seq_output, sup_output, _ = self.encoder(all_output, init_hidden_state, all_alpha, 1)

        output = sup_output.reshape(batch, num_nodes, -1)
        logits = self.fc(self.relu(self.dropout(output)))
        pool_logits, _ = torch.max(logits, dim=1)
        return pool_logits

class multi_GADN(nn.Module):
    def __init__(self, shared_layers=2,num_tasks=3,class_list=[1,3,4],in_channels=62*1,heads=2, concat=True,num_rnn_layers=2,max_diffusion_step=2,gat1_channels=64,gat2_channels=32,hid_dim=64,attention_num=2):
        super(multi_GADN, self).__init__()
        self.shared_layers = shared_layers
        self.num_tasks = num_tasks
        self.shared_conv1 = GATv2Conv(in_channels, gat1_channels, heads=heads, concat=concat, edge_dim=1) \
            if shared_layers >= 1 else None
        self.shared_conv2 = GATv2Conv(gat1_channels * heads, gat2_channels, heads=heads, concat=concat, edge_dim=1) \
            if shared_layers >= 2 else None
        if shared_layers < 2:
            self.task_conv2 = nn.ModuleList([
                GATv2Conv(gat1_channels * heads, gat2_channels, heads=heads, concat=concat, edge_dim=1)
                for _ in range(num_tasks)
            ])
        self.multi_encoder = Multi_DCRNNEncoder(
            shared_layers=shared_layers, num_tasks=num_tasks,
            input_dim=gat1_channels * heads,
            max_diffusion_step=max_diffusion_step,
            hid_dim=hid_dim, num_nodes=19,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation='tanh',
            filter_type='attention',
            attention_num=attention_num)

        self.task_fc = nn.ModuleList([
            nn.Linear(hid_dim, out_dim) for out_dim in class_list
        ])
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x,adj_mats):
        batch = x.shape[0]
        num_graph = x.shape[1]
        time_step = x.shape[2]
        num_nodes = x.shape[3]
        x_copy = x.clone()
        x = x.permute(1,0,3,2,4)
        x = x.reshape(num_graph,batch,num_nodes,-1)
        adj_mats = adj_mats.permute(1,0,2,3)
        all_output = []
        all_alpha = [[] for _ in range(self.num_tasks)]
        for i in range(num_graph):
            edge_index, edge_attr = convent_to_coo(adj_mats[i])
            data = x[i]
            output = []
            task_alphas = [[] for _ in range(self.num_tasks)]
            for j in range(batch):
                shared_out1 = self.shared_conv1(x=data[j],edge_index=edge_index[j],edge_attr=edge_attr[j].float())
                if self.shared_layers>=2:
                    shared_out2, attn2 = self.shared_conv2(x=shared_out1, edge_index=edge_index[j], edge_attr=edge_attr[j].float(),
                                                       return_attention_weights=True)
                    H = attn2[1].size(1)
                    dense_matrix = torch.zeros((H, 19, 19))
                    indices = attn2[0]
                    values = attn2[1]
                    dense_matrix = torch.zeros((H, 19, 19), device=x.device)
                    dense_matrix[:, indices[0, :], indices[1, :]] = values.t()
                    for task_id in range(self.num_tasks):
                        task_alphas[task_id].append(dense_matrix)
                else:
                    for task_id in range(self.num_tasks):
                        out2, attn2 = self.task_conv2[task_id](x=shared_out1, edge_index=edge_index[j],
                                                               edge_attr=edge_attr[j].float(),
                                                               return_attention_weights=True)
                        H = attn2[1].size(1)
                        dense_matrix = torch.zeros((H, 19, 19))
                        indices = attn2[0]
                        values = attn2[1]
                        dense_matrix = torch.zeros((H, 19, 19), device=x.device)
                        dense_matrix[:, indices[0, :], indices[1, :]] = values.t()
                        task_alphas[task_id].append(dense_matrix)

                output.append(shared_out1)
            all_output.append(torch.stack(output,dim=0))
            for task_id, alpha in enumerate(task_alphas):
                all_alpha[task_id].append(torch.stack(alpha,dim=0))
        all_output = torch.stack(all_output,dim=0)
        for task_id,alphas in enumerate(all_alpha):
            this_alpha = torch.stack(alphas,dim=0)
            all_alpha[task_id] = this_alpha.permute(0,2,1,3,4).to(x.device)

        all_output = all_output.reshape(num_graph, batch, -1)

        shared_init_states,task_init_states = self.multi_encoder.init_hidden(batch,x.device)
        final_hidden_state = []
        if self.shared_layers==4:
            final_shared_statue,_ = self.multi_encoder(all_output, shared_init_states, task_init_states, all_alpha, 1)
            for task_id in range(self.num_tasks):
                final_hidden_state.append(final_shared_statue)
        else:
            _,final_hidden_state = self.multi_encoder(all_output, shared_init_states,task_init_states,all_alpha, 1)

        outputs = []
        for task_id in range(self.num_tasks):
            output = final_hidden_state[task_id].reshape(batch, num_nodes, -1)
            logits = self.task_fc[task_id](self.relu(self.dropout(output)))
            pool_logits, _ = torch.max(logits, dim=1)
            outputs.append(pool_logits)

        return outputs