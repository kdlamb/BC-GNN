import torch
from torch.nn import Linear
from torch.functional import F
from torch_geometric.nn import GCNConv, GraphConv, MetaLayer, MessagePassing
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, SiLU
from torch_scatter import scatter_mean, scatter_sum
from egnn_clean import EGNN

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=CN3sRVuaQ88l
class vanillaGCN(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred):
        super(vanillaGCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(n_f, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, n_hlin)
        self.lin = Linear(n_hlin, n_pred)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GCNlayers(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred, n_layers):
        super(GCNlayers, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_conv_model(n_f,hidden_channels))
        for i in range(0,n_layers):
            self.convs.append(self.build_conv_model(hidden_channels, hidden_channels))
        self.convs.append(self.build_conv_model(hidden_channels, n_hlin))
        self.lin = Linear(n_hlin, n_pred)
        self.num_layers = n_layers

    def build_conv_model(self,input_dim,hidden_dim):
        return GCNConv(input_dim,hidden_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GCNlayerssum(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred, n_layers):
        super(GCNlayerssum, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_conv_model(n_f,hidden_channels))
        for i in range(0,n_layers):
            self.convs.append(self.build_conv_model(hidden_channels, hidden_channels))
        self.convs.append(self.build_conv_model(hidden_channels, n_hlin))
        self.lin = Linear(n_hlin, n_pred)
        self.num_layers = n_layers

    def build_conv_model(self,input_dim,hidden_dim):
        return GCNConv(input_dim,hidden_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GCNlayersmax(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred, n_layers):
        super(GCNlayersmax, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_conv_model(n_f,hidden_channels))
        for i in range(0,n_layers):
            self.convs.append(self.build_conv_model(hidden_channels, hidden_channels))
        self.convs.append(self.build_conv_model(hidden_channels, n_hlin))
        self.lin = Linear(n_hlin, n_pred)
        self.num_layers = n_layers

    def build_conv_model(self,input_dim,hidden_dim):
        return GCNConv(input_dim,hidden_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class WGCNlayers(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred, n_layers):
        super(WGCNlayers, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_conv_model(n_f,hidden_channels))
        for i in range(0,n_layers):
            self.convs.append(self.build_conv_model(hidden_channels, hidden_channels))
        self.convs.append(self.build_conv_model(hidden_channels, n_hlin))
        self.lin = Linear(n_hlin, n_pred)
        self.num_layers = n_layers

    def build_conv_model(self,input_dim,hidden_dim):
        return GCNConv(input_dim,hidden_dim)

    def forward(self, x, edge_index,edge_weight, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index,edge_weight)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class SGCN(torch.nn.Module):
    def __init__(self, n_f, hidden_channels, n_pred,n_layers):
        super(SGCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = SGConv(n_f,hidden_channels,K=n_layers)
        self.lin = Linear(hidden_channels, n_pred)

    def forward(self, x, edge_index, edge_attr,batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GINlayers(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred, n_layers):
        super(GINlayers, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_conv_model(n_f,hidden_channels))
        for i in range(0,n_layers):
            self.convs.append(self.build_conv_model(hidden_channels, hidden_channels))
        self.convs.append(self.build_conv_model(hidden_channels, n_hlin))
        self.lin = Linear(n_hlin, n_pred)
        self.num_layers = n_layers

    def build_conv_model(self,input_dim,hidden_dim):
        return GINConv(Seq(Linear(input_dim, hidden_dim),ReLU(), Linear(hidden_dim, hidden_dim)))

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class vanillaGCNglobal(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred):
        super(vanillaGCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(n_f, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, n_hlin)
        self.lin1 = Linear(n_hlin, n_hlin)
        self.lin2 = Linear(n_hlin, n_hlin)
        self.lin3 = Linear(n_hlin, n_pred)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)

        return x
class vanillaGCNglobaldo(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred):
        super(vanillaGCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(n_f, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, n_hlin)
        self.lin1 = Linear(n_hlin, n_hlin)
        self.lin2 = Linear(n_hlin, n_hlin)
        self.lin3 = Linear(n_hlin, n_pred)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)

        return x
class vanillaGCNsum(torch.nn.Module):
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred):
        super(vanillaGCNsum, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(n_f, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, n_hlin)
        self.lin = Linear(n_hlin, n_pred)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class vanillaGraphConv(torch.nn.Module):
    # Add skip connection as in Morris et al. 2018 to GNN layer to preserve central node information
    def __init__(self, n_f, hidden_channels,n_hlin, n_pred):
        super(vanillaGraphConv, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GraphConv(n_f, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, n_hlin)
        self.lin = Linear(n_hlin, n_pred)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
#https://github.com/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb
class GN(MessagePassing):
    def __init__(self, n_f, n_ef, msg_dim, npredict, hidden=300, aggr='add'):
        super(GN, self).__init__(aggr=aggr)
        self.msg_fnc = Seq(
            Lin(2 * n_f + n_ef, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )

        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, npredict)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e_ij=edge_attr)

    def message(self, x_i, x_j, e_ij):
        tmp = torch.cat([x_i, x_j, e_ij], dim=1)

        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)
class GNrelu(MessagePassing):
    def __init__(self, n_f, n_ef, msg_dim, npredict, hidden=300, aggr='add'):
        super(GNrelu, self).__init__(aggr=aggr)
        self.msg_fnc = Seq(
            Lin(2 * n_f + n_ef, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )

        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, npredict)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e_ij=edge_attr)

    def message(self, x_i, x_j, e_ij):
        tmp = torch.cat([x_i, x_j, e_ij], dim=1)

        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)
class GNswish(MessagePassing):
    def __init__(self, n_f, n_ef, msg_dim, npredict, hidden=300, aggr='add'):
        super(GNswish, self).__init__(aggr=aggr)
        self.msg_fnc = Seq(
            Lin(2 * n_f + n_ef, hidden),
            SiLU(),
            Lin(hidden, hidden),
            SiLU(),
            Lin(hidden, msg_dim)
        )

        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            SiLU(),
            Lin(hidden, hidden),
            SiLU(),
            Lin(hidden, npredict)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e_ij=edge_attr)

    def message(self, x_i, x_j, e_ij):
        tmp = torch.cat([x_i, x_j, e_ij], dim=1)

        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)
class GNpool(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpool,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GN(n_nf, n_ef, msg_dim, n_hidden)
        # self.GN2 = GN(n_hidden, n_ef, msg_dim, n_hidden)
        # self.GN3 = GN(n_hidden, n_ef, msg_dim, n_hidden)

        self.lin = Linear(n_hidden, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN2(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN3(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN(x, edge_index, edge_attr, batch)

        return out
class GNpoolswish(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpoolswish,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GNswish(n_nf, n_ef, msg_dim, n_hidden)
        self.lin = Linear(n_hidden, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN1(x, edge_index, edge_attr, batch)

        return out
class GNpoolMLP(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpoolMLP,self).__init__()

        # the message passing layer for the nodes
        #self.GN1 = GN(n_nf, n_ef, msg_dim, n_hidden)
        self.GN1 = GNrelu(n_nf, n_ef, msg_dim, n_hidden)
        # self.GN2 = GN(n_hidden, n_ef, msg_dim, n_hidden)
        # self.GN3 = GN(n_hidden, n_ef, msg_dim, n_hidden)

        #self.lin = Linear(n_hidden, n_predict)
        self.global_mlp = Seq(
            Lin(n_hidden, n_hidden),
            ReLU(),
            Lin(n_hidden, n_hidden),
            ReLU(),
            Lin(n_hidden,n_predict))

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN2(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN3(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        #out = self.lin(x)
        out = self.global_mlp(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN1(x, edge_index, edge_attr, batch)

        return out
class GNpoolsMLP(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpoolsMLP,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GNswish(n_nf, n_ef, msg_dim, n_hidden)
        self.global_mlp = Seq(
            Lin(n_hidden, n_hidden),
            SiLU(),
            Lin(n_hidden,n_predict))

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        #out = self.lin(x)
        out = self.global_mlp(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN1(x, edge_index, edge_attr, batch)

        return out
class GNpoolmax(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpoolmax,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GN(n_nf, n_ef, msg_dim, n_hidden)
        # self.GN2 = GN(n_hidden, n_ef, msg_dim, n_hidden)
        # self.GN3 = GN(n_hidden, n_ef, msg_dim, n_hidden)

        self.lin = Linear(n_hidden, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN2(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN3(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN(x, edge_index, edge_attr, batch)

        return out
class GNpoolsum(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpoolsum,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GN(n_nf, n_ef, msg_dim, n_hidden)
        # self.GN2 = GN(n_hidden, n_ef, msg_dim, n_hidden)
        # self.GN3 = GN(n_hidden, n_ef, msg_dim, n_hidden)

        self.lin = Linear(n_hidden, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN2(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN3(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN(x, edge_index, edge_attr, batch)

        return out
class GN2(MessagePassing):
    def __init__(self, n_f, n_ef, msg_dim, npredict, hidden=300, aggr='add'):
        super(GN2, self).__init__(aggr=aggr)
        self.msg_fnc = Seq(
            Lin(2 * n_f + n_ef, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )

        self.node_fnc = Seq(
            Lin(msg_dim, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, npredict)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e_ij=edge_attr)

    def message(self, x_i, x_j, e_ij):
        tmp = torch.cat([x_i, x_j, e_ij], dim=1)

        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):

        return self.node_fnc(aggr_out)
class GNpool2(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(GNpool2,self).__init__()

        # the message passing layer for the nodes
        self.GN1 = GN2(n_nf, n_ef, msg_dim, n_hidden)
        # self.GN2 = GN2(n_hidden, n_ef, msg_dim, n_hidden)
        # self.GN3 = GN2(n_hidden, n_ef, msg_dim, n_hidden)

        self.lin = Linear(n_hidden, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.GN1(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN2(x, edge_index, edge_attr, batch)
        # x = x.relu()
        # x = self.GN3(x, edge_index, edge_attr, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)

        return out
class GNNlayers(torch.nn.Module):
    def __init__(self, n_nf,n_ef,msg_dim,n_hidden,n_predict, n_layers):
        super(GNNlayers, self).__init__()
        #torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        # for the first layer
        self.convs.append(self.build_gnn_model(n_nf,n_ef,msg_dim,n_nf))
        for i in range(0,n_layers):
            self.convs.append(self.build_gnn_model(n_nf, n_ef,msg_dim,n_nf))
        self.convs.append(self.build_gnn_model(n_nf, n_ef,msg_dim,n_hidden))
        self.lin = Linear(n_hidden, n_predict)
        self.num_layers = n_layers

    def build_gnn_model(self,n_nf, n_ef, msg_dim, n_hidden):
        return GN(n_nf, n_ef, msg_dim, n_hidden)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        # print(x.size(1))

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr, batch)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GNclass(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict1,n_predict2):
        super(GNclass,self).__init__()

        # the message passing layer for the nodes
        self.GN = GN(n_nf, n_ef, msg_dim, n_hidden)

        self.global_mlp1 = Seq(
            Lin(n_hidden, n_hidden),
            ReLU(),
            Lin(n_hidden,n_predict1))

        # The model to predict g, Sij(theta),theta
        self.global_mlp2 = Seq(
            Lin(n_hidden, n_hidden),
            ReLU(),
            Lin(n_hidden,n_hidden))

        self.lin = Linear(n_hidden, n_predict2)

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.GN(x, edge_index, edge_attr, batch)

        # Sum up all the Qext_i and Qabs_i to get Qext, Qabs
        # Take in the encoded nodes - decode to sphere level predictions
        out1 = self.global_mlp1(out)

        # Sum up all the Qext_i and Qabs_i to get Qext, Qabs
        out1 = scatter_sum(out1, batch, dim=0)

        # Do global pooling to get asymmetry parameter and Sij elements
        out2 = self.global_mlp2(out)
        out2 = global_mean_pool(out2, batch)
        out2 = self.lin(out2)

        out = torch.cat([out1,out2],dim=1)
        return out
    def sphere_predict(self,x,edge_index, edge_attr, batch):
        # Get the sphere level predictions for Qext_i, Qabs_i
        out = self.GN(x, edge_index, edge_attr, batch)

        return self.global_mlp1(out)
class EGNNpool(torch.nn.Module):
    def __init__(self,n_nf,n_ef,msg_dim,n_hidden,n_predict):
        super(EGNNpool,self).__init__()

        # the message passing layer for the nodes
        self.EGNNlayer = EGNN(in_node_nf=n_nf-3, hidden_nf=msg_dim, out_node_nf=n_hidden, in_edge_nf=n_ef)
        self.lin = Linear(n_hidden+3, n_predict)

    def forward(self, x, edge_index, edge_attr, batch):

        h, c = self.EGNNlayer(x[:,0:2], x[:,2:5], edge_index, edge_attr)

        # 2. Readout layer
        h = global_mean_pool(h, batch)  # [batch_size, hidden_channels]
        c = global_mean_pool(c, batch)

        # 3. Apply a final classifier
        h = F.dropout(h, p=0.5, training=self.training)

        out = self.lin(torch.cat([c,h],dim=1))

        return out


