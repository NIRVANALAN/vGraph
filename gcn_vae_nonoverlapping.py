from __future__ import division, print_function

import argparse
import collections
import math
import re
import time
from subprocess import check_output
import pdb

import community
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from dgl.nn.pytorch import GraphConv
from dgl.transform import remove_self_loop
from sklearn.cluster import KMeans
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm, trange

from data_utils import load_cora_citeseer, load_webkb
from gcn.gcn import GCN
from score_utils import calc_nonoverlap_nmi

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='s', help="models used")
parser.add_argument('--lamda', type=float, default=.1, help="")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1001,
                    help='Number of epochs to train.')
parser.add_argument('--embedding-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='type of dataset.')
parser.add_argument("--gpu", type=int, default=7, help="gpu")
# * GCN
parser.add_argument("--n-layers", type=int, default=1,
                    help="number of hidden gcn layers")
parser.add_argument("--self-loop", action='store_true',
                    help="graph self-loop (default=False)")
# parser.add_argument('--task', type=str, default='community', help='type of dataset.')


def logging(args, epochs, nmi, modularity):
    with open('mynlog', 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('gcn_vae', args.dataset,
                                                   args.lr, args.embedding_dim, args.lamda, epochs, nmi, modularity))


def write_to_file(fpath, clist):
    with open(fpath, 'w') as f:
        for c in clist:
            f.write(' '.join(map(str, c)) + '\n')


def preprocess(fpath):
    clist = []
    with open(fpath, 'rb') as f:
        for line in f:
            tmp = re.split(b' |\t', line.strip())[1:]
            clist.append([x.decode('utf-8') for x in tmp])

    write_to_file(fpath, clist)


def get_assignment(G, model, num_classes=5, tpe=0, features=None):
    model.eval()
    edges = [(u, v) for u, v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1., features)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    import pdb; pdb.set_trace()
    res = res.argmax(axis=-1)
    assignment = {i: res[i] for i in range(res.shape[0])}
    return res, assignment


def classical_modularity_calculator(graph, embedding, model='gcn_vae', cluster_number=5):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """
    if model == 'gcn_vae':
        assignments = embedding
    else:
        kmeans = KMeans(n_clusters=cluster_number,
                        random_state=0, n_init=1).fit(embedding)
        assignments = {i: int(kmeans.labels_[i])
                       for i in range(0, embedding.shape[0])}

    modularity = community.modularity(assignments, graph)
    return modularity


def loss_function(recon_c, q_y, prior, c, norm=None, pos_weight=None):

    BCE = F.cross_entropy(recon_c, c, reduction='sum') / c.shape[0]
    KLD = F.kl_div(torch.log(prior), q_y, reduction='batchmean')
    return BCE, KLD


class GCNModelGumbel(nn.Module):
    def __init__(self, size, embedding_dim, categorical_dim, device, embedding_model):
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        self.community_embeddings = nn.Linear(
            embedding_dim, categorical_dim, bias=False).to(device)
        # self.node_embeddings_encoder = nn.Embedding(size, embedding_dim)
        self.node_embeddings_encoder = embedding_model
        # self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, size),
        ).to(device)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, temp, features):

        encoder_h = self.node_embeddings_encoder(features)
        w = encoder_h[w]
        c = encoder_h[c]
        # w = self.node_embeddings_encoder(w).to(self.device)
        # c = self.node_embeddings_encoder(c).to(self.device)

        q = self.community_embeddings(w*c)
        # q.shape: [batch_size, categorical_dim]
        # z = self._sample_discrete(q, temp)
        if self.training:
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True) #! hard estimator z is one-hot
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)
        # prior.shape [batch_num_nodes,

        # z.shape [batch_size, categorical_dim]
        import pdb; pdb.set_trace()
        new_z = torch.mm(z, self.community_embeddings.weight) #! estimated community membership embedding of each edge
        recon = self.decoder(new_z)

        return recon, F.softmax(q, dim=-1), prior


if __name__ == '__main__':
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.3
    ANNEAL_RATE = 0.00003

    # In[13]:
    if args.dataset in ['cora', 'citeseer']:
        G, adj, gt_membership = load_cora_citeseer(args.dataset)
    else:
        G, adj, gt_membership = load_webkb(args.dataset)
    # * LOAD GCN DECODER
    data = load_data(args)
    torch.cuda.set_device(args.gpu)
    features = torch.FloatTensor(data.features).cuda()
    device = features.device  # TODO
    in_feats = features.shape[1]
    n_classes = data.num_labels
    # graph preprocess and calculate normalization factor
    # add self loop
    g = DGLGraph(G)
    if args.self_loop:
        g = remove_self_loop(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    gcn_encoder = GCN(g,
                      in_feats,
                      args.embedding_dim,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout, output_projection=False).cuda()
    print(gcn_encoder)
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    categorical_dim = len(set(gt_membership))
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)

    model = GCNModelGumbel(
        adj.shape[0], embedding_dim, categorical_dim, device, gcn_encoder)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_edges = [(u, v) for u, v in G.edges()]
    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))
    with trange(epochs, desc='gcn_vae') as t:
        for epoch in t:

            start = time.time()
            batch = torch.LongTensor(train_edges)
            assert batch.shape == (len(train_edges), 2)

            model.train()
            optimizer.zero_grad()

            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))
            recon, q, prior = model(w, c, temp, features)

            res = torch.zeros([n_nodes, categorical_dim],
                              dtype=torch.float32).to(device)
            for idx, e in enumerate(train_edges):
                res[e[0], :] += q[idx, :]
                res[e[1], :] += q[idx, :]
            smoothing_loss = args.lamda * ((res[w] - res[c])**2).mean()

            bce, kld = loss_function(recon, q, prior, c.to(device), None, None)
            t.set_postfix(bce=bce.item(), kld=kld.item(),
                          smoothing=smoothing_loss.item())
            loss = bce+kld+smoothing_loss
            # print(f'loss: {loss}')

            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            if epoch % 100 == 0:
                temp = np.maximum(temp*np.exp(-ANNEAL_RATE*epoch), temp_min)

                model.eval()

                membership, assignment = get_assignment(
                    G, model, categorical_dim, features=features)
                #print([(membership == i).sum() for i in range(categorical_dim)])
                #print([(np.array(gt_membership) == i).sum() for i in range(categorical_dim)])
                modularity = classical_modularity_calculator(G, assignment)
                nmi = calc_nonoverlap_nmi(membership.tolist(), gt_membership)

                print(epoch, nmi, modularity)
                logging(args, epoch, nmi, modularity)

    print(f"Optimization Finished in {time.time() - start}")
