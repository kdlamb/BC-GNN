#
# trainGCN.py - Train GCN on MSTM data sets
#
# positional arguments:
#   expname               File extension for saving current experiment outputs.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --testsubset          Use only a small portion of the data.
#   --loaddata LOADDATA   File extension for training/test data sets.
#   --inference           Perform inference on train/test data sets.
#   --loadmodelparams LOADMODELPARAMS
#                         Load previously trained model parameters from log
#                         directory.
#   -pp {0,1}, --preprocess {0,1}
#                         Options for preprocessing optical constants (default = 0)
#   -nm NMAX, --nmax NMAX
#                         Maximum aggregate size to use in test/train (default = 1000).
#   -ng NGRAPHS, --ngraphs NGRAPHS
#                        Total number of aggregates (default = 15000).
#   -bs BATCHSIZE, --batchsize BATCHSIZE
#                         Batch size of training data sets (default = 20.0).
#   -c CUTOFF, --cutoff CUTOFF
#                         Cutoff for nodes to be connected (default = 1.0).
#   --edges               Include Edge Attributes in Graph Data (default =
#                         False).
#   --nhidden NHIDDEN     Number of hidden layers (default = 100).
#   --nhlin NHLIN         Number of hidden layers for final layer (default = 100).
#   --s11                 Include the S11 matrix as a target.
#   -fc, --fullyconnected
#                         Add a node that is fully connected to all the other
#                         nodes
#   --model {vanillaGCN,vanillaGCNsum,vanillaGCNglobal,vanillaGCNglobaldo,vanillaGraphConv,GNpool,GNclass,GCNlayers}
#                       Which model to use (default = vanillaGCN).
#
#   -a ALPHA, --alpha ALPHA Relative factor between MSE loss for Opt. Properties and S11.
#
#   --n_layers N_LAYERS   For models with a variable number of layers how many layers to use.
#
#    -pc {0,1}, --constraints {0,1}
#                         Which physical constraints to impose on output
#                         (default = 0).
#   -e EPOCHS, --epochs EPOCHS
#                         Number of epochs to train model over (default = 50).
#   -lr INIT_LR, --init_lr INIT_LR
#                         Initial learning rate for optimizer (default = 1e-4).
#   --mincriteria {train, validation} Use train or validation loss as criteria
#                         to save best model(default=train).
#   --nosavefigures       Don't save model predictions for test/train data sets
#   --nosavemodel         Don't save the trained model for the current
#                         experiment to the log folder.

import os
import re
import math
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import subprocess
from scipy import integrate

import torch
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm

from models import GCNlayers, WGCNlayers
from models import GNpool,GNpool2, GNclass, GINlayers, GNNlayers, SGCN,vanillaGCN
from models import GCNlayersmax, GCNlayerssum, GNpoolmax,GNpoolsum, GNpoolMLP, GNpoolswish, GNpoolsMLP
from models import EGNNpool

from plotresults import ploteffs, plotS11, plotsij
import mstm_output_reader as mstmfast

def calc_distance_metric(sp_mat,cutoff,fullyconnected=False,ba=False,scaledistance=False):
    # for the sphere_position array of Ns, calculate an Ns x Ns matrix of relative distances
    ## ba = characteristic length scale for scale free network
    Ns = sp_mat.shape[0]
    rel_dist = np.zeros([Ns, Ns])
    rel_dist_edges = np.zeros([Ns,Ns,1])
    Rg = 0
    charlength = np.log(Ns)
    if ba:
        charlength = np.log(Ns)/(np.log(np.log(Ns)))
    for i in range(Ns):
        xi = sp_mat[i, 1]
        yi = sp_mat[i, 2]
        zi = sp_mat[i, 3]
        for j in range(Ns):
            xj = sp_mat[j, 1]
            yj = sp_mat[j, 2]
            zj = sp_mat[j, 3]

            dist = ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) ** (0.5)
            if dist < (cutoff*charlength):
                rel_dist[i, j] = dist
                if scaledistance:
                    rel_dist[i, j] = dist/charlength
                #rel_dist_edges[i,j,0] = dist
                # rel_dist_edges[i,j,1] = xi - xj
                # rel_dist_edges[i,j,2] = yi - yj
                # rel_dist_edges[i,j,3] = zi - zj
                #
                # rel_dist_edges[i,j,4] = xj - xi
                # rel_dist_edges[i,j,5] = yj - yi
                # rel_dist_edges[i,j,6] = zj - zi
            Rg = Rg + (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
    # normalize the matrix

    connected = (rel_dist > 0)
    # adds a fully connected node to the graph an avg. dist away from all the other nodes
    if fullyconnected==True:
        avgdist = np.mean(rel_dist)
        col = np.ones(Ns)*avgdist
        row = np.ones(Ns+1)*avgdist
        rel_dist = np.column_stack((rel_dist,col))
        rel_dist = np.row_stack((rel_dist,row))
        rel_dist[Ns,Ns]=0

        sp_mat_updated = np.zeros((Ns+1,4))
        sp_mat_updated[0:-1,:]=sp_mat
        sp_mat_updated[-1:,0]=sp_mat_updated[0,0]
        sp_mat_updated[-1:,1:4]=(avgdist)**(1/3)
    else:
        sp_mat_updated = sp_mat

    return sp_mat_updated, rel_dist, charlength

def mstm_aggparams(dirname, filename, newformat=False):
    # just get the aggregate information from the filename without loading any data
    fn = os.path.join(dirname, filename)
    # print(fn)
    if (newformat == False):
        m = re.match("a(.*)_N(.*)_R(.*)_Df(.*).out", filename)
        if m == None:
            print(dirname, filename)
        values = m.group(1), m.group(2), m.group(3), m.group(4)
        Xv = 1.0
    else:
        m = re.match("a(.*)N(.*)R(.*)Df(.*)x(.*).out", filename)
        if m == None:
            print(dirname, filename)
        values = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        Xv = float(m.group(5)) / 10.0

    # print(values)
    agg = float(m.group(1))
    Ns = float(m.group(2))
    R = float(m.group(3)) / 10.0
    Df = float(m.group(4))

    mstmout = np.zeros([5])
    mstmout[0] = agg
    mstmout[1] = Ns
    mstmout[2] = R
    mstmout[3] = Df
    mstmout[4] = Xv

    return mstmout
def mstm_loader_fast(dirname,filename, newformat=False):
    # A slightly faster data loader
    aggparams=mstm_aggparams(dirname, filename, newformat=newformat)
    Ns = aggparams[1].astype(int)
    fn=os.path.join(dirname,filename)

    mstmout,scatmatrix,spherematrix=mstmfast.load_mstm_fast(fn,Ns)

    mstmoutall = np.concatenate((aggparams,mstmout),axis=0)

    return mstmoutall,scatmatrix.reshape(181,11),spherematrix.reshape(Ns,10)
# Test data loader
def dense_to_sparse_ref(reftensor,tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes of reftensor

    Args:
        reftensor (Tensor): The dense adjaceny matrix of the reference tensor
        tensor (Tensor): The dense adjacency matrix of the target tensor
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert reftensor.dim() == 2
    assert tensor.dim() == 2
    index = reftensor.nonzero(as_tuple=False).t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value
def getaggdata(filename,dirname,format,cutoff,pp=0,edges=False,fullyconnected=False,ba=False,sij=-1,scaledistance=False):
    # get the aggregate data from the file and return the graph data for that aggregate
    fn = filename
    dr = dirname
    fm = format

    # mstmout, scatmatrix, spheremat
    # check which format the file is
    if (fm==0):
        qq, st, sm =mstm_loader_fast(dr,fn, newformat=False)
    else:
        qq, st, sm =mstm_loader_fast(dr,fn, newformat=True)

    qs = sm[:,0:4]



    # node features

    # Check if it is good data or not
    if (qq[5]>-0.5):
        #print(qq[1], qq[2], qq[3], num_edges)

        # Total morphological parameters for the aggregates:
        # Ns, Df, kf, Xv, Re(nk)
        graph_feat = np.zeros(5)
        graph_feat[0] = qq[1]
        graph_feat[1] = qq[3]
        graph_feat[2] = 1.2
        graph_feat[3] = qq[4]
        graph_feat[4] = qq[2]

        # Scale cutoff by size parameter to get consistent number of edges per aggregate
        scaled_cutoff = cutoff*qq[4]

        #sp_mat_updated, rel_dist, charlength
        qs, ss, l0 = calc_distance_metric(qs, scaled_cutoff,fullyconnected=fullyconnected,ba=ba,scaledistance=scaledistance)

        num_nodes = qs.shape[0]
        node_feat = np.zeros((num_nodes, 5))
        pos_feat = np.zeros((num_nodes,3))
        sphere_tar = np.zeros((num_nodes,4))



        fractal = torch.tensor(ss[:, :], dtype=torch.float)

        Fsparse = dense_to_sparse(fractal)
        # Create feature matrix for edges (num_edges, edge_ID)
        num_edges = len(Fsparse[1])
        X = Fsparse[1].numpy()

        # edge_attr
        #edge_index = torch.tensor(Fsparse[0], dtype=torch.long)
        edge_index = Fsparse[0].clone().detach()
        #edge_attr = torch.tensor(Fsparse[1], dtype=torch.float)
        if edges:
            edgedata_list = []

            edgedata=torch.tensor(ss[:,:],dtype=torch.float)
            #edgedata = ss.clone().detach()
            edgetensor=dense_to_sparse_ref(fractal,edgedata)
            edgedata_list.append(edgetensor[1])

            stacked_edgedata = torch.transpose(torch.stack(edgedata_list),0,1)
            #edge_attr = torch.tensor(stacked_edgedata, dtype=torch.float)
            edge_attr = stacked_edgedata.clone().detach()


        # global features - agg, Ns, R, Df, Xv (qq[0:5])
        #glob_attr = torch.tensor(qq[1:5], dtype=torch.float)
        for kk in range(num_nodes):
            # node_feat[kk, 0:3] = qs[kk, 1:4]
            # node_feat[kk, 4:5] = qq[2:3]
            pos_feat[kk,0:3] = qs[kk,1:4]
            node_feat[kk, 0:4] = qs[kk, 0:4]
            node_feat[kk, 4:5] = qq[2]

            if kk>(sm.shape[0]-1):
                sphere_tar[kk,0:4] = 0
            else:
                sphere_tar[kk,0:4] = sm[kk,6:]

        node_features = torch.tensor(node_feat, dtype=torch.float)
        pos_features = torch.tensor(pos_feat, dtype=torch.float)
        sphere_targets = torch.tensor(sphere_tar, dtype=torch.float)
        #s11 = st['11'].values
        theta = np.arange(0,181,1)
        theta_rad = theta*math.pi/180.0
        s11 = st[:,1]
        stall = torch.tensor(st)
        # preprocessing options
        if pp < 1:
            # scale scattering, abs., ext. by 10.0
            qext = qq[5]
            qabs = qq[6]
            qscat = qq[7]
            qq[5]=qext
            qq[6]=qabs
            qq[7]=qscat

        if pp == 1:
            qext = qq[5]
            qabs = qq[6]
            qscat = qq[7]

            qq[5]=qext/qext
            qq[6]=qabs/qext
            qq[7]=qscat/qext

            s11 = 0.5*s11*np.sin(theta_rad)

        # targets - total ext, abs, scat efficiencies,asym
        # If preds11=True s11 is a target

        n_p = 4
        targets = np.zeros(n_p + len(s11))
        targets[0:4] = qq[5:9]
        targets[n_p:] = s11

        if (sij>0):
                # unnormalize the other matrix elements
            nangles = sij
            angles = np.sort(np.random.choice(theta,size=nangles,replace=False))
            stall = torch.tensor(st)
            st[:,1] = st[:,1]/st[0,1]
            targets = np.zeros(n_p + nangles*11)
                #targets[0:4] = qq[5:9]
            for i in range(nangles):
                sijtargs=np.append(theta_rad[angles[i]],st[angles[i],1:])
                targets[4+i*11:(4+(i+1)*11)]=sijtargs

        y = torch.tensor(targets, dtype=torch.float)
        g_features = torch.tensor(graph_feat, dtype=torch.float)

        #                 aggdata = Data(Variable(node_features), edge_index=edge_index, edge_attr=edge_attr,
        #                                glob_attr=glob_attr, y=Variable(y))
        if edges:
            #if (sij>0)or(preds11):
            aggdata = Data(Variable(node_features), edge_index=edge_index, edge_attr=edge_attr, y=Variable(y),
                            node_targets=Variable(sphere_targets),pos=Variable(pos_features), g_attr=Variable(g_features),
                            stokes=Variable(stall))
            #else:
            #    aggdata = Data(Variable(node_features), edge_index=edge_index, edge_attr=edge_attr, y=Variable(y),
            #                   node_targets=Variable(sphere_targets), pos=Variable(pos_features), g_attr=Variable(g_features))
        else:
            #if (sij>0)or(preds11):
            aggdata = Data(Variable(node_features), edge_index=edge_index, y=Variable(y),
                            node_targets=Variable(sphere_targets), pos=Variable(pos_features), g_attr=Variable(g_features),
                            stokes=Variable(stall))
            #else:
            #    aggdata = Data(Variable(node_features), edge_index=edge_index, y=Variable(y),
            #                   node_targets=Variable(sphere_targets), g_attr=Variable(g_features), pos=Variable(pos_features))

    return aggdata

def graphdata(cutoff, traindir, testdir,bs=20,ngraphs=15000,maxNs=3000,pp=0,edges=False,
              fullyconnected=False,ba=False,sij=-1,scaledistance=False):
    # return graph data sets with the given cutoff radius
    # traindir = directory with the training/validation data set
    # testdir = directory with test data set
    # pp = preprocessing options {0, 1}
    # edges = add edge attribute (the distance between neighboring nodes)
    # fullyconnected = add one fully connected node
    # sij = const. = # of angles to predict Sij matrix at.
    trainfiles = []
    trainformat = []
    traindirs = []

    testfiles = []
    testformat = []
    testdirs = []

    for f in listdir(traindir):
        currfile=join(traindir,f)
        if isfile(currfile):
            # Make sure its a good file
            traindirs.append(traindir)
            # just the file name for the data loader
            trainfiles.append(f)
            # This just checks which format the file name has
            if re.match("a(.*)_N(.*)_R(.*)_Df(.*).out",f) is not None:
                trainformat.append(0)
            else:
                trainformat.append(1)
    print("Number of files in training dir:")
    print(len(trainfiles))

    for f in listdir(testdir):
        currfile=join(testdir,f)
        if isfile(currfile):
            # Make sure its a good file
            testdirs.append(testdir)
            # just the file name for the data loader
            testfiles.append(f)
            # This just checks which format the file name has
            if re.match("a(.*)_N(.*)_R(.*)_Df(.*).out",f) is not None:
                testformat.append(0)
            else:
                testformat.append(1)
    print("Number of files in test dir")
    print(len(testfiles))

    allNs = []
    allidx = []

    i0 = 0
    # Use only a selection of the training data set
    idxarray = np.arange(len(trainfiles))
    nchose = min([len(trainfiles),ngraphs])
    seed = np.random.seed(123)
    chosen = np.random.choice(idxarray.shape[0],nchose,replace=False)

    for ii in range(len(chosen)):
        fn = trainfiles[chosen[ii]]
        dr = traindirs[chosen[ii]]
        fm = trainformat[chosen[ii]]
        # print(fn)
        if (fm==0):
            #qq,_=mstm_random_loader(dr,fn,get_sphere_pos=False,newformat=False)
            qq=mstm_aggparams(dr,fn, newformat=False)
        else:
            #print(dr,fn)
            #qq,_=mstm_random_loader(dr,fn,get_sphere_pos=False,newformat=True)
            qq=mstm_aggparams(dr,fn, newformat=True)
        #print(qq)
        # Limit to aggregates smaller than maxNs
        if qq[1]<maxNs:
            allNs.append(qq[1])
            allidx.append(chosen[ii])

    fractalarray = np.array(allNs)
    chosen = np.array(allidx)

    # separate into test and training sets - retain largest quarter for test sets
    print("Number of aggregates:", len(allNs))
    print("Max. Agg. in Train and Val. Set, Max. Agg.:")
    print(maxNs)

    # Create training, validation, and test graph data lists
    traindata_list = []
    valdata_list = []
    testdata_list = []

    # Create the zero shot validation, and test graph data lists
    vallargedata_list = []
    testlargedata_list = []

    vv=0
    print("Creating Train, Val., and Test Data Sets:")
    for ii in range(len(chosen)):
        fn = trainfiles[chosen[ii]]
        dr = traindirs[chosen[ii]]
        fm = trainformat[chosen[ii]]

        # Get the aggregate graph data
        aggdata=getaggdata(fn,dr,fm,cutoff,pp=pp,edges=edges,fullyconnected=fullyconnected,ba=ba,sij=sij,scaledistance=scaledistance)
        # Training set
        if (ii%500 == 0):
            print(ii)

        if (len(aggdata.edge_index[0])>0):
            if (qq[1] < maxNs):
                if (vv%4 < 2):
                    traindata_list.append(aggdata)
                elif (vv%4 == 3):
                    valdata_list.append(aggdata)
                else:
                    testdata_list.append(aggdata)
                vv = vv +1
    print("Creating Large Test and Large Val. Data Set:")
    vv = 0
    for ii in range(0,len(testfiles)):
        # Get the aggregate graph data
        fn = testfiles[ii]
        dr = testdirs[ii]
        fm = testformat[ii]

        # Get the aggregate graph data
        aggdata=getaggdata(fn,dr,fm,cutoff,pp=pp,edges=edges,fullyconnected=fullyconnected,ba=ba,sij=sij,scaledistance=scaledistance)

        if (ii%100 == 0):
            print(ii)
        if (len(aggdata.edge_index[0])>0):
            # Test set
            if (vv%2==0):
                testlargedata_list.append(aggdata)
            else:
                vallargedata_list.append(aggdata)
            vv = vv +1

    trainloader = DataLoader(traindata_list, batch_size=bs, shuffle=True)
    valloader = DataLoader(valdata_list, batch_size=bs, shuffle=True)
    testloader = DataLoader(testdata_list, batch_size=bs, shuffle=True)
    testlargeloader = DataLoader(testlargedata_list, batch_size=bs, shuffle=True)
    vallargeloader = DataLoader(vallargedata_list,batch_size=bs,shuffle=True)

    return trainloader, valloader, testloader, vallargeloader,testlargeloader

def train(modelname,train_loader,constraints,device,alpha,predint=True,preds11=False,predsij=False):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # xx=data.x
        # print(xx.shape)
        # yy=data.y
        # print(yy.shape)
        data.to(device)
        if (modelname[0:2] == "GN" or modelname[0] == "E" ):
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        elif (modelname[0] == "S" or modelname[0] == "W"):
            edge_att = torch.squeeze(data.edge_attr)
            out = model(data.x, data.edge_index, edge_att, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        if constraints == 1:
                # Impose constraint that C_ext = C_scat+C_abs
                # y: C_ext.,C_abs., C_scat., Asym. parm.
            out[:, 0] = out[:, 1] + out[:, 2]
        if constraints == 2:
            theta = np.arange(0,181,1)
            theta_rad = theta*math.pi/180.0
            s11 = out[:,4:].detach().cpu().numpy()
            s11_int = integrate.trapz(s11*np.cos(theta_rad)*np.sin(theta_rad),theta_rad)/2.0
            out[:,3] = torch.tensor(s11_int)

        #loss = criterion(out.view(-1), data.y)  # Compute the loss.
        ntargets = 185
        if predsij:
            ntargets = 4+11*181
        real = data.y.view(-1, ntargets)
        opty = real[:, 0:4]
        ntargets = out.shape[1]
        n_p = 0
        loss = 0
        if predint:
            n_p = 4
            optout = out[:, 0:4]
            loss = criterion(optout, opty)
        if preds11 or predsij:
            s11out = out[:, n_p:]
            s11y = real[:, 4:]
            loss = loss + alpha * criterion(s11out, s11y)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        sched.step()
        optimizer.zero_grad()  # Clear gradients.
    return model

def test(modelname,loader,constraints,device,alpha,predint=True,preds11=False,predsij=False):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            if (modelname[0:2] == "GN" or modelname[0] == "E" ):
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            elif (modelname[0] == "S" or modelname[0] == "W"):
                edge_att = torch.squeeze(data.edge_attr)
                out = model(data.x, data.edge_index, edge_att, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            if constraints == 1:
                # Impose constraint that C_ext = C_scat+C_abs
                # y: C_ext.,C_abs., C_scat., Asym. parm.
                out[:, 0] = out[:, 1] + out[:, 2]
            if constraints == 2:
                theta = np.arange(0,181,1)
                theta_rad = theta*math.pi/180.0
                s11 = out[:,4:].detach().cpu().numpy()
                s11_int = integrate.trapz(s11*np.cos(theta_rad)*np.sin(theta_rad),theta_rad)/2.0
                out[:,3] = torch.tensor(s11_int)

            ntargets = 185
            if predsij:
                ntargets = 4+11*181
            real = data.y.view(-1,ntargets)
            opty = real[:,0:4]

            n_p = 0
            loss = 0
            if predint:
                n_p = 4
                optout = out[:, 0:4]
                loss = criterion(optout, opty)
            if (preds11 or predsij):
                s11out = out[:, n_p:]
                s11y = real[:, 4:]
                loss = loss + alpha * criterion(s11out, s11y)

            correct = correct + loss
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def inference(modelname,model,loader,constraints,device,predint=True,preds11=False,predsij=False):
    model.eval()
    dataset = loader.dataset
    for data in dataset:
        # Figure out how many targets there are
        ntargets=data.y.numpy().shape
        ntargets=ntargets[0]
        break
    if predint:
        nstart = 0
        nend = 4
    if preds11:
        nstart = 4
        nend = 185
    if predsij:
        nstart = 4
        nend = 1995
    if (preds11 and predint):
        nstart = 0
        nend = 185

    pred = np.zeros((len(dataset),ntargets))
    real = np.zeros((len(dataset),ntargets))
    jj=0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        if (modelname[0:2] == "GN" or modelname[0] == "E" ):
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        elif (modelname[0] == "S" or modelname[0] == "W"):
            edge_att = torch.squeeze(data.edge_attr)
            out = model(data.x, data.edge_index, edge_att, data.batch)
        else:
            out = model(data.x, data.edge_index,data.batch)
        bs=out.size(0)
        nt=out.size(1)
        #print(out.size(0))
        if out is not(None):
            if constraints == 1:
                # Impose constraint that C_ext = C_scat+C_abs
                # y: C_ext.,C_abs., C_scat., Asym. parm.
                out[:, 0] = out[:, 1] + out[:, 2]
            if constraints == 2:
                theta = np.arange(0,181,1)
                theta_rad = theta*math.pi/180.0
                s11 = out[:,4:].detach().cpu().numpy()
                s11_int = integrate.trapz(s11*np.cos(theta_rad)*np.sin(theta_rad),theta_rad)/2.0
                out[:,3] = torch.tensor(s11_int)
            real[jj:jj+bs,:]=data.y.cpu().numpy().reshape(bs,ntargets)
            pred[jj:jj+bs,nstart:nend]=out.cpu().detach().numpy().reshape(bs,nt)
            jj=jj+bs
        else:
            break
    return real,pred
if __name__ == "__main__":

    # The directory where the mstm output files (random orientation) are stored.
    traindir = "/burg/glab/users/kl3231/Projects/fractal-gnn/fractal-gnn/datasets/data/train"
    testdir = "/burg/glab/users/kl3231/Projects/fractal-gnn/fractal-gnn/datasets/data/balancedtest"
    # The directory where the trained model will be saved.
    modeldirname = "/burg/glab/users/kl3231/Projects/fractal-gnn/fractal-gnn/log"
    datasetdirname = "/burg/glab/users/kl3231/Projects/fractal-gnn/fractal-gnn/datasets"

    print("Starting...")
    parser = argparse.ArgumentParser(description="Train GCN on MSTM data sets")
    parser.add_argument("expname",type=str,help="File extension for saving current experiment outputs.")
    parser.add_argument("--testsubset",help="Use only a small portion of the data.", action="store_true")

    parser.add_argument("--loaddata",type=str,help="File extension for training/test data sets.")
    parser.add_argument("--inference", help="Perform inference on train/test data sets.", action="store_true")
    parser.add_argument("--loadmodelparams",type=str, help="Load previously trained model parameters from log directory.")

    # Graph Data parameters
    parser.add_argument("-pp","--preprocess",type=int,help="Options for preprocessing optical constants (default = 0)",
                        default=0, choices=[0,1])
    parser.add_argument("-nm","--nmax", type=int, help="Maximum aggregate size to use in train (default = 1000).",
                        default=1000)
    parser.add_argument("-ng","--ngraphs", type=int, help="Total number of aggregates (default = 15000).",
                        default=15000)
    parser.add_argument("-bs","--batchsize", type=int, help="Batch size of training data sets (default = 20).",
                        default=20)
    parser.add_argument("-c","--cutoff", type=float, help="Cutoff for nodes to be connected (default = 1.0).",
                        default=1.0)
    parser.add_argument("--edges", help="Include Edge Attributes in Graph Data (default = False).", action="store_true")
    parser.add_argument("-sd","--scaledistance",help="Scale the distance relative to the characteristic length (default = False).", action="store_true")
    parser.add_argument("--int",help = "Predict the Integral optical properties as targets.",action="store_true")
    parser.add_argument("--s11",help="Include the S11 matrix as a target.",action="store_true")
    parser.add_argument("-fc","--fullyconnected",help="Add a node that is fully connected to all the other nodes",action="store_true")
    parser.add_argument("--ba",help="Use Barabasi-Albert characteristic length",action="store_true")
    parser.add_argument("--sij",type=int,help="Number of angles of Sij matrix to predict (default = -1).", default=-1)

    # Model parameters
    parser.add_argument("--nhidden",type=int,help="Number of hidden layers (default = 100).",
                        default=100)
    parser.add_argument("--nhlin",type=int,help="Number of hidden layers for final layer (default = 100).",
                        default=100)
    parser.add_argument("--model",type=str,help="Which model to use (default = vanillaGCN).", default="vanillaGCN",
                        choices = ['vanillaGCN','vanillaGCNsum','vanillaGCNglobal','vanillaGCNglobaldo','vanillaGraphConv','GNpool','GNpool2',
                            'GNclass','GCNlayers','GINlayers','GNNlayers','SGCN','WGCNlayers','GNpoolsMLP','EGNNpool',
                            'GCNlayersmax', 'GCNlayerssum', 'GNpoolmax','GNpoolsum','GNpoolMLP','GNpoolswish'])
    parser.add_argument("-a","--alpha",type=float,help="Relative factor between MSE loss for Opt. Properties and S11.", default=1.0)
    parser.add_argument("--n_layers",type=int,help="For models with a variable number of layers how many layers to use.",default=3)
    parser.add_argument("--dropout",type=float,help='Dropout for models with dropout.',default=0.5)

    # Output parameters
    parser.add_argument("-pc","--constraints",type=int,help="Which physical constraints to impose on output (default = 0).", default=0,
                        choices = [0,1,2])
    #parser.add_argument("--stokes",help="Predict the first element of the Stokes matrix S(11) as output of the model", action="store_true")

    # Training parameters
    parser.add_argument("-e","--epochs", type=int, help="Number of epochs to train model over (default = 50).",
                        default=50)
    parser.add_argument("-lr","--init_lr",type=float, help="Initial learning rate for optimizer (default = 1e-4).",
                        default=1e-4)
    parser.add_argument("-mlr","--max_lr",type=float, help="Max. learning rate for optimizer (default = 1e-4).",
                        default=1e-4)
    parser.add_argument("--mincriteria",type=str, help="Use train or validation loss as criteria to save best model (default = train).",
                        default="train",choices=["train","validation"])
    parser.add_argument("--noval",help="No validation data set (use the test data set)", action="store_true")

    # Output options - save figures and model by default
    parser.add_argument("--nosavefigures",help="Don't save model predictions for test/train data sets",action="store_true")
    parser.add_argument("--nosavemodel",help="Don't save the trained model for the current experiment to the log folder.",
                        action="store_true")

    print("Compiling args parser")
    args = parser.parse_args()
    # output - file names
    currexpname = args.expname
    print("Current experiment name: ")
    print(currexpname)

    # training and test data set names
    trainname = "trainload_"+currexpname+".pth"
    valname = "valload_"+currexpname+".pth"
    testname = "testload_"+currexpname+".pth"
    testlargename = "testlargeload_"+currexpname+".pth"
    vallargename = "vallargeload_"+currexpname+".pth"
    # train

    # Model evaluation figure names
    trainfigtemplate = "TrainTestAcc_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"

    resultstemplate = "Effs_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"

    extfigtemplate = "QExt_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    absfigtemplate = "QAbs_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    scatfigtemplate = "QScat_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    asymfigtemplate = "Asym_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    stokesfigtemplate = "S11_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    stokespredfigtemplate = "S11pred_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    sijfigtemplate = "Sij_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"
    sijpredfigtemplate = "Sijpred_ep{}_lr{}ne{}_nh{}_nhl{}_{}.png"


    #Current trained model parameters filename template
    currmodeltemplate = "{}_ep{}_lr{}ne{}_nh{}_nhl{}_{}"

    # Use only a small subset of the data for testing
    if (args.testsubset):
        outdirs=["out_new9"]
        outdirs2=[] #"out3"]

    #create training and test sets
    if (args.loaddata):
        print("Loading the data sets ")
        loadexp = args.loaddata
        print(loadexp)
        trainname = "trainload_" + loadexp + ".pth"
        valname = "valload_" + loadexp + ".pth"
        vallargename = "vallargeload_" + loadexp + ".pth"
        if args.noval:
            valname = "testload_" + loadexp + ".pth"
            vallargename = "testload_" + loadexp + ".pth"

        testname = "testload_" + loadexp + ".pth"
        testlargename = "testlargeload_" + loadexp + ".pth"
        print("Training Data: ", trainname)
        print("Val. Data: ", valname)
        print("Val. Large Data: ", vallargename)
        print("Test Data: ", testname)
        print("Test Large Data: ", testlargename)

        trainfn = os.path.join(datasetdirname,trainname)
        valfn = os.path.join(datasetdirname,valname)
        vallargefn = os.path.join(datasetdirname,vallargename)
        testfn = os.path.join(datasetdirname,testname)
        testlargefn = os.path.join(datasetdirname,testlargename)

        trainload = torch.load(trainfn)
        valload = torch.load(valfn)
        vallargeload = torch.load(vallargefn)
        testload = torch.load(testfn)
        testlargeload = torch.load(testlargefn)
    else:
        print("Making the datasets.")
        trainload, valload, testload, vallargeload, testlargeload = graphdata(args.cutoff, traindir, testdir,bs=args.batchsize,
                                        ngraphs=args.ngraphs,maxNs=args.nmax,pp=args.preprocess,edges=args.edges,
                                        fullyconnected=args.fullyconnected,ba=args.ba,sij=args.sij,scaledistance=args.scaledistance)
        trainfn = os.path.join(datasetdirname,trainname)
        valfn = os.path.join(datasetdirname,valname)
        testfn = os.path.join(datasetdirname,testname)
        vallargefn = os.path.join(datasetdirname,vallargename)
        testlargefn = os.path.join(datasetdirname,testlargename)

        torch.save(trainload, trainfn)
        torch.save(valload, valfn)
        torch.save(testload,testfn)
        torch.save(vallargeload,vallargefn)
        torch.save(testlargeload, testlargefn)
    print("Train/Val/Test sizes, Val Large, Test Large:")
    print(len(trainload),len(valload),len(testload),len(vallargeload),len(testlargeload))

    n_f = 5
    n_ef = 1
    n_h = args.nhidden
    n_hlin = args.nhlin

    n_p = 0
    n_p1 = 0
    n_p2 = 0
    predsij = False

    if args.int:
        n_p = 4
        n_p1 = 3
        n_p2 = 1
    if args.s11:
        n_p = n_p+181
        n_p2 = n_p2+181
    elif (args.sij>0):
        n_p = n_p + args.sij*11
        n_p2 = n_p2 + args.sij*11
        predsij = True

    init_lr = args.init_lr
    max_lr = args.max_lr
    weight_decay = 1e-8
    total_epochs = args.epochs

    # Learning rate strings so its in the correct format for the filenames
    lrval='{:3.0e}'.format(init_lr)[0]
    lrexp = '{:3.0e}'.format(init_lr)[4]

    # Model evaluation figure names
    trainfigname = trainfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h),str(n_hlin), currexpname)

    effsfigname = resultstemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    extfigname = extfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    absfigname = absfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    scatfigname = scatfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    asymfigname = asymfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    stokesfigname = stokesfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    stokespredfigname = stokespredfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    sijfigname = sijfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
    sijpredfigname = sijpredfigtemplate.format(str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)

    # update this if I try some other cases
    print(args.model)
    # if args.model == "vanillaGraphConv":
    #     model = vanillaGraphConv(n_f, n_h, n_hlin, n_p)
    # elif args.model == "vanillaGCNsum":
    #     model = vanillaGCNsum(n_f, n_h, n_hlin, n_p)

    if args.model == "GNpool":
        model = GNpool(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpoolswish":
        model = GNpoolswish(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpoolsum":
        model = GNpoolsum(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpoolmax":
        model = GNpoolmax(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpoolMLP":
        model = GNpoolMLP(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpoolsMLP":
        model = GNpoolsMLP(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNpool2":
        model = GNpool2(n_f,n_ef,n_h,n_hlin,n_p)
    elif args.model == "GNNlayers":
        model = GNNlayers(n_f,n_ef,n_h,n_hlin,n_p,args.n_layers)
    elif args.model == "GNclass":
        model = GNclass(n_f,n_ef,n_h,n_hlin,n_p1,n_p2)
    elif args.model == "GCNlayers":
        model = GCNlayers(n_f, n_h, n_hlin, n_p, args.n_layers)
    elif args.model == "GCNlayerssum":
        model = GCNlayerssum(n_f, n_h, n_hlin, n_p, args.n_layers)
    elif args.model == "GCNlayersmax":
        model = GCNlayersmax(n_f, n_h, n_hlin, n_p, args.n_layers)
    elif args.model == "WGCNlayers":
        model = WGCNlayers(n_f, n_h, n_hlin, n_p, args.n_layers)
    elif args.model == "GINlayers":
        model = GINlayers(n_f, n_h, n_hlin, n_p, args.n_layers)
    elif args.model == "SGCN":
        model = SGCN(n_f, n_h, n_p,args.n_layers)
    elif args.model == "EGNNpool":
        model = EGNNpool(n_f,n_ef,n_h,n_hlin,n_p)
    else:
        model = vanillaGCN(n_f, n_h, n_hlin, n_p)

    # Check if there is a gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    sched = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=int(len(trainload)),
                       epochs=total_epochs, final_div_factor=1e4)

    criterion = torch.nn.MSELoss()

    val_accur = []
    train_accur = []
    recorded_models = []

    print("Criteria for best model : ")
    print(args.mincriteria)
    print("Physical constraints: ")
    print(args.constraints)
    # If loading a model is specified
    if args.loadmodelparams is not None:
        PATH = os.path.join(modeldirname, args.loadmodelparams)
        model.load_state_dict(torch.load(PATH))
    # Train the model
    else:
        print("Training the model.")

        min_train_acc=10000.0
        epoch = 0
        for epoch in tqdm(range(0, total_epochs)):
            mod = train(args.model,trainload,args.constraints,device,args.alpha,predint=args.int,preds11=args.s11,predsij=predsij)
            train_acc = test(args.model,trainload,args.constraints,device,args.alpha,predint=args.int,preds11=args.s11,predsij=predsij)
            val_acc = test(args.model,valload,args.constraints,device,args.alpha,predint=args.int,preds11=args.s11,predsij=predsij)
            val_accur.append(val_acc)
            train_accur.append(train_acc)

            # Use train or validation accuracy as criteria for best model choice:
            if (args.mincriteria == "train"):
                criteria_acc = train_acc
            else:
                criteria_acc = val_acc

            if criteria_acc<min_train_acc:
                min_train_acc=criteria_acc
                recorded_models=mod.state_dict()
                minepoch=epoch
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.8f}, Val. Acc: {val_acc:.8f}')

        plt.figure(figsize=(10, 6))
        plt.plot(train_accur)
        plt.plot(val_accur)
        plt.yscale('Log')
        plt.ylabel("Loss", fontsize=20)
        plt.xlabel("Epochs", fontsize=20)
        #plt.show()
        plt.savefig(trainfigname)

        # For the best model
        bestmodel = recorded_models
        print("Best model at epoch ")
        print(minepoch)

        model.load_state_dict(bestmodel)

    print("Performing Inference")
    if args.inference:
        print("Performing Inference on training data set.")
        yr_train, yp_train = inference(args.model,model,trainload,args.constraints,device,predint=args.int,preds11=args.s11,predsij=predsij)
        print("Performing Inference on large validation data set.")
        yr_test, yp_test = inference(args.model,model,vallargeload,args.constraints,device,predint=args.int,preds11=args.s11,predsij=predsij)

    # Save the model
    if args.nosavemodel:
        pass
    else:
        print("Saving the model at")
        currmodelname=currmodeltemplate.format(args.model,str(total_epochs), lrval, lrexp, str(n_h), str(n_hlin), currexpname)
        PATH = os.path.join(modeldirname, currmodelname)
        print(PATH)

        # Which version of the code was used
        gitcommit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode('ascii').strip()

        loss = train_accur[minepoch]
        val_loss = val_accur[minepoch]

        torch.save({'Exp name' : currexpname,
                    'model' : args.model,
                    'epochs' : total_epochs,
                    'mincriteria' : args.mincriteria,
                    'phys_constraints': args.constraints,
                    'minepoch' : minepoch,
                    'batch_size' : args.batchsize,
                    'criterion' : "MSE",
                    'alpha' : args.alpha,
                    'cutoff' : args.cutoff,
                    'preprocessing ': args.preprocess,
                    'optimizer' : "Adam",
                    'init_lr' : init_lr,
                    'weight_decay' : weight_decay,
                    'n_hidden' : n_h,
                    'n_hlin': n_hlin,
                    'dropout' : args.dropout,
                    'loss': loss,
                    'val_loss': val_loss,
                    'train_loss_epoch' : train_accur,
                    'val_loss_epoch' : val_accur,
                    'model_state_dict': bestmodel,
                    'train_name' : trainname,
                    'val_name' : valname,
                    'test_name' : testname,
                    'vallarge_name' : vallargename,
                    'testlarge_name' : testlargename,
                    'git_commit' : gitcommit},PATH)

    if args.nosavefigures:
        pass
    else:
        if args.int:
            ploteffs(yr_train,yp_train,yr_test,yp_test,figtitle=effsfigname)

        if args.s11:
            fig = plt.figure(figsize=(6, 10))

            # Use only a smaller subset of total data set
            idxarray = np.arange(yr_test.shape[0])
            nchose = min([yr_test.shape[0], 181])

            chosen = np.random.choice(idxarray.shape[0], nchose, replace=False)

            stokeseval_tr = np.zeros((nchose, 181))
            stokeseval_te = np.zeros((nchose, 181))

            for ii in range(0, nchose):
                stokeseval_tr[ii, :] = yr_train[chosen[ii], 4:] - yp_train[chosen[ii], 4:]
                stokeseval_te[ii, :] = yr_test[chosen[ii], 4:] - yp_test[chosen[ii], 4:]

            fig.add_subplot(2, 1, 1)
            plt.imshow(stokeseval_tr)
            plt.title("S11 - train")
            fig.add_subplot(2, 1, 2)
            plt.imshow(stokeseval_te)
            plt.title("S11 - test")

            plt.savefig(stokesfigname)

            nstart = 4 # n_p-181
            agg = np.random.randint(0,yr_test.shape[0],size=4)
            plotS11(yr_train,yp_train,yr_test,yp_test,agg,nstart,figtitle=stokespredfigname)
        if args.sij>0:
            fig = plt.figure(figsize=(6, 10))
            nsij = args.sij*11
            # Use only a smaller subset of total data set
            idxarray = np.arange(yr_test.shape[0])
            nchose = min([yr_test.shape[0], nsij])

            chosen = np.random.choice(idxarray.shape[0], nchose, replace=False)

            stokeseval_tr = np.zeros((nchose, nsij))
            stokeseval_te = np.zeros((nchose, nsij))

            for ii in range(0, nchose):
                stokeseval_tr[ii, :] = yr_train[chosen[ii], 4:] - yp_train[chosen[ii], 4:]
                stokeseval_te[ii, :] = yr_test[chosen[ii], 4:] - yp_test[chosen[ii], 4:]

            fig.add_subplot(2, 1, 1)
            plt.imshow(stokeseval_tr)
            plt.title("Sij - train")
            fig.add_subplot(2, 1, 2)
            plt.imshow(stokeseval_te)
            plt.title("Sij - test")

            plt.savefig(sijfigname)

            yr_sij_test = yr_test[:,4:]
            sijr_test = yr_sij_test.reshape(yr_test.shape[0],nsij,11)

            yp_sij_test = yp_test[:,4:]
            sijp_test = yp_sij_test.reshape(yp_test.shape[0],nsij,11)

            agg = np.random.randint(0,yr_test.shape[0])
            plotsij(sijr_test,sijp_test,agg,figtitle=sijpredfigname)

