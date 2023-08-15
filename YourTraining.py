import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed

lr = 1e-2   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-2  # L2正则化
batchsize = 128

def buildGraph(Y):
    nodes = [StdScaler(mnist.mean_X, mnist.std_X), Linear(mnist.num_feat, 256),Dropout(0.2),relu(),
             Linear(256,256),Dropout(0.11),relu(),
             Linear(256, mnist.num_class),LogSoftmax(), NLLLoss(Y)] 
    graph=Graph(nodes)
    return graph

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/YT.npy"

X=mnist.trn_X
Y=mnist.trn_Y 
A = X.copy()
B = Y.copy()
mask = np.random.rand(*A.shape) < 0.1
np.putmask(A, mask, 0)
X = np.concatenate((X,A),axis = 0)
Y = np.concatenate((Y,B),axis = 0)


if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 200+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)