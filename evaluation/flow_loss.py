from torch.autograd import Function
from query_representation.utils import *
from cardinality_estimation.dataset import to_variable

import torch
import time
import pdb
from multiprocessing import Pool
from scipy import sparse
import scipy
import numpy as np

import platform
from ctypes import *
import os
import copy
import pkg_resources
from .cost_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lib_file = None
fl_cpp = None
system = platform.system()
if system == 'Linux':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lib_dir = os.path.join(dir_path, "..", "flow_loss_cpp")
    lib_file = "libflowloss.so"
    lib_file = lib_dir + "/" + lib_file
    fl_cpp = CDLL(lib_file, mode=RTLD_GLOBAL)
else:
    print("flow loss C library not being used as we are not on linux")

DEBUG_JAX = False
DEBUG = False
SOURCE_NODE_CONST = 100000

def get_costs_jax(card1, card2, card3, nilj, cost_model,
        total1, total2,penalty, rc, rf):

    if cost_model == "C":
        if nilj == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1;
        elif nilj == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2;
        # elif nilj == 3:
            # nilj_cost = card1 + card2;
        elif nilj == 4:
            nilj_cost = card2 + NILJ_CONSTANT*card1;
        else:
            assert False
        cost2 = card1*card2
        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
    else:
        assert False

    return cost

def get_subsetg_vectors(sample, cost_model, source_node=None,
        mysql_costs=False):
    start = time.time()
    # used for the mysql cost model
    edges_read_costs = None
    edges_rows_fetched = None

    node_dict = {}

    subsetg = sample["subset_graph"]
    add_single_node_edges(subsetg, SOURCE_NODE)

    nodes = list(subsetg.nodes())
    if SOURCE_NODE in nodes:
        nodes.remove(SOURCE_NODE)

    nodes.sort()

    join_graph = sample["join_graph"]
    edges = list(sample["subset_graph"].edges())
    edges.sort()

    N = len(nodes)
    num_edges = len(edges)
    M = len(edges)

    totals = np.zeros(N, dtype=np.float32)
    edges_head = [0]*M
    edges_tail = [0]*M
    edges_cost_node1 = [0]*M
    edges_cost_node2 = [0]*M
    edges_penalties = [1]*M

    nilj = [0]*M
    nilj2 = [0]*M
    final_node = 0
    max_len_nodes = 0

    for nodei, node in enumerate(nodes):
        node_dict[node] = nodei
        if len(node) > max_len_nodes:
            max_len_nodes = len(node)
            final_node = nodei

    for edgei, edge in enumerate(edges):
        if len(edge[0]) == len(edge[1]):
            assert edge[1] == SOURCE_NODE
            edges_head[edgei] = node_dict[edge[0]]
            edges_tail[edgei] = SOURCE_NODE_CONST
            edges_cost_node1[edgei] = SOURCE_NODE_CONST
            edges_cost_node2[edgei] = SOURCE_NODE_CONST
            if edges_read_costs is not None:
                edges_read_costs[edgei] = 1.0
                edges_rows_fetched[edgei] = 1.0
            continue

        edges_head[edgei] = node_dict[edge[0]]
        edges_tail[edgei] = node_dict[edge[1]]

        assert len(edge[1]) < len(edge[0])
        assert edge[1][0] in edge[0]
        ## FIXME:
        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        # node2.sort()
        node2 = tuple(node2)
        assert node2 in subsetg.nodes()

        # edge[0] is the joined node
        # node2 is the new node we joined to get to edge[0]
        assert len(node2) == 1

        edges_cost_node1[edgei] = node_dict[node1]
        edges_cost_node2[edgei] = node_dict[node2]

        ## FIXME: simplify these conditions
        if len(node1) == 1:
            # nilj[edgei] = 1
            nilj[edgei] = 4

            ## version from lc
            # fkey_join = True
            # join_edge_data = join_graph[node1[0]]
            # for other_node in node2:
                # if other_node not in join_edge_data:
                    # continue
                # jc = join_edge_data[other_node]["join_condition"]
                # if "!=" in jc:
                    # fkey_join = False
                    # break

            # if fkey_join:
                # nilj[edgei] = 2
            # else:
                # nilj[edgei] = 3

        elif len(node2) == 1:
            fkey_join = True
            join_edge_data = join_graph[node2[0]]
            for other_node in node1:
                if other_node not in join_edge_data:
                    continue
                jc = join_edge_data[other_node]["join_condition"]
                if "!=" in jc:
                    fkey_join = False
                    break

            if fkey_join:
                nilj[edgei] = 2
            else:
                nilj[edgei] = 3

    edges_head = np.array(edges_head, dtype=np.int32)
    edges_tail = np.array(edges_tail, dtype=np.int32)
    edges_cost_node1 = np.array(edges_cost_node1, dtype=np.int32)
    edges_cost_node2 = np.array(edges_cost_node2, dtype=np.int32)
    nilj = np.array(nilj, dtype=np.int32)
    edges_penalties = np.array(edges_penalties, dtype=np.float32)

    if edges_read_costs is not None:
        edges_read_costs = np.array(edges_read_costs, dtype=np.float32)
        edges_rows_fetched = np.array(edges_rows_fetched, dtype = np.float32)

    return totals, edges_head, edges_tail, nilj, \
            edges_cost_node1, edges_cost_node2, \
            edges_read_costs, edges_rows_fetched, \
            final_node, edges_penalties

def get_optimization_variables(ests, totals, min_val, max_val,
        ynormalization, edges_cost_node1, edges_cost_node2,
        nilj, edges_head, edges_tail, edges_read_costs, edges_rows_fetched,
        cost_model, edges_penalties):
    '''
    @ests: these are actual values for each estimate. totals,min_val,max_val
    are only required for the derivatives.
    '''
    start = time.time()
    if edges_read_costs is None:
        # dummy values just so the code doesn't crash
        edges_read_costs = np.random.rand(10)
        edges_rows_fetched = np.random.rand(10)

    if ynormalization == "log":
        norm_type = 2
    elif ynormalization is None:
        norm_type = 0
    else:
        assert False

    if cost_model == "C":
        cost_model_num = 11
    else:
        assert False

    # TODO: make sure everything is the correct type beforehand
    if min_val is None:
        min_val = 0.0
        max_val = 0.0

    if not isinstance(ests, np.ndarray):
        ests = ests.detach().cpu().numpy()

    costs2 = np.zeros(len(edges_cost_node1), dtype=np.float32)
    dgdxT2 = np.zeros((len(ests), len(edges_cost_node1)), dtype=np.float32)
    G2 = np.zeros((len(ests),len(ests)), dtype=np.float32)
    Q2 = np.zeros((len(edges_cost_node1),len(ests)), dtype=np.float32)

    assert ests.dtype == np.float32
    ests = np.maximum(ests, 1.0)

    start = time.time()
    fl_cpp.get_optimization_variables(ests.ctypes.data_as(c_void_p),
            totals.ctypes.data_as(c_void_p),
            c_double(min_val),
            c_double(max_val),
            c_int(norm_type),
            edges_cost_node1.ctypes.data_as(c_void_p),
            edges_cost_node2.ctypes.data_as(c_void_p),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            nilj.ctypes.data_as(c_void_p),
            edges_penalties.ctypes.data_as(c_void_p),
            edges_read_costs.ctypes.data_as(c_void_p),
            edges_rows_fetched.ctypes.data_as(c_void_p),
            c_int(len(ests)),
            c_int(len(costs2)),
            costs2.ctypes.data_as(c_void_p),
            dgdxT2.ctypes.data_as(c_void_p),
            G2.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p),
            c_int(cost_model_num))

    return costs2, dgdxT2, G2, Q2

def get_optimization_variables_jax(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2, edges_head,
        nilj, edges_read_costs, edges_rows_fetched,
        cost_model, G, Q, penalties):
    '''
    returns costs, and updates numpy arrays G, Q in place.
    '''
    ests = jp.exp((yhat+min_val)*(max_val-min_val))
    costs = jp.zeros(len(edges_cost_node1))
    ests = jp.maximum(ests, 1.0)

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs = jax.ops.index_update(costs, jax.ops.index[i], 1.0)
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = ests[node1]
        card2 = ests[node2]
        card3 = ests[edges_head[i]]
        total1 = totals[node1]
        total2 = totals[node2]
        penalty = penalties[i]
        if edges_read_costs is not None:
            rc = edges_read_costs[i]
            rf = edges_rows_fetched[i]
        else:
            rc = 0
            rf = 0

        cost = get_costs_jax(card1, card2, card3, nilj[i], cost_model, total1,
                total2, penalty, rc, rf)
        assert cost != 0.0
        costs = jax.ops.index_update(costs, jax.ops.index[i], cost)

    costs = 1 / costs
    return costs

def get_edge_costs3(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj):
    ests = jp.exp((yhat+min_val)*(max_val-min_val))
    costs = jp.zeros(len(edges_cost_node1))

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs = jax.ops.index_update(costs, jax.ops.index[i], 1.0)
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = ests[node1]
        card2 = ests[node2]
        if nilj[i] == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj[i] == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False
        cost = nilj_cost
        assert cost != 0.0
        costs = jax.ops.index_update(costs, jax.ops.index[i], cost)

    costs = 1 / costs;
    return costs

def get_edge_costs2(yhat, totals, min_val, max_val,
        normalization_type, edges_cost_node1, edges_cost_node2,
        nilj):
    '''
    @ests: cardinality estimates for each nodes (sorted by node_names)
    @totals: Total estimates for each node (sorted ...)
    @min_val, max_val, normalization_type

    @edges_cost_node1:
    @edges_cost_node2: these are single tables..
    '''
    start = time.time()
    ests = to_variable(np.zeros(len(yhat), dtype=np.float32),
            requires_grad=True).float()
    ests.requires_grad = True
    for i in range(len(yhat)):
        if normalization_type == "log":
            ests[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            ests[i] = yhat[i]*totals[i]
        else:
            assert False

    dgdxT = torch.zeros(len(ests), len(edges_cost_node1))
    # costs = to_variable(np.zeros(len(edges_cost_node1)), requires_grad=True).float()
    costs = torch.zeros(len(edges_cost_node1), requires_grad=True).float()

    for i in range(len(edges_cost_node1)):
        if edges_cost_node1[i] == SOURCE_NODE_CONST:
            costs[i] = 1.0
            continue

        node1 = edges_cost_node1[i]
        node2 = edges_cost_node2[i]
        card1 = torch.add(ests[node1], 1.0)
        card2 = torch.add(ests[node2], 1.0)
        hash_join_cost = card1 + card2
        if nilj[i] == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif nilj[i] == 2:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            nilj_cost = 10000000000
        cost = torch.min(hash_join_cost, nilj_cost)
        assert cost != 0.0
        costs[i] = cost

        if normalization_type is None:
            continue

        # time to compute gradients
        if normalization_type == "pg_total_selectivity":
            total1 = totals[node1]
            total2 = totals[node2]
            if hash_join_cost < nilj_cost:
                assert cost == hash_join_cost
                # - (a / (ax_1 + bx_2)**2)
                dgdxT[node1, i] = - (total1 / ((hash_join_cost)**2))
                # - (b / (ax_1 + bx_2)**2)
                dgdxT[node2, i] = - (total2 / ((hash_join_cost)**2))
            else:
                # index nested loop join
                assert cost == nilj_cost
                if nilj[i] == 1:
                    # - (a / (ax_1 + bx_2)**2)
                    num1 = total1*NILJ_CONSTANT
                    dgdxT[node1, i] = - (num1 / ((cost)**2))
                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[node2, i] = - (total2 / ((cost)**2))
                else:
                    # node 2
                    # - (a / (ax_1 + bx_2)**2)
                    num2 = total2*NILJ_CONSTANT
                    dgdxT[node1, i] = - (total1 / ((cost)**2))
                    # - (b / (ax_1 + bx_2)**2)
                    dgdxT[node2, i] = - (num2 / ((cost)**2))
        else:
            if hash_join_cost <= nilj_cost:
                assert cost == hash_join_cost
                # - (ae^{ax} / (e^{ax} + e^{ax2})**2)
                # e^{ax} is just card1
                dgdxT[node1, i] = - (max_val*card1 / ((hash_join_cost)**2))
                dgdxT[node2, i] = - (max_val*card2 / ((hash_join_cost)**2))

            else:
                # index nested loop join
                assert cost == nilj_cost
                if nilj[i]  == 1:
                    dgdxT[node1, i] = - (max_val*card1*NILJ_CONSTANT / ((cost)**2))
                    dgdxT[node2, i] = - (max_val*card2 / ((cost)**2))
                else:
                    # num2 = card2*NILJ_CONSTANT
                    dgdxT[node1, i] = - (max_val*card1 / ((cost)**2))
                    dgdxT[node2, i] = - (max_val*card2*NILJ_CONSTANT / ((cost)**2))

    # print("get edge costs took: ", time.time()-start)
    return costs, dgdxT

def single_forward2(yhat, totals, edges_head, edges_tail, edges_cost_node1,
        edges_cost_node2, nilj, normalization_type, min_val, max_val,
        trueC_vec, final_node, edges_read_costs, edges_rows_fetched,
        cost_model, penalties):
    '''
    @yhat: NN outputs for nodes (sorted by nodes.sort())
    @totals: Total estimates for each node (sorted ...)
    @edges_head: len() == num edges. Each element is an index of the head node
    in that edge
    @edges_tail: ...
    ## which nodes determine the cost in each edge
    @edges_cost_node1:
    @edges_cost_node2: these are single tables..
    '''
    if DEBUG_JAX:
        import jax
        import jax.numpy as jp
        from jax import jacfwd, jacrev

        # costs_grad_fn = jacfwd(get_optimization_variables_jax, argnums=0)
        # costs_grad_fn = jacrev(get_optimization_variables_jax, argnums=0)
        jax_start = time.time()
        yhat = np.array(yhat)
        G = np.zeros((len(yhat),len(yhat)), dtype=np.float32)
        Q = np.zeros((len(edges_cost_node1),len(yhat)), dtype=np.float32)

        costs = get_optimization_variables_jax(yhat, totals, min_val, max_val,
                normalization_type, edges_cost_node1, edges_cost_node2, edges_head,
                nilj,
                edges_read_costs, edges_rows_fetched,
                cost_model, G, Q, penalties)
        print(costs.shape, np.max(costs))
        costs_grad = jacfwd(get_optimization_variables_jax, argnums=0)(yhat, totals, min_val,
                max_val, normalization_type, edges_cost_node1, edges_cost_node2,
                edges_head, nilj,
                edges_read_costs, edges_rows_fetched,
                cost_model, G, Q, penalties)
        costs_grad = costs_grad.T
        print(costs_grad.shape, np.max(costs_grad))

        print("jax stuff took: ", time.time()-jax_start)
        yhat = torch.from_numpy(yhat)

    start = time.time()
    est_cards = to_variable(np.zeros(len(yhat), dtype=np.float32)).float()
    for i in range(len(yhat)):
        if normalization_type == "log":
            est_cards[i] = torch.exp(((yhat[i] + min_val)*(max_val-min_val)))
        elif normalization_type == "pg_total_selectivity":
            est_cards[i] = yhat[i]*totals[i]
        else:
            assert False

    start = time.time()
    predC2, dgdxT2, G2, Q2 = get_optimization_variables(est_cards, totals,
            min_val, max_val, normalization_type, edges_cost_node1,
            edges_cost_node2, nilj, edges_head, edges_tail,
            edges_read_costs, edges_rows_fetched,
            cost_model,
            penalties)

    if DEBUG_JAX:
        print("min max estimates: ", np.min(est_cards.detach().numpy()),
                np.max(est_cards.detach().numpy()))
        print("non jax stuff took: ", time.time()-start)
        predC = 1.0 / predC2
        print(np.allclose(predC, costs))
        print(np.min(predC), np.max(predC))
        print(np.min(costs), np.max(costs))

        if not np.allclose(predC, costs):
            print("costs not close!")
            # pdb.set_trace()
            assert False

        print(np.allclose(dgdxT2, costs_grad))
        if not np.allclose(dgdxT2, costs_grad):
            print("cost grads not close!")
            print("norm diff: ", np.linalg.norm(dgdxT2 - costs_grad))
            # assert False

        print(np.min(dgdxT2), np.max(dgdxT2))
        print(np.min(costs_grad), np.max(costs_grad))

        # pdb.set_trace()

    Gv2 = np.zeros(len(totals), dtype=np.float32)
    Gv2[final_node] = 1.0

    mat_start = time.time()
    predC2 = to_variable(predC2).float()
    dgdxT2 = to_variable(dgdxT2).float()

    # start = time.time()
    # np.savez("test", data=M.data, indices=M.indices,
                         # indptr=M.indptr, shape=M.shape)
    # np.save("test.npy", G2)
    # A=scipy.sparse.csr_matrix(G2, dtype=np.float32)
    # ml = pyamg.ruge_stuben_solver(A)
    # x = ml.solve(Gv2, tol=1e-8)
    # print("linear solver took: ", time.time() - start)
    # invG2 = np.outer(x, Gv2)
    # print(invG2.shape)
    # print("residual: ", np.linalg.norm(Gv2-A*x))
    # start = time.time()

    Gv2 = to_variable(Gv2).float().to(device)
    G2 = to_variable(G2).float().to(device)

    invstart = time.time()
    # might fail if it is not invertible
    try:
        invG = torch.inverse(G2)
    except:
        print("exception inverting!")
        invG = torch.pinverse(G2)

    invG = to_variable(invG).float()

    # print("inversion took: ", time.time() - start)
    # print(np.linalg.norm(invG.detach().cpu().numpy() - invG2))

    v = invG @ Gv2 # vshape: Nx1
    v = v.detach().cpu().numpy()

    # invG = torch.pinverse(G2)

    # invG = scipy.linalg.inv(G2)
    # invG = np.linalg.inv(G2)

    # invG = tf.linalg.inv(G2)
    # invG = invG.numpy()

    # M=scipy.sparse.csc_matrix(G2)
    # invG = scipy.sparse.linalg.inv(M)
    # invG = invG.A


    # flows = Q2 @ v
    # if np.min(flows) < 0.0:
        # print("flows max-min-mean!")
        # print(np.max(flows), np.min(flows), np.mean(flows))

    # TODO: we don't even need to compute the loss here if we don't want to
    loss2 = np.zeros(1, dtype=np.float32)

    v = v.astype(np.float32)
    assert Q2.dtype == np.float32
    assert v.dtype == np.float32
    if isinstance(trueC_vec, torch.Tensor):
        trueC_vec = trueC_vec.detach().cpu().numpy()
    assert trueC_vec.dtype == np.float32
    # just computes the loss
    fl_cpp.get_qvtqv(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q2.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            trueC_vec.ctypes.data_as(c_void_p),
            loss2.ctypes.data_as(c_void_p)
            )

    # print("forward took: ", time.time()-start)
    return to_variable(loss2).float(), dgdxT2.detach(), invG.detach().cpu().numpy(), Q2, v
    # return to_variable(loss2).float(), to_variable(dgdxT2).float(), invG, Q2, v

def single_backward(Q, invG,
        v, dgdxT, opt_flow_loss, trueC_vec,
        edges_head, edges_tail, normalize_flow_loss):

    start = time.time()

    assert Q.dtype == np.float32
    assert invG.dtype == np.float32

    QinvG2 = np.zeros((Q.shape[0], invG.shape[1]), dtype=np.float32)
    fl_cpp.get_qinvg(c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q.ctypes.data_as(c_void_p),
            invG.ctypes.data_as(c_void_p),
            QinvG2.ctypes.data_as(c_void_p))

    if isinstance(trueC_vec, torch.Tensor):
        # trueC_vec = trueC_vec.detach().numpy()
        trueC_vec = trueC_vec.detach().cpu().numpy()

    assert trueC_vec.dtype == np.float32
    assert Q.dtype == np.float32
    assert v.dtype == np.float32

    tqv_start = time.time()
    tQv = np.zeros(len(edges_head), dtype=np.float32)
    fl_cpp.get_tqv(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            Q.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            trueC_vec.ctypes.data_as(c_void_p),
            c_int(2),
            tQv.ctypes.data_as(c_void_p)
            )
    dCdg2 = np.zeros(tQv.shape, dtype=np.float32)

    dfdg = np.zeros((len(edges_head), len(edges_head)), dtype=np.float32)
    dfdg_start = time.time()
    num_threads = int(len(edges_head) / 400)
    num_threads = max(1, num_threads)
    num_threads = min(32, num_threads)

    fl_cpp.get_dfdg(
            c_int(len(edges_head)),
            c_int(len(v)),
            edges_head.ctypes.data_as(c_void_p),
            edges_tail.ctypes.data_as(c_void_p),
            QinvG2.ctypes.data_as(c_void_p),
            tQv.ctypes.data_as(c_void_p),
            v.ctypes.data_as(c_void_p),
            dfdg.ctypes.data_as(c_void_p),
            dCdg2.ctypes.data_as(c_void_p),
            c_int(num_threads))

    # mat_start = time.time()
    # slow matrix mul when we have too many edges
    # dCdg = dfdg @ tQv
    # print("mat took: ", time.time()-mat_start)

    yhat_grad = dgdxT @ to_variable(dCdg2).float()

    if normalize_flow_loss:
        yhat_grad /= opt_flow_loss

    return yhat_grad

class FlowLoss(Function):
    @staticmethod
    def forward(ctx, yhat, y, normalization_type,
            min_val, max_val, subsetg_vectors,
            normalize_flow_loss,
            pool, cost_model):
        '''
        '''

        # Note: do flow loss computation and save G, invG etc. for backward
        # pass
        torch.set_num_threads(1)
        start = time.time()
        yhat = yhat.detach()
        ctx.pool = pool
        ctx.normalize_flow_loss = normalize_flow_loss
        ctx.subsetg_vectors = subsetg_vectors
        assert len(subsetg_vectors[0][0]) == 10
        start = time.time()
        ctx.dgdxTs = []
        ctx.invGs = []
        ctx.Qs = []
        ctx.vs = []

        totals, edges_head, edges_tail, nilj, edges_cost_node1, \
                edges_cost_node2, \
                edges_read_costs, edges_rows_fetched, \
                final_node, edge_penalties = ctx.subsetg_vectors[0][0]
        trueC_vec, opt_flow_loss = ctx.subsetg_vectors[0][1], \
                        ctx.subsetg_vectors[0][2]

        start = time.time()
        res = single_forward2(yhat, totals,
                edges_head, edges_tail, edges_cost_node1,
                edges_cost_node2,
                nilj,
                normalization_type,
                min_val, max_val,
                trueC_vec, final_node,
                edges_read_costs, edges_rows_fetched,
                cost_model, edge_penalties)

        loss = res[0]
        ctx.dgdxTs.append(res[1])
        ctx.invGs.append(res[2])
        ctx.Qs.append(res[3])
        ctx.vs.append(res[4])
        # print("Num ctx.Qs in forward: ", len(ctx.Qs))
        assert len(ctx.Qs) == 1

        if normalize_flow_loss == 1:
            loss = loss / opt_flow_loss

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        return gradients wrt preds, and bunch of Nones
        '''
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        start = time.time()
        assert ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]

        _, edges_head, edges_tail, _, _, \
                _, _,_,_,_ = ctx.subsetg_vectors[0][0]
        trueC_vec, opt_cost = ctx.subsetg_vectors[0][1], \
                                ctx.subsetg_vectors[0][2]

        yhat_grad = single_backward(
                         ctx.Qs[0], ctx.invGs[0],
                         ctx.vs[0], ctx.dgdxTs[0],
                         opt_cost, trueC_vec,
                         edges_head,
                         edges_tail,
                         ctx.normalize_flow_loss)

        yhat_grad = yhat_grad.to(device, non_blocking=True)
        return yhat_grad,None,None,None,None,None,None,None,None,None

