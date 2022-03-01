#include <iostream>
#include <cmath>
#include <math.h>
#include "omp.h"

#define NILJ_CONSTANT 0.001
#define NILJ_CONSTANT2 2.0
#define RATIO_MUL_CONSTANT 1.0
#define SOURCE_NODE_CONST 100000
#define NILJ_MIN_CARD 5.0
#define CARD_DIVIDER 0.001
#define SEQ_CONSTANT 5.0

//#define OLD_TIMEOUT_CONSTANT 150001001.0
#define RC_CONST 699999.0

void get_dfdg_par(int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *QinvG, float *v, float *dfdg, int batch,
    int batch_size, float* tQv, float *dcdg)
{
  //printf("thread %d\n", omp_get_thread_num());
  int hnode, tnode, start, end;
  float vh, vt, qgh, qgt, cur_dfdg, cur_tqv;
  start = batch*batch_size;
  end = start+batch_size;
  if (end > num_edges) end = num_edges;

  for (int i = start; i < end; i++) {
    hnode = edges_head[i];
    tnode = edges_tail[i];
    vh = v[hnode];
    if (tnode != SOURCE_NODE_CONST) {
      vt = v[tnode];
    } else vt = 0.0;
    //printf("edge: %d, hnode: %d, tnode: %d, vh: %f, vt: %f\n", i, hnode, tnode, vh, vt);

    // going to calculate the value of dcdg[i] here
    dcdg[i] = 0.0;
    for (int j = 0; j < num_edges; j++) {
      if (i == j) {
        qgh = 1.0;
        qgt = -1.0;
      } else {
        qgh = 0.0;
        qgt = 0.0;
      }

      dfdg[i*num_edges + j] = vh*(qgh - QinvG[j*num_nodes + hnode]);
      if (tnode != SOURCE_NODE_CONST) {
        dfdg[i*num_edges + j] += vh*(QinvG[j*num_nodes + tnode]);
        dfdg[i*num_edges + j] += vt*(qgt + QinvG[j*num_nodes + hnode] \
            - QinvG[j*num_nodes+tnode]);
      }
      cur_dfdg = dfdg[i*num_edges + j];
      cur_tqv = tQv[j];
      dcdg[i] += cur_dfdg*cur_tqv;
    }
  }
}

extern "C" void get_dfdg (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *QinvG, float *tQv,
    float *v, float *dfdg, float *dcdg,
    int num_workers)
{
  //printf("get dfdg, %d, %d\n", num_edges, num_nodes);
  //int hnode, tnode;
  //float vh, vt, qgh, qgt;
  int batch_size, num_batches;
  batch_size = num_edges / num_workers;

  int rem = num_edges % batch_size;
  num_batches = ceil(num_edges / batch_size);
  //printf("rem: %d\n", rem);
  if (rem != 0) num_batches += 1;

  #pragma omp parallel for num_threads(num_workers)
  for (int batch = 0; batch < num_batches; batch++) {
    get_dfdg_par(num_edges, num_nodes, edges_head,
        edges_tail, QinvG, v, dfdg, batch, batch_size, tQv, dcdg);
  }
}

/* fills in tqv, Mx1 array
 * */
extern "C" void get_tqv (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *v, float *trueC, int const_mul,
    float *tqv)
{
  int head_node, tail_node;

  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    /* will set tqv[i] now */
    tqv[i] = Q[i*num_nodes + head_node] * v[head_node];
    //printf("head node: %d, tail node: %d\n", head_node, tail_node);
    if (tail_node != SOURCE_NODE_CONST) {
      tqv[i] += Q[i*num_nodes + tail_node] * v[tail_node];
    }
    tqv[i] *= trueC[i]*const_mul;
  }
}

/* fills in tqv, len 1 array
 * */
extern "C" void get_qvtqv (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *v, float *trueC,
    float *tqv)
{
  int head_node, tail_node;
  float cur_val;

  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    /* will set tqv[i] now */
    cur_val = 0.0;
    cur_val += Q[i*num_nodes + head_node] * v[head_node];
    if (tail_node != SOURCE_NODE_CONST) {
      cur_val += Q[i*num_nodes + tail_node] * v[tail_node];
    }
    cur_val = cur_val * cur_val;
    //printf("%d, %d\n", head_node, tail_node);
    tqv[0] += trueC[i]*cur_val;
  }
}

/* fills in QinvG mat mul, knowing the sparsity pattern of Q
 * M: num_edges, N: num_nodes
 * Q: M x N: each row has zeros except in the nodes making that edge
 * invG: NxN not sparse
 * QinvG: MxN
 * */
extern "C" void get_qinvg (int num_edges, int num_nodes,
    int *edges_head, int *edges_tail,
    float *Q, float *invG,
    float *QinvG)
{
  int head_node, tail_node, idx;
  for (int i = 0; i < num_edges; i++)
  {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    for (int j = 0; j < num_nodes; j++)
    {
      idx = i*num_nodes + j;
      QinvG[idx] = Q[i*num_nodes + head_node]*invG[head_node*num_nodes + j];
      if (tail_node != SOURCE_NODE_CONST) {
        QinvG[idx] += Q[i*num_nodes + tail_node]*invG[tail_node*num_nodes + j];
      }
    }
  }
}

void get_costs10(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  float cost1, cost2;

  // FIXME: need to multiply card2 by another constant
  if (card1 < NILJ_MIN_CARD || card2 < NILJ_MIN_CARD) {
    cost1 = 100000000000.0;
    cost2 = card1*card2;
    costs[i] = card1*card2;
  } else {
    if (card1 > card2) {
      cost1 = NILJ_CONSTANT2*card1*RATIO_MUL_CONSTANT;
    } else {
      cost1 = NILJ_CONSTANT2*card2*RATIO_MUL_CONSTANT;
    }
    cost2 = card1*card2;
    if (cost1 <= cost2) costs[i] = cost1;
    else costs[i] = cost2;
  }

  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
    exit(-1);
  } else if (normalization_type == 2) {
      if (cost1 <= cost2) {
        dgdxT[node1*num_edges + i] = 0.0;
        dgdxT[node2*num_edges + i] = 0.0;
        if (card1 < card2) dgdxT[node2*num_edges + i] = - max_val / costs[i];
        else dgdxT[node1*num_edges + i] = - max_val / costs[i];

      } else {
          // cost2 < cost1, implies cost = card1*card2
          // derivative of 1 / (e^{ax}*e^{ay}), where y is the other nodes
          // prediction (x = log(card1) / max_val, y = log(card2) / max_val)
          dgdxT[node1*num_edges + i] = - max_val / costs[i];
          dgdxT[node2*num_edges + i] = - max_val / costs[i];
      }
  }
}

void get_costs13(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, cost1, cost2, total1, total2, node_selectivity,
      joined_node_est;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  total1 = totals[node1];
  total2 = totals[node2];

  if (nilj[i] == 1) {
    // using index on 1
    cost1 = card2;
    node_selectivity = total1 / card1;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else if (nilj[i] == 2) {
    cost1 = card1;
    node_selectivity = total2 / card2;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }
  //cost2 = card1*card2;
  //if (cost2 < cost1) {
    //costs[i] = cost2;
  costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
    // index nested loop join
    if (nilj[i]  == 1) {
        // used index on node1.
        // derivative of: 1 / (c3*e^{-ax}*total + c2)
        dgdxT[node1*num_edges + i] = (max_val*card3*total1) / (card1*cost*cost);
        dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
        dgdxT[head_node*num_edges + i] = - (max_val*card3*total1) / (card1*cost*cost);
    } else {
        dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
        dgdxT[node2*num_edges + i] = (max_val*card3*total2) / (card2*cost*cost);
        dgdxT[head_node*num_edges + i] = - (max_val*card3*total2) / (card2*cost*cost);
    }
  }
}

void get_costs15(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node,
    float *edges_penalties)
{
  // head node is the joined node
  float card1, card2, card3, cost1, cost2, total1, total2, node_selectivity,
      joined_node_est, penalty;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  total1 = totals[node1];
  total2 = totals[node2];
  penalty = edges_penalties[i];

  if (nilj[i] == 1) {
    // using index on 1
    cost1 = card2;
    node_selectivity = total1 / card1;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
    cost1 *= penalty;
  } else if (nilj[i] == 2) {
    cost1 = card1;
    node_selectivity = total2 / card2;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
    cost1 *= penalty;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }

  cost2 = card1*card2;
  if (cost2 < cost1) {
    costs[i] = cost2;
  } else costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
    if (cost2 < cost1) {
          dgdxT[node1*num_edges + i] = - max_val / costs[i];
          dgdxT[node2*num_edges + i] = - max_val / costs[i];
    } else {
        // index nested loop join
        if (nilj[i]  == 1) {
            // used index on node1.
            // derivative of: 1 / (c3*e^{-ax}*total + c2)
            dgdxT[node1*num_edges + i] = (penalty*max_val*card3*total1) / (card1*cost*cost);
            dgdxT[node2*num_edges + i] = - (penalty*max_val*card2) / (cost*cost);
            dgdxT[head_node*num_edges + i] = - (penalty*max_val*card3*total1) / (card1*cost*cost);
        } else {
            dgdxT[node1*num_edges + i] = - (penalty*max_val*card1) / (cost*cost);
            dgdxT[node2*num_edges + i] = (penalty*max_val*card3*total2) / (card2*cost*cost);
            dgdxT[head_node*num_edges + i] = - (penalty*max_val*card3*total2) / (card2*cost*cost);
        }
    }
  }
}

void get_costs17(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, cost1, cost2, total1, total2, node_selectivity,
      joined_node_est;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  total1 = totals[node1];
  total2 = totals[node2];

  if (nilj[i] == 1) {
    // using index on 1
    cost1 = card2;
    node_selectivity = total1 / card1;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else if (nilj[i] == 2) {
    cost1 = card1;
    node_selectivity = total2 / card2;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else if (nilj[i] == 3) {
    cost1 = SEQ_CONSTANT*card1*card2;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }

  costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
    if (nilj[i] == 3) {
          dgdxT[node1*num_edges + i] = - SEQ_CONSTANT*max_val / costs[i];
          dgdxT[node2*num_edges + i] = - SEQ_CONSTANT*max_val / costs[i];
    } else {
        // index nested loop join
        if (nilj[i]  == 1) {
            // used index on node1.
            // derivative of: 1 / (c3*e^{-ax}*total + c2)
            dgdxT[node1*num_edges + i] = (max_val*card3*total1) / (card1*cost*cost);
            dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total1) / (card1*cost*cost);
        } else {
            dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
            dgdxT[node2*num_edges + i] = (max_val*card3*total2) / (card2*cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total2) / (card2*cost*cost);
        }
    }
  }
}

void get_costs12(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, cost1, cost2, total1, total2, node_selectivity,
      joined_node_est;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  total1 = totals[node1];
  total2 = totals[node2];

  if (nilj[i] == 1) {
    // using index on 1
    cost1 = card2;
    node_selectivity = total1 / card1;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else if (nilj[i] == 2) {
    cost1 = card1;
    node_selectivity = total2 / card2;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }
  cost2 = SEQ_CONSTANT*card1*card2;
  if (cost2 < cost1) {
    costs[i] = cost2;
  } else costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
    if (cost2 < cost1) {
          dgdxT[node1*num_edges + i] = - SEQ_CONSTANT*max_val / costs[i];
          dgdxT[node2*num_edges + i] = - SEQ_CONSTANT*max_val / costs[i];
    } else {
        // index nested loop join
        if (nilj[i]  == 1) {
            // used index on node1.
            // derivative of: 1 / (c3*e^{-ax}*total + c2)
            dgdxT[node1*num_edges + i] = (max_val*card3*total1) / (card1*cost*cost);
            dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total1) / (card1*cost*cost);
        } else {
            dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
            dgdxT[node2*num_edges + i] = (max_val*card3*total2) / (card2*cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total2) / (card2*cost*cost);
        }
    }
  }
}

void get_costs16(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, cost1, cost2, total1, total2, node_selectivity,
      joined_node_est;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  total1 = totals[node1];
  total2 = totals[node2];

  if (nilj[i] == 1) {
    // using index on 1
    cost1 = card2 + NILJ_CONSTANT*card1;
    cost2 = NILJ_CONSTANT*card1;
    node_selectivity = total1 / card1;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else if (nilj[i] == 2) {
    cost1 = card1 + NILJ_CONSTANT*card2;
    cost2 = NILJ_CONSTANT*card2;
    node_selectivity = total2 / card2;
    joined_node_est = card3*node_selectivity;
    cost1 += joined_node_est;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }
  cost2 += card1*card2;
  if (cost2 < cost1) {
    costs[i] = cost2;
  } else costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
    if (cost2 < cost1) {
        if (nilj[i] == 1) {
          //dgdxT[node1*num_edges + i] = - max_val / costs[i];
          //dgdxT[node2*num_edges + i] = - max_val / costs[i];
          dgdxT[node1*num_edges + i] = -(max_val*card1*NILJ_CONSTANT + \
              max_val*card2*card1) / (costs[i]*costs[i]);
          dgdxT[node2*num_edges + i] = -max_val*card2*card1 / (costs[i]*costs[i]);
        } else {
          dgdxT[node2*num_edges + i] = -(max_val*card2*NILJ_CONSTANT + \
              max_val*card2*card1) / (costs[i]*costs[i]);
          dgdxT[node1*num_edges + i] = - max_val*card2*card1 / (costs[i]*costs[i]);
        }
    } else {
        // index nested loop join
        if (nilj[i]  == 1) {
            // used index on node1, let card of node 1: e^{ax}
            // derivative of: 1 / (c3*e^{-ax}*total + c2 + NILJ_CONST*e^{ax})
            dgdxT[node1*num_edges + i] = (max_val*card3*total1 - \
                      max_val*NILJ_CONSTANT*card1) / (card1*cost*cost);
            dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total1) / (card1*cost*cost);
        } else {
            dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
            dgdxT[node2*num_edges + i] = (max_val*card3*total2 - \
                max_val*NILJ_CONSTANT*card2) / (card2*cost*cost);
            dgdxT[head_node*num_edges + i] = - (max_val*card3*total2) / (card2*cost*cost);
        }
    }
  }
}

void get_costs21(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *nilj, float *edges_read_costs, float *edges_rows_fetched,
    int num_nodes, int num_edges,
    float *costs, float *dgdxt, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, nilj_cost, ratio_mul, rc, rf;
  int node1, node2;
  float cost1, cost2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  rf = edges_rows_fetched[i];

  rc = edges_read_costs[i];
  //rc = 0.0;
  //rc = edges_read_costs
  if (nilj[i] == 4) {
    rc += card1;
    rc += card2;
  } else if (card2 > card1*rf) {
    rc += card1*rf;
  } else {
    rc += card2;
  }

  float eval_const = 0.1;

  costs[i] = rc + eval_const*card1*rf;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    dgdxt[node1*num_edges + i] = -(max_val*card1*eval_const*rf) / (cost*cost);
    if (nilj[i] == 4) {
      dgdxt[node1*num_edges + i] -= (max_val*card1) / (cost*cost);
      dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
    } else if (card2 > card1*rf) {
      dgdxt[node1*num_edges + i] -= (max_val*card1*rf) / (cost*cost);
      dgdxt[node2*num_edges + i] = 0.0;
    } else {
      dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
    }
    dgdxt[head_node*num_edges + i] = 0.0;
  }
}

void get_costs20(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *nilj, float *edges_read_costs, float *edges_rows_fetched,
    int num_nodes, int num_edges,
    float *costs, float *dgdxt, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, nilj_cost, ratio_mul, rc, rf;
  int node1, node2;
  float cost1, cost2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];

  rf = edges_rows_fetched[i];
  card3 = ests[head_node];

  rf = edges_rows_fetched[i];

  rc = 0.0;
  if (nilj[i] == 4) {
    rc += card1;
    rc += card2;
  } else {
    rc += card2;
  }

  float eval_const = 0.1;

  cost1 = rc + eval_const*card1*rf;
  cost2 = eval_const*card1*card2;

  if (cost2 < cost1) {
    costs[i] = cost2;
  } else costs[i] = cost1;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
      if (cost2 < cost1) {
          // FIXME: eval_const**2 or not??
          dgdxt[node1*num_edges + i] = - (max_val) / (costs[i]);
          dgdxt[node2*num_edges + i] = - (max_val) / (costs[i]);
      } else {
          dgdxt[node1*num_edges + i] = -(max_val*card1*eval_const*rf) / (cost*cost);
          if (nilj[i] == 4) {
            dgdxt[node1*num_edges + i] -= (max_val*card1) / (cost*cost);
            dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          } else {
            dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          }
          dgdxt[head_node*num_edges + i] = 0.0;
      }
  }
}

void get_costs19(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *nilj, float *edges_read_costs, float *edges_rows_fetched,
    int num_nodes, int num_edges,
    float *costs, float *dgdxt, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, nilj_cost, ratio_mul, rc, rf;
  int node1, node2;
  float cost1, cost2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];

  rf = edges_rows_fetched[i];

  rc = 0.0;
  if (nilj[i] == 4) {
    rc += card1;
    rc += card2;
  } else if (card2 > card1*rf) {
    rc += card1*rf;
  } else {
    rc += card2;
  }

  float eval_const = 0.1;

  costs[i] = rc + eval_const*card1*rf;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    dgdxt[node1*num_edges + i] = -(max_val*card1*eval_const*rf) / (cost*cost);
    if (nilj[i] == 4) {
      dgdxt[node1*num_edges + i] -= (max_val*card1) / (cost*cost);
      dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
    } else if (card2 > card1*rf) {
      dgdxt[node1*num_edges + i] -= (max_val*card1*rf) / (cost*cost);
      dgdxt[node2*num_edges + i] = 0.0;
    } else {
      dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
    }
    dgdxt[head_node*num_edges + i] = 0.0;
  }
}

void get_costs18(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *nilj, float *edges_read_costs, float *edges_rows_fetched,
    int num_nodes, int num_edges,
    float *costs, float *dgdxt, int i, int head_node)
{
  // head node is the joined node
  float card1, card2, card3, nilj_cost, ratio_mul, rc, rf;
  int node1, node2;
  float cost1, cost2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  //card3 = ests[head_node];

  rf = edges_rows_fetched[i];

  rc = 0.0;
  if (nilj[i] == 4) {
    rc += card1;
    rc += card2;
  } else if (card2 > card1*rf) {
    rc += card1*rf;
  } else {
    rc += card2;
  }

  float eval_const = 0.1;

  cost1 = rc + eval_const*card1*rf;
  cost2 = eval_const*card1*card2;

  if (cost2 < cost1) {
    costs[i] = cost2;
  } else costs[i] = cost1;
  float cost = costs[i];

  //costs[i] = cost;

  /* time to compute gradients */
  if (normalization_type == 2) {
      if (cost2 < cost1) {
          // FIXME: eval_const**2 or not??
          dgdxt[node1*num_edges + i] = - (max_val) / (costs[i]);
          dgdxt[node2*num_edges + i] = - (max_val) / (costs[i]);
      } else {
          dgdxt[node1*num_edges + i] = -(max_val*card1*eval_const*rf) / (cost*cost);
          if (nilj[i] == 4) {
            dgdxt[node1*num_edges + i] -= (max_val*card1) / (cost*cost);
            dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          } else if (card2 > card1*rf) {
            dgdxt[node1*num_edges + i] -= (max_val*card1*rf) / (cost*cost);
            dgdxt[node2*num_edges + i] = 0.0;
          } else {
            dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
          }

          //if (nilj[i] == 4) {
            //dgdxt[node1*num_edges + i] = - (max_val*card1*eval_const*rf + max_val*card1) / (cost*cost);
          //} else {
            //dgdxt[node1*num_edges + i] = - (max_val*card1*eval_const*rf) / (cost*cost);
          //}

          dgdxt[head_node*num_edges + i] = 0.0;
      }
  }
}

void get_costs11(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxt, int i, int head_node)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  float cost1, cost2;

  if (nilj[i] == 1 || nilj[i] == 4) {
    nilj_cost = card2 + NILJ_CONSTANT*card1;
  } else if (nilj[i] == 2) {
    nilj_cost = card1 + NILJ_CONSTANT*card2;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }

  cost2 = card1*card2;
  if (cost2 < nilj_cost) {
    costs[i] = cost2;
  } else costs[i] = nilj_cost;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
      if (cost2 < nilj_cost) {
          dgdxt[node1*num_edges + i] = - max_val / costs[i];
          dgdxt[node2*num_edges + i] = - max_val / costs[i];
      } else {
        // index nested loop join
        if (nilj[i]  == 1 || nilj[i] == 4) {
            dgdxt[node1*num_edges + i] = - (max_val*card1*NILJ_CONSTANT) / (cost*cost);
            dgdxt[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
            dgdxt[head_node*num_edges + i] = 0.0;
        } else {
            dgdxt[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
            dgdxt[node2*num_edges + i] = - (max_val*card2*NILJ_CONSTANT) / (cost*cost);
            //dgdxt[node2*num_edges + i] = 0.0;
            dgdxt[head_node*num_edges + i] = 0.0;
        }
      }
  }
}

void get_costs6(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  float cost1, cost2;

  // FIXME: need to multiply card2 by another constant
  if (card1 < NILJ_MIN_CARD || card2 < NILJ_MIN_CARD) {
    cost1 = 1000000000.0;
    cost2 = card1*card2;
    costs[i] = card1*card2;
  } else {
    if (nilj[i] == 1) {
      ratio_mul = card3 / card2;
      if (ratio_mul < 1.00) ratio_mul = 1.00;
      ratio_mul *= RATIO_MUL_CONSTANT;
      cost1 = NILJ_CONSTANT2*card2*ratio_mul;
    } else if (nilj[i] == 2) {
      ratio_mul = card3 / card1;
      if (ratio_mul < 1.00) ratio_mul = 1.00;
      ratio_mul *= RATIO_MUL_CONSTANT;
      cost1 = NILJ_CONSTANT2*card1*ratio_mul;
    } else {
      // FIXME: for bushy plans, consider adding card1*card2 here
      printf("THIS SHOULD NOT HAVE HAPPENED\n");
      exit(-1);
    }
    cost2 = card1*card2;
    if (cost1 <= cost2) costs[i] = cost1;
    else costs[i] = cost2;
  }


  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
    exit(-1);
  } else if (normalization_type == 2) {
      if (cost1 <= cost2) {
        // index nested loop join
        // since ratio_mul*RATIO_MUL_CONSTANT always done
        if (ratio_mul > RATIO_MUL_CONSTANT) {
          // neither node1, or node2 play a role here
          dgdxT[node1*num_edges + i] = 0.0;
          dgdxT[node2*num_edges + i] = 0.0;
          dgdxT[head_node*num_edges + i] = - max_val / costs[i];
        } else {
          // card3 does not play a role anymore
          if (nilj[i]  == 1) {
              // only depends on card2
              dgdxT[node1*num_edges + i] = 0;
              dgdxT[node2*num_edges + i] = - max_val / costs[i];
          } else {
              dgdxT[node1*num_edges + i] = - max_val / costs[i];
              dgdxT[node2*num_edges + i] = 0;
          }
        }
      } else {
          // cost2 < cost1, implies cost = card1*card2
          // derivative of 1 / (e^{ax}*e^{ay}), where y is the other nodes
          // prediction (x = log(card1) / max_val, y = log(card2) / max_val)
          dgdxT[node1*num_edges + i] = - max_val / costs[i];
          dgdxT[node2*num_edges + i] = - max_val / costs[i];
          //printf("going to set both derivatives to: %f\n", -max_val/costs[i]);
          //dgdxT[node1*num_edges + i] = 0.0;
          //dgdxT[node2*num_edges + i] = 0.0;
      }
  }
}

void get_costs5(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];
  float cost1, cost2;

  // FIXME: need to multiply card2 by another constant
  if (nilj[i] == 1) {
    ratio_mul = card3 / card2;
    if (ratio_mul < 1.00) ratio_mul = 1.00;
    ratio_mul *= RATIO_MUL_CONSTANT;
    cost1 = NILJ_CONSTANT2*card2*ratio_mul;
  } else if (nilj[i] == 2) {
    ratio_mul = card3 / card1;
    if (ratio_mul < 1.00) ratio_mul = 1.00;
    ratio_mul *= RATIO_MUL_CONSTANT;
    cost1 = NILJ_CONSTANT2*card1*ratio_mul;
  } else {
    // FIXME: for bushy plans, consider adding card1*card2 here
    printf("THIS SHOULD NOT HAVE HAPPENED\n");
    exit(-1);
  }
  cost2 = card1*card2;
  if (cost1 <= cost2) costs[i] = cost1;
  else costs[i] = cost2;

  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
    exit(-1);
  } else if (normalization_type == 2) {
      if (cost1 <= cost2) {
        // index nested loop join
        if (ratio_mul > RATIO_MUL_CONSTANT) {
          // neither node1, or node2 play a role here
          dgdxT[node1*num_edges + i] = 0.0;
          dgdxT[node2*num_edges + i] = 0.0;
          dgdxT[head_node*num_edges + i] = - max_val / costs[i];
        } else {
          // card3 does not play a role anymore
          if (nilj[i]  == 1) {
              // only depends on card2
              dgdxT[node1*num_edges + i] = 0;
              dgdxT[node2*num_edges + i] = - max_val / costs[i];
          } else {
              dgdxT[node1*num_edges + i] = - max_val / costs[i];
              dgdxT[node2*num_edges + i] = 0;
          }
        }
      } else {
          // cost2 < cost1, implies cost = card1*card2
          // derivative of 1 / (e^{ax}*e^{ay}), where y is the other nodes
          // prediction (x = log(card1) / max_val, y = log(card2) / max_val)
          dgdxT[node1*num_edges + i] = - max_val / costs[i];
          dgdxT[node2*num_edges + i] = - max_val / costs[i];
      }
  }
}

void get_costs7(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  float cost1, cost2;

  float cost = card1*card2;
  costs[i] = cost;

  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
    exit(-1);
  } else if (normalization_type == 2) {
    dgdxT[node1*num_edges + i] = - max_val / costs[i];
    dgdxT[node2*num_edges + i] = - max_val / costs[i];
  }
}

void get_costs2(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    float *costs, float *dgdxT, int i, int head_node)
{
  // head node is the joined node
  double card1, card2, card3, nilj_cost, ratio_mul;
  int node1, node2;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  card3 = ests[head_node];

  // FIXME: need to multiply card2 by another constant
  if (nilj[i] == 1) {
    ratio_mul = card3 / card2;
    if (ratio_mul < 1.00) ratio_mul = 1.00;
    costs[i] = NILJ_CONSTANT2*card2*ratio_mul;
  } else if (nilj[i] == 2) {
    ratio_mul = card3 / card1;
    if (ratio_mul < 1.00) ratio_mul = 1.00;
    costs[i] = NILJ_CONSTANT2*card1*ratio_mul;
  } else {
    // FIXME: for bushy plans, consider adding card1*card2 here
    printf("THIS SHOULD NOT HAVE HAPPENED\n");
    exit(-1);
  }

  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
  } else if (normalization_type == 2) {
      // index nested loop join
      if (ratio_mul > 1.0) {
        // neither node1, or node2 play a role here
        dgdxT[node1*num_edges + i] = 0;
        dgdxT[node2*num_edges + i] = 0;
        dgdxT[head_node*num_edges + i] = - max_val / costs[i];
      } else {
        // card3 does not play a role anymore
        if (nilj[i]  == 1) {
            // only depends on card2
            dgdxT[node1*num_edges + i] = 0;
            dgdxT[node2*num_edges + i] = - max_val / costs[i];
        } else {
            dgdxT[node1*num_edges + i] = - max_val / costs[i];
            dgdxT[node2*num_edges + i] = 0;
        }
      }
  }
}

void get_costs3(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT, int i)
{
  double card1, card2, nilj_cost;
  int node1, node2, head_node, tail_node;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];

  // FIXME: need to multiply card2 by another constant
  if (nilj[i] == 1) {
    costs[i] = NILJ_CONSTANT2*card2;
  } else if (nilj[i] == 2) {
    costs[i] = NILJ_CONSTANT2*card1;
  } else {
    // FIXME: for bushy plans, consider adding card1*card2 here
    printf("THIS SHOULD NOT HAVE HAPPENED\n");
    exit(-1);
  }

  if (normalization_type == 1) {
    // FIXME:
    printf("handle pg total selectivity case for nested loop index\n");
  } else if (normalization_type == 2) {
      // index nested loop join
      if (nilj[i]  == 1) {
          // only depends on card2
          dgdxT[node1*num_edges + i] = 0;
          dgdxT[node2*num_edges + i] = - max_val / costs[i];
      } else {
          dgdxT[node1*num_edges + i] = - max_val / costs[i];
          dgdxT[node2*num_edges + i] = 0;
      }
  }
}

void get_costs4(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT, int i)
{
  double card1, card2, hash_join_cost, nilj_cost;
  int node1, node2, head_node, tail_node;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  if (nilj[i] == 1) {
    nilj_cost = card2 + NILJ_CONSTANT*card1;
  } else if (nilj[i] == 2) {
    nilj_cost = card1 + NILJ_CONSTANT*card2;
  } else {
    printf("should not have happened!\n");
    exit(-1);
  }
  costs[i] = nilj_cost;
  float cost = costs[i];

  /* time to compute gradients */
  if (normalization_type == 2) {
    // log normalization type
      // index nested loop join
      if (nilj[i]  == 1) {
          dgdxT[node1*num_edges + i] = - (max_val*card1*NILJ_CONSTANT) / (cost*cost);
          dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
      } else {
          //float num2 = card2*NILJ_CONSTANT;
          dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
          dgdxT[node2*num_edges + i] = - (max_val*card2*NILJ_CONSTANT) / (cost*cost);
      }
  }
}

void get_costs8(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT, int i)
{
  double card1, card2, hash_join_cost, nilj_cost;
  int node1, node2, head_node, tail_node;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  float cost = CARD_DIVIDER*card1 + CARD_DIVIDER*card2;
  costs[i] = cost;

  /* time to update the derivatives */
  //if (normalization_type == 0) continue;

  /* time to compute gradients */
  if (normalization_type == 1) {
    float total1 = totals[node1];
    float total2 = totals[node2];
    //- (a / (ax_1 + bx_2)**2)
    dgdxT[node1*num_edges + i] = - (total1 / (cost*cost));
    dgdxT[node2*num_edges + i] = - (total2 / (cost*cost));
  } else if (normalization_type == 2) {
    // log normalization type
    dgdxT[node1*num_edges + i] = - (max_val*card1*CARD_DIVIDER) / (cost*cost);
    dgdxT[node2*num_edges + i] = - (max_val*card2*CARD_DIVIDER) / (cost*cost);
  }
}

void get_costs1(float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    //int *edges_head, int *edges_tail,
    int *nilj, int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT, int i)
{
  double card1, card2, hash_join_cost, nilj_cost;
  int node1, node2, head_node, tail_node;
  node1 = edges_cost_node1[i];
  node2 = edges_cost_node2[i];
  card1 = ests[node1];
  card2 = ests[node2];
  hash_join_cost = card1 + card2;
  if (nilj[i] == 1) {
    nilj_cost = card2 + NILJ_CONSTANT*card1;
  } else if (nilj[i] == 2) {
    nilj_cost = card1 + NILJ_CONSTANT*card2;
  } else {
    nilj_cost = 10000000000.0;
  }
  if (hash_join_cost < nilj_cost) costs[i] = hash_join_cost;
  else costs[i] = nilj_cost;
  float cost = costs[i];

  /* time to update the derivatives */
  //if (normalization_type == 0) continue;

  /* time to compute gradients */
  if (normalization_type == 1) {
    float total1 = totals[node1];
    float total2 = totals[node2];
    if (hash_join_cost < nilj_cost) {
      //- (a / (ax_1 + bx_2)**2)
      dgdxT[node1*num_edges + i] = - (total1 / (hash_join_cost*hash_join_cost));
      dgdxT[node2*num_edges + i] = - (total2 / (hash_join_cost*hash_join_cost));
    } else {
        if (nilj[i] == 1) {
          float num1 = total1*NILJ_CONSTANT;
          dgdxT[node1*num_edges + i] = - (num1 / (costs[i]*costs[i]));
          dgdxT[node2*num_edges + i] = - (total2 / (costs[i]*costs[i]));
        } else {
          //# node 2
          //# - (a / (ax_1 + bx_2)**2)
          float num2 = total2*NILJ_CONSTANT;
          dgdxT[node1*num_edges + i] = - (total1 / (costs[i]*costs[i]));
          //# - (b / (ax_1 + bx_2)**2)
          dgdxT[node2*num_edges + i] = - (num2 / (costs[i]*costs[i]));
        }
    }
  } else if (normalization_type == 2) {
    // log normalization type
    if (hash_join_cost <= nilj_cost) {
        //- (ae^{ax} / (e^{ax} + e^{ax2})**2)
         //e^{ax} is just card1
        dgdxT[node1*num_edges + i] = - (max_val*card1) / (hash_join_cost*hash_join_cost);
        dgdxT[node2*num_edges + i] = - (max_val*card2) / (hash_join_cost*hash_join_cost);
    } else {
        // index nested loop join
        if (nilj[i]  == 1) {
            dgdxT[node1*num_edges + i] = - (max_val*card1*NILJ_CONSTANT) / (cost*cost);
            dgdxT[node2*num_edges + i] = - (max_val*card2) / (cost*cost);
        } else {
            //float num2 = card2*NILJ_CONSTANT;
            dgdxT[node1*num_edges + i] = - (max_val*card1) / (cost*cost);
            dgdxT[node2*num_edges + i] = - (max_val*card2*NILJ_CONSTANT) / (cost*cost);
        }
    }
  }
}

extern "C" void get_optimization_variables(
    float *ests, float *totals,
    double min_val, double max_val, int normalization_type,
    int *edges_cost_node1, int *edges_cost_node2,
    int *edges_head, int *edges_tail,
    int *nilj, float *edges_penalties,
    float *edges_read_costs, float *edges_rows_fetched,
    int num_nodes, int num_edges,
    // return arguments below, will be edited in place
    float *costs, float *dgdxT,
    float *G, float *Q, int cost_model)
{
  double card1, card2, hash_join_cost, nilj_cost, penalty;
  int node1, node2, head_node, tail_node;

  for (int i = 0; i < num_edges; i++) {
    head_node = edges_head[i];
    tail_node = edges_tail[i];
    if (edges_cost_node1[i] == SOURCE_NODE_CONST) {
      costs[i] = 1.0;
      // still need to construct the matrices stuff
      // tail node was the final node, which we don't have in Q / G
      Q[i*num_nodes + head_node] = 1 / costs[i];
      G[head_node*num_nodes + head_node] += 1 / costs[i];
      continue;
    }

    if (cost_model == 1) {
      get_costs1(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 2) {
      // nested_loop_index, considering the size of the joined node
      get_costs2(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 3) {
      // nested_loop_index
      get_costs3(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 4) {
      get_costs4(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 5) {
      // nested loop index considering nested loop join w/o indexes too
      get_costs5(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 6) {
      // nested_loop_index5
      get_costs6(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 7) {
      get_costs7(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 8) {
      get_costs8(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 9) {
      // this is exactly the same calculation as cost_model8 -- hash join
      // costs, without any constants
      get_costs8(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 10) {
      get_costs10(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i);
    } else if (cost_model == 11) {
      get_costs11(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 12) {
      get_costs12(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 13) {
      get_costs13(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 16) {
      get_costs16(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 15) {
      get_costs15(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node, edges_penalties);
    } else if (cost_model == 17) {
      get_costs17(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj, num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 18) {
      get_costs18(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj,
          edges_read_costs, edges_rows_fetched,
          num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 19) {
      get_costs19(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj,
          edges_read_costs, edges_rows_fetched,
          num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 20) {
      get_costs20(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj,
          edges_read_costs, edges_rows_fetched,
          num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    } else if (cost_model == 21) {
      get_costs21(ests, totals, min_val, max_val, normalization_type,
          edges_cost_node1, edges_cost_node2, nilj,
          edges_read_costs, edges_rows_fetched,
          num_nodes, num_edges,
          costs, dgdxT, i, head_node);
    }

    float cost = 1.0 / costs[i];

    /* construct G, Q */
    Q[i*num_nodes + head_node] = cost;
    G[head_node*num_nodes + head_node] += cost;

    if (tail_node != SOURCE_NODE_CONST) {
        Q[i*num_nodes + tail_node] = -cost;
        G[tail_node*num_nodes + tail_node] += cost;
        G[head_node*num_nodes + tail_node] -= cost;
        G[tail_node*num_nodes + head_node] -= cost;
    }
  }
}

