import numpy as np
import json
import pdb
import networkx as nx
import time
import matplotlib

import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import grandalf
from grandalf.layouts import SugiyamaLayout

from .utils import SOURCE_NODE
from evaluation.cost_model import update_subplan_costs

CROSS_JOIN_CARD = 19329323

def _find_all_tables(plan):
    '''
    '''
    # find all the scan nodes under the current level, and return those
    table_names = extract_values(plan, "Relation Name")
    alias_names = extract_values(plan, "Alias")
    table_names.sort()
    alias_names.sort()

    return table_names, alias_names

def explain_to_nx(explain):
    '''
    '''
    base_table_nodes = []
    join_nodes = []

    def _get_node_name(tables):
        name = ""
        if len(tables) > 1:
            name = str(deterministic_hash(str(tables)))[0:5]
            join_nodes.append(name)
        else:
            name = tables[0]
            if len(name) >= 6:
                # no aliases, shorten it
                name = "".join([n[0] for n in name.split("_")])
                if name in base_table_nodes:
                    name = name + "2"
            base_table_nodes.append(name)
        return name

    def _add_node_stats(node, plan):
        # add stats for the join
        G.nodes[node]["Plan Rows"] = plan["Plan Rows"]
        if "Actual Rows" in plan:
            G.nodes[node]["Actual Rows"] = plan["Actual Rows"]
        else:
            G.nodes[node]["Actual Rows"] = -1.0

        if "Node Type" in plan:
            G.nodes[node]["Node Type"] = plan["Node Type"]
        total_cost = plan["Total Cost"]
        G.nodes[node]["Total Cost"] = total_cost
        aliases = G.nodes[node]["aliases"]
        if len(G.nodes[node]["tables"]) > 1:
            children_cost = plan["Plans"][0]["Total Cost"] \
                    + plan["Plans"][1]["Total Cost"]

            # +1 to avoid cases which are very close
            # if not total_cost+1 >= children_cost:
                # print("aliases: {} children cost: {}, total cost: {}".format(\
                        # aliases, children_cost, total_cost))
                # pdb.set_trace()
            G.nodes[node]["cur_cost"] = total_cost - children_cost
            G.nodes[node]["node_label"] = plan["Node Type"][0]
            G.nodes[node]["scan_type"] = ""
        else:
            # FIXME: debug
            G.nodes[node]["cur_cost"] = total_cost
            G.nodes[node]["node_label"] = node
            # what type of scan was this?
            node_types = extract_values(plan, "Node Type")
            for i, full_n in enumerate(node_types):
                shortn = ""
                for n in full_n.split(" "):
                    shortn += n[0]
                node_types[i] = shortn

            scan_type = "\n".join(node_types)
            G.nodes[node]["scan_type"] = scan_type

    def traverse(obj):
        if isinstance(obj, dict):
            if "Plans" in obj:
                if len(obj["Plans"]) == 2:
                    # these are all the joins
                    left_tables, left_aliases = _find_all_tables(obj["Plans"][0])
                    right_tables, right_aliases = _find_all_tables(obj["Plans"][1])
                    if len(left_tables) == 0 or len(right_tables) == 0:
                        return
                    all_tables = left_tables + right_tables
                    all_aliases = left_aliases + right_aliases
                    all_aliases.sort()
                    all_tables.sort()

                    if len(left_aliases) > 0:
                        node0 = _get_node_name(left_aliases)
                        node1 = _get_node_name(right_aliases)
                        node_new = _get_node_name(all_aliases)
                    else:
                        node0 = _get_node_name(left_tables)
                        node1 = _get_node_name(right_tables)
                        node_new = _get_node_name(all_tables)

                    # update graph
                    G.add_edge(node_new, node0)
                    G.add_edge(node_new, node1)
                    G.edges[(node_new, node0)]["join_direction"] = "left"
                    G.edges[(node_new, node1)]["join_direction"] = "right"

                    # add other parameters on the nodes
                    G.nodes[node0]["tables"] = left_tables
                    G.nodes[node1]["tables"] = right_tables
                    G.nodes[node0]["aliases"] = left_aliases
                    G.nodes[node1]["aliases"] = right_aliases
                    G.nodes[node_new]["tables"] = all_tables
                    G.nodes[node_new]["aliases"] = all_aliases

                    # TODO: if either the left, or right were a scan, then add
                    # scan stats
                    _add_node_stats(node_new, obj)

                    if len(left_tables) == 1:
                        _add_node_stats(node0, obj["Plans"][0])
                    if len(right_tables) == 1:
                        _add_node_stats(node1, obj["Plans"][1])

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    traverse(v)

        elif isinstance(obj, list) or isinstance(obj,tuple):
            for item in obj:
                traverse(item)

    G = nx.DiGraph()
    traverse(explain)
    G.base_table_nodes = base_table_nodes
    G.join_nodes = join_nodes
    return G

NODE_COLORS = {}
# NODE_COLORS["Hash Join"] = 'b'
# NODE_COLORS["Merge Join"] = 'r'
# NODE_COLORS["Nested Loop"] = 'c'

NODE_COLORS["Index Scan"] = 'k'
NODE_COLORS["Seq Scan"] = 'k'
NODE_COLORS["Bitmap Heap Scan"] = 'k'

NODE_COLORS["Hash"] = 'k'
NODE_COLORS["Materialize"] = 'k'
NODE_COLORS["Sort"] = 'k'

# for signifying whether the join was a left join or right join
EDGE_COLORS = {}
EDGE_COLORS["left"] = "k"
EDGE_COLORS["right"] = "k"

def _plot_join_order_graph(G, base_table_nodes, join_nodes, pdf, title,
        fn):

    def format_ints(num):
        # returns the number formatted to closest 1000 + K
        return str(round(num, -3)).replace("000","") + "K"

    def _plot_labels(xdiff, ydiff, key, font_color, font_size):
        labels = {}
        label_pos = {}
        for k, v in pos.items():
            label_pos[k] = (v[0]+xdiff, v[1]+ydiff)
            if key in G.nodes[k]:
                if is_float(G.nodes[k][key]):
                    labels[k] = format_ints(G.nodes[k][key])
                else:
                    labels[k] = G.nodes[k][key]
            else:
                est_labels[k] = -1

        nx.draw_networkx_labels(G, label_pos, labels,
                font_size=font_size, font_color=font_color, ax=ax)

    fig,ax = plt.subplots(figsize=(8,7))
    NODE_SIZE = 600

    # FUCK fucking graphviz
    # pos = graphviz_layout(G, prog='dot')
    # pos = graphviz_layout(G, prog='dot',
            # args='-Gnodesep=0.05')

    # graphviz is better, but its is a bitch to install, so grandalf is also ok otherwise
    # G = G.reverse(copy=True)

    g = grandalf.utils.convert_nextworkx_graph_to_grandalf(G) # undocumented function
    class defaultview(object):
        w, h = 10, 10
    for v in g.V(): v.view = defaultview()
    sug = SugiyamaLayout(g.C[0])
    sug.init_all() # roots=[V[0]])
    # sug.init_all(roots=[g.V[0]],inverted_edges=[g.V[4].e_to(g.V[0])])
    # This is a bit of a misnomer, as grandalf doesn't actually come with any
    # visualization methods. This method instead calculates positions
    sug.draw()     # Extracts the positions
    pos = {v.data: (v.view.xy[0], v.view.xy[1]) for v in g.C[0].sV}

    # ugly hacks; want to draw the graph upside down than what grandalf gives
    # us (graphviz actually gave the correct layout...)
    ys = []
    levels = {}
    leveltoy = {}
    newlevels = {}
    for k,v in pos.items():
        ys.append(v[1])
    ys.sort()
    ys = np.unique(ys)
    level = 0
    for y in ys:
        levels[y] = level
        leveltoy[level] = y
        newlevels[y] = len(ys)-1-level
        level += 1
    pos2 = {}
    for k,v in pos.items():
        lv = newlevels[v[1]]
        newy = leveltoy[lv]
        pos2[k] = (v[0], newy)

    pos = pos2

    plt.title(title)
    color_intensity = [G.nodes[n]["cur_cost"] for n in G.nodes()]
    vmin = min(color_intensity)
    vmax = max(color_intensity)
    # cmap = 'viridis_r'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])

    nx.draw_networkx_nodes(G, pos,
               node_size=NODE_SIZE,
               node_color = color_intensity,
               cmap = cmap,
               alpha=0.2,
               vmin=vmin, vmax=vmax,
               ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin,
        vmax=vmax))

    sm._A = []
    plt.colorbar(sm, alpha=0.2, fraction=0.1, pad=0.0,
            label="PostgreSQL Estimated Cost")

    _plot_labels(0, -10, "est_card", "b", 8)
    _plot_labels(0, +10, "true_card", "darkorange", 8)
    _plot_labels(0, 0, "node_label", "k", 14)

    patch1 = mpatches.Patch(color='b', label='Estimated Cardinality')
    patch2 = mpatches.Patch(color='darkorange', label='True Cardinality')
    plt.legend(handles=[patch1,patch2])

    # TODO: shape of node based on scan types
    # _plot_labels(+25, +5, "scan_type", "b", 10)

    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    plt.xlim(x_min - x_margin, x_max + x_margin)

    edge_colors = []
    for edge in G.edges():
        edge_colors.append(EDGE_COLORS[G.edges[edge]["join_direction"]])

    nx.draw_networkx_edges(G, pos, width=1.0,
            alpha=1.0,  arrows=False,
            edge_color=edge_colors, ax=ax)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig()
    elif fn is not None:
        plt.savefig(fn)
    else:
        plt.show()

    plt.close()

def plot_explain_join_order(explain, true_cardinalities,
        est_cardinalities, pdf, title, fn=None):
    '''
    @true_cardinalities: dict for this particular explain
    '''
    G = explain_to_nx(explain)
    for node in G.nodes():
        aliases = G.nodes[node]["aliases"]
        aliases.sort()
        card_key = " ".join(aliases)
        if true_cardinalities is None:
            G.nodes[node]["est_card"] = G.nodes[node]["Plan Rows"]
            G.nodes[node]["true_card"] = G.nodes[node]["Actual Rows"]
        elif card_key in true_cardinalities:
            G.nodes[node]["est_card"] = est_cardinalities[card_key]
            G.nodes[node]["true_card"] = true_cardinalities[card_key]
        elif tuple(aliases) in true_cardinalities:
            G.nodes[node]["est_card"] = est_cardinalities[tuple(aliases)]
            G.nodes[node]["true_card"] = true_cardinalities[tuple(aliases)]
        else:
            # unknown, might be a cross-join?
            G.nodes[node]["est_card"] = CROSS_JOIN_CARD
            G.nodes[node]["true_card"] = CROSS_JOIN_CARD
            # pdb.set_trace()

    _plot_join_order_graph(G, G.base_table_nodes, G.join_nodes, pdf, title, fn)
    return G

def draw_plan_graph(subsetg, y, cost_model, ax=None,
        source_node=SOURCE_NODE, final_node=None, font_size=40,
        cbar_fontsize=24, cax=None, fig=None, width=None,
        edge_color=None,
        bold_opt_path=True, bold_path=None):

    for n in subsetg.nodes():
        joined = " \Join ".join(n)
        joined = "$" + joined + "$"
        subsetg.nodes()[n]["label"] = joined

    if y is not None and cost_model is not None:
        cost_key = "tmp_cost"
        subsetg = subsetg.reverse()
        tcost = update_subplan_costs(subsetg, cost_model,
                cost_key=cost_key, ests=y)

        # TODO: need to add the flow-loss computing module
        # flows, edges = get_flows(subsetg, cost_model+cost_key)
        # Flow-Loss specific widths
        # MIN: 2...6
        # MIN_WIDTH = 1.0
        # MAX_WIDTH = 30.0
        # NEW_RANGE = MAX_WIDTH - MIN_WIDTH
        # OLD_RANGE = max(flows) - min(flows)

        # edge_widths = {}
        # for i, x in enumerate(flows):
            # normx = (((x - min(flows))*NEW_RANGE) / OLD_RANGE) + MIN_WIDTH
            # edge_widths[edges[i]] = normx
        # widths = []
        # for edge in subsetg.edges():
            # key = tuple([edge[1], edge[0]])
            # widths.append(edge_widths[key])

        # reverse back
        subsetg = subsetg.reverse()
        widths = []
        for edge in subsetg.edges():
            key = tuple([edge[1], edge[0]])
            widths.append(1.0)

        edge_colors = []
        for edge in subsetg.edges(data=True):
            edge_colors.append(edge[2][cost_model+cost_key])

        vmin = min(edge_colors)
        vmax = max(edge_colors)

        # assert len(edge_colors) == len(flows)
        opt_labels_list = nx.shortest_path(subsetg, source_node,
                final_node, weight=cost_model+cost_key)
        opt_labels = {}
        for n in subsetg.nodes(data=True):
            if n[0] in opt_labels_list:
                opt_labels[n[0]] = n[1]["label"]

        cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

    else:
        widths = []
        for edge in subsetg.edges():
            key = tuple([edge[1], edge[0]])
            widths.append(2.0)
        cm = None

    pos = nx.nx_pydot.pydot_layout(subsetg, prog="dot")

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(30,20))

    labels = nx.get_node_attributes(subsetg, 'label')

    nx.draw_networkx_labels(subsetg, pos=pos,
            labels=labels,
            ax=ax, font_size=font_size,
            bbox=dict(facecolor="w", edgecolor='k', boxstyle='round,pad=0.1'))

    if bold_opt_path and cost_model is not None:
        nx.draw_networkx_labels(subsetg, pos=pos,
                labels=opt_labels, ax=ax,
                font_size=font_size,
                bbox=dict(facecolor="w", edgecolor='k',
                lw=font_size/2, boxstyle='round,pad=0.5', fill=True))

    if bold_path and cost_model is not None:
        bold_labels = {}
        for n in subsetg.nodes(data=True):
            if n[0] in bold_path:
                bold_labels[n[0]] = n[1]["label"]
        nx.draw_networkx_labels(subsetg, pos=pos,
                labels=bold_labels, ax=ax,
                font_size=font_size,
                bbox=dict(facecolor="w", edgecolor='k',
                lw=font_size/2, boxstyle='round,pad=0.5', fill=True))

    if edge_color is not None:
        edge_colors = edge_color

    edges = nx.draw_networkx_edges(subsetg, pos,
            edge_color=edge_colors,
            width=widths, ax = ax, edge_cmap=cm,
            arrows=True,
            arrowsize=font_size / 2,
            arrowstyle='simple',
            min_target_margin=5.0)

    if y is not None and cost_model is not None:
        plt.style.use("seaborn-white")
        sm = plt.cm.ScalarMappable(cmap=cm,
                norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        if fig is None:
            cbar = plt.colorbar(sm, aspect=50,
                    orientation="horizontal", pad =
                    0.02)
        else:
            cbar = fig.colorbar(sm, ax=ax,
                    pad = 0.02,
                    aspect=50,
                    orientation="horizontal")

        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label("Cost", fontsize=font_size)
        cbar.ax.xaxis.get_offset_text().set_fontsize(font_size)

    plt.tight_layout()
