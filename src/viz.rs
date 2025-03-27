use crate::{left_right::LeftRight, Net, Ptr, PtrTag, Redex};
use bilge::prelude::u29;
use petgraph::{dot::Dot, Directed};
use std::collections::HashSet;

type VizGraph = petgraph::stable_graph::StableGraph<String, String, Directed>;

fn viz_graph(viz: &VizGraph) -> String {
    Dot::new(viz).to_string()
}
pub fn mem_to_dot(mem: &Net) -> String {
    let active_nodes = mem.redex.iter().map(|x| &x.0).flatten();
    let color_active_nodes = active_nodes
        .map(|Redex(x, y)| {
            format!(
                "A{0}_{1} [color = \"red\"]\n\
                {0} [color = \"red\"]\n\
                {1} [color = \"red\"]\n\
                A{0}_{1} -> {0}\n\
                A{0}_{1} -> {1}\n",
                x.slot_usize(),
                y.slot_usize()
            )
        })
        .chain(["\n}".to_owned()])
        .fold(String::new(), |mut acc, x| {
            acc.push_str(&x);
            acc
        });
    let mut dot_output = viz_graph(&mem_to_graph(mem));
    dot_output.truncate(dot_output.len() - 2);
    dot_output.push_str(&color_active_nodes);
    dot_output
}

fn mem_to_graph(net: &Net) -> VizGraph {
    let mut graph = petgraph::stable_graph::StableGraph::new();
    let mut empty = Vec::new();
    // From, to
    let mut visited_from_to = HashSet::<((u29, LeftRight), Ptr)>::new();
    let mut to_visit = vec![];
    for (i, node) in net.nodes.0.iter().enumerate() {
        graph.add_node(format!("{}", i));
        if node.left == Ptr::EMP() && node.right == Ptr::EMP() {
            empty.push(i);
        }
    }

    for tip in net
        .redex
        .iter()
        .flat_map(|x| x.0.iter().flat_map(|x| [x.0, x.1]))
    {
        to_visit.push(tip);
        // Visit over the tree starting from the tip.
        while let Some(current) = to_visit.pop() {
            // if let Some(parent) = parent {
            //     visited_from_to.insert((parent, next));
            // }
            match net.read(current) {
                crate::Either::A(node) => {
                    // The node at has an edge where its two aux ptrs go.
                    // If either is an `IN` that isn't an outgoing, hopefully that will be connected later by the graph traversal.
                    if node.left != Ptr::IN() {
                        visited_from_to.insert(((current.slot(), LeftRight::Left), node.left));
                        to_visit.push(node.left);
                    }
                    if node.right != Ptr::IN() {
                        visited_from_to.insert(((current.slot(), LeftRight::Right), node.right));
                        to_visit.push(node.right);
                    }
                }
                crate::Either::B(aux) => {
                    // The node at `next.slot()` either has an edge terminating at a `Ptr::IN` of `aux` (which is already recorded by case A) or
                    // is redirected by where `aux` points.
                    if aux != Ptr::IN() {
                        visited_from_to.insert(((current.slot(), current.tag().aux_side()), aux));
                        to_visit.push(aux);
                    }
                }
            }
        }
    }

    for (from, to) in visited_from_to {
        graph.add_edge(
            from.0.value().into(),
            to.slot_u32().into(),
            format!("{}{}->{}{}", from.0, from.1, to.slot(), to.tag()),
        );
        if let Some(x) = graph.node_weight_mut(to.slot_u32().into()) {
            if !to.tag().is_aux() {
                *x = format!("{}{}", to.slot_u32(), to.tag());
            }
        }
    }
    for to in net
        .redex
        .iter()
        .flat_map(|x| x.0.iter())
        .flat_map(|Redex(x, y)| [x, y])
    {
        let w = graph.node_weight_mut(to.slot_u32().into()).unwrap();
        if !w.is_empty() {
            w.push_str(",");
        }
        w.push_str(&format!("{}", to.tag()));
    }

    /*     type EdgeLabel = i64;
    // node_idx, port
    type NodeIdx = Vec<(usize, u8)>;
    let mut map = HashMap::<EdgeLabel, NodeIdx>::new();
    let mut graph = petgraph::Graph::default();

    for (i, node) in mem.iter().filter(|x| x[0] != EMPTY).enumerate() {
        let node_label = match node[0] {
            crate::inet::REDIRECT => 'R',
            crate::inet::ERAS => 'ε',
            crate::inet::LAM => 'λ', // ζ
            crate::inet::APP => '@', // 'ζ',
            crate::inet::DUP => 'δ',
            i => panic!("invalid tag in graph: {i}"),
        };
        graph.add_node(format!("{node_label}{i}"));
        for (port_num, edge_label) in (if node[0] == crate::inet::REDIRECT {
            &node[1..3]
        } else {
            &node[1..]
        })
        .into_iter()
        .enumerate()
        {
            let port_num: Port = port_num.try_into().unwrap();
            match map.entry(edge_label.abs()) {
                std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                    occupied_entry.get_mut().push((i, port_num));
                    if occupied_entry.get().iter().len() > 2 {
                        println!("malformed mem: {edge_label} has >2 connected nodes")
                    }
                }
                std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(vec![(i, port_num)]);
                }
            }
        }
    }

    // Add edges
    for nodes in map.values() {
        for node_i in nodes.iter() {
            for node_j in nodes.iter() {
                if node_i >= node_j {
                    // don't duplicate edges between nodes
                    continue;
                }
                graph.add_edge(
                    node_i.0.into(),
                    node_j.0.into(),
                    format!("{}-{}", node_i.1, node_j.1),
                );
            }
        }
    } */
    for node in empty {
        graph.remove_node(u32::try_from(node).unwrap().into());
    }

    graph
}
