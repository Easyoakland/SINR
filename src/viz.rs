use crate::{
    left_right::LeftRight,
    net::{Either, Net},
    redex::Redex,
    Ptr, PtrTag, Slot,
};
use petgraph::{dot::Dot, Directed};
use std::collections::HashSet;

type VizGraph = petgraph::stable_graph::StableGraph<String, String, Directed>;

fn viz_graph(viz: &VizGraph) -> String {
    Dot::new(viz).to_string()
}
pub fn mem_to_dot(mem: &Net) -> String {
    let erasing_nodes = mem
        .redexes
        .erase
        .0
        .iter()
        .map(|x| {
            format!(
                "\neps{0} [label = \"ε\", color = \"red\"]\n\
                {0} [color = \"red\"]\n\
                eps{0} -> {0} [label = \"{1}{2}\"]\n",
                x.slot_usize(),
                x.slot(),
                x.tag()
            )
        })
        .collect::<String>();
    let active_nodes = mem.redexes.regular.iter().map(|x| &x.0).flatten();
    let color_active_nodes = active_nodes
        .map(|Redex(x, y)| {
            format!(
                "A{0}_{1} [color = \"red\"]\n\
                {0} [color = \"red\"]\n\
                {1} [color = \"red\"]\n\
                A{0}_{1} -> {0}[label=\"{2}{3}\"]\n\
                A{0}_{1} -> {1}[label=\"{4}{5}\"]\n",
                x.slot_usize(),
                y.slot_usize(),
                x.slot(),
                x.tag(),
                y.slot(),
                y.tag()
            )
        })
        .chain(["\n}".to_owned()])
        .collect::<String>();
    let mut dot_output = viz_graph(&mem_to_graph(mem));
    dot_output.truncate(dot_output.len() - 2);
    dot_output.push_str(&erasing_nodes);
    dot_output.push_str(&color_active_nodes);
    dot_output
}

fn mem_to_graph(net: &Net) -> VizGraph {
    let mut graph = petgraph::stable_graph::StableGraph::new();
    let mut empty = Vec::new();
    // From, to
    let mut visited_from_to = HashSet::<((Slot, LeftRight), Ptr)>::new();
    let mut to_visit = vec![];
    for (i, node) in net.nodes.0.iter().enumerate() {
        let node = node.get();
        graph.add_node(format!("{}", i));
        if node.left == Ptr::EMP() && node.right == Ptr::EMP() {
            empty.push(i);
        }
    }

    for tip in net
        .redexes
        .regular
        .iter()
        .flat_map(|x| x.0.iter().flat_map(|x| [x.0, x.1]))
        .chain(net.redexes.erase.0.iter().copied())
    {
        if tip == Ptr::EMP() {
            panic!()
        }
        to_visit.push(tip);
        // Visit over the tree starting from the tip.
        while let Some(current) = to_visit.pop() {
            match net.read(current) {
                Either::A(node) => {
                    // The node at has an edge where its two aux ptrs go.
                    // If either is an `IN` that isn't an outgoing, hopefully that will be connected later by the graph traversal.
                    if node.left != Ptr::IN() {
                        visited_from_to.insert(((current.slot(), LeftRight::Left), node.left));
                        if current.tag() != PtrTag::Era {
                            to_visit.push(node.left);
                        }
                    }
                    if node.right != Ptr::IN() {
                        visited_from_to.insert(((current.slot(), LeftRight::Right), node.right));
                        if current.tag() != PtrTag::Era {
                            to_visit.push(node.right);
                        }
                    }
                }
                Either::B(aux) => {
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
            u32::try_from(from.0.value()).unwrap().into(),
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
        .redexes
        .regular
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
    // for to in net.redexes.erase.0.iter() {
    //     let w = graph.node_weight_mut(to.slot_u32().into()).unwrap();
    //     if !w.is_empty() {
    //         w.push_str(", ");
    //     }
    //     w.push_str("redex ε");
    // }

    // for node in empty {
    //     graph.remove_node(u32::try_from(node).unwrap().into());
    // }

    graph
}
