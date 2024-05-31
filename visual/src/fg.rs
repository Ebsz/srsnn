use crate::window::{DrawContext, WindowComponent};

use utils::random;

use force_graph::{ForceGraph, NodeData};

use sdl2::pixels::Color;


const INIT_SPREAD: f32 = 100.0;

const VERTICE_COLOR: Color = Color::RGB(200, 150, 135);
const EDGE_COLOR: Color = Color::RGB(255,255,255);

pub struct ForceGraphComponent {
    pub graph: ForceGraph,
}

impl ForceGraphComponent {
    const NODE_SIZE: i16 = 20;

    pub fn new(n: usize, edges: Vec<(u32, u32)>) -> ForceGraphComponent {
        let mut graph = <ForceGraph>::new(Default::default());

        let mut ids = Vec::new();

        for _ in 0..n {
            let id = graph.add_node(NodeData {
                x: random::random_range((-INIT_SPREAD, INIT_SPREAD)),
                y: random::random_range((-INIT_SPREAD, INIT_SPREAD)),
                ..Default::default()
            });
            ids.push(id);
        }

        for (id1, id2) in edges {
            graph.add_edge(ids[id1 as usize], ids[id2 as usize], Default::default());
        }

        ForceGraphComponent {
            graph
        }
    }
}

impl WindowComponent for ForceGraphComponent {
    fn update(&mut self) {
        self.graph.update(0.01);
    }

    fn render(&self, context: &mut DrawContext) {
        let x_offset: i16 = (context.size.0 / 2) as i16;
        let y_offset: i16 = (context.size.1 / 2) as i16;

        self.graph.visit_edges(|n1, n2, _edge| {
            let x1 = n1.x() as i16 + x_offset;
            let y1 = n1.y() as i16 + y_offset;

            let x2 = n2.x() as i16 + x_offset;
            let y2 = n2.y() as i16 + y_offset;

            context.draw_line((x1,y1),(x2,y2), 4, EDGE_COLOR);

        });

        self.graph.visit_nodes(|node| {
            let x = node.x() as i16 + x_offset;
            let y = node.y() as i16 + y_offset;

            context.draw_circle(x, y, Self::NODE_SIZE, VERTICE_COLOR);
        });
    }
}
