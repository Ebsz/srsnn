pub mod plots;
pub mod window;
pub mod tasks;
pub mod task_window;
pub mod fg;

use crate::window::{Window, WindowComponent};
use crate::task_window::TaskWindow;
use crate::fg::ForceGraphComponent;

use model::neuron::NeuronModel;
use model::network::representation::{NetworkRepresentation, NeuronDescription, NeuronRole};
use model::network::builder::NetworkBuilder;

use sdl2::pixels::Color;

use ndarray::Array1;


const PLOTS_DIR: &str = "plots";

const INPUT_COLOR: Color = Color::RGB(240, 100, 30);
const OUTPUT_COLOR: Color = Color::RGB(100, 240, 30);
const NETWORK_COLOR: Color = Color::RGB(200, 150, 135);


fn role_to_color(r: NeuronRole) -> Color {
    match r {
        NeuronRole::Input => INPUT_COLOR,
        NeuronRole::Output => OUTPUT_COLOR,
        NeuronRole::Network => NETWORK_COLOR,
    }
}

pub fn visualize_network_representation<N: NeuronModel>(repr: &NetworkRepresentation<NeuronDescription<N>>) {
    let edges = repr.edges();
    let colors: Array1<Color> = repr.neurons.map(|n| role_to_color(n.role));

    let components: Vec<Box<dyn WindowComponent>> = vec![
        Box::new(ForceGraphComponent::new(repr.n, edges, Some(colors)))
    ];

    let mut w = Window::new(Window::DEFAULT_WINDOW_SIZE, components);

    w.run();
}

pub fn visualize_network(n: usize, edges: Vec<(u32, u32)>) {
    let components: Vec<Box<dyn WindowComponent>> = vec![
        Box::new(ForceGraphComponent::new(n, edges, None))
    ];

    let mut w = Window::new(Window::DEFAULT_WINDOW_SIZE, components);

    w.run();
}
