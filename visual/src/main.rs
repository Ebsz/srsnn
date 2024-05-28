use visual::window::{DrawContext, Window};
use visual::window::WindowComponent;

use visual::fg::ForceGraphComponent;


const WINDOW_SIZE: (u32, u32) = (1200, 800);

fn main() {
    let components: Vec<Box<dyn WindowComponent>> = vec![
        Box::new(ForceGraphComponent::new(10, vec![]))
    ];

    let mut w = Window::new(WINDOW_SIZE, components);

    println!("Running..");
    w.run();
}
