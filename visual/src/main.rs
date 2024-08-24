use visual::tasks::TaskRenderer;

use tasks::{Task, TaskInput};
use tasks::catching_task::{CatchingTask, CatchingTaskSetup};
use tasks::movement_task::{MovementTask, MovementTaskSetup};
use tasks::survival_task::{SurvivalTask, SurvivalTaskSetup};
use tasks::pole_balancing_task::{PoleBalancingTask, PoleBalancingSetup};

use ndarray::Array;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};

use sdl2::Sdl;
use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::render::WindowCanvas;
use sdl2::keyboard::Keycode;

use std::collections::HashMap;
use std::time::Instant;


fn main() {
    const TASKNAME: &str = "pole_balancing";

    let mut input_map: HashMap<Keycode, u32> = HashMap::new();

    match TASKNAME {
        "movement" => {
            input_map.insert(Keycode::D, 0);
            input_map.insert(Keycode::A, 1);
            input_map.insert(Keycode::W, 2);
            input_map.insert(Keycode::S, 3);

            let task = MovementTask::new(&MovementTaskSetup {});

            let mut t = TaskTester::new(task, input_map);
            t.run();

        },
        "catching" => {
            input_map.insert(Keycode::D, 0);
            input_map.insert(Keycode::A, 1);

            let mut rng = StdRng::seed_from_u64(0);
            let size = CatchingTask::render_size().0;
            let x: i32 = rng.gen_range(0..size);

            let task = CatchingTask::new(&CatchingTaskSetup {
                target_pos: x
            });

            let mut t = TaskTester::new(task, input_map);
            t.run();
        }
        "survival" => {
            input_map.insert(Keycode::A, 0);
            input_map.insert(Keycode::D, 1);
            input_map.insert(Keycode::W, 2);
            input_map.insert(Keycode::S, 3);

            let task = SurvivalTask::new(&SurvivalTaskSetup {food_spawn_rate: 30});

            let mut t = TaskTester::new(task, input_map);
            t.run();
        },
        "pole_balancing" => {
            input_map.insert(Keycode::A, 0);
            input_map.insert(Keycode::D, 1);

            let task = PoleBalancingTask::new(&PoleBalancingSetup {});

            let mut t = TaskTester::new(task, input_map);
            t.run();

        }
        _ => { println!("Unknown task"); }
    }
}

/// Enables running a task with human input, which is
/// very useful when developing and testing tasks.
pub struct TaskTester<T: Task + TaskRenderer>
{
    canvas: WindowCanvas,
    context: Sdl,
    task: T,
    input_map: HashMap<Keycode, u32>,
    finished: bool,
}

impl<T: Task + TaskRenderer> TaskTester<T>

{
    const TARGET_FPS: u128 = 60;

    pub fn new(task: T, input_map: HashMap<Keycode, u32>) -> TaskTester<T> {
        let sdl_context = sdl2::init().unwrap();
        let video_subsystem = sdl_context.video().unwrap();

        let size = T::render_size();

        let window = video_subsystem.window("window", size.0 as u32, size.1 as u32)
            .position_centered()
            .build()
            .unwrap();

        let canvas = window.into_canvas().build().unwrap();

        TaskTester {
            context: sdl_context,
            canvas,
            task,
            input_map,
            finished: false,
        }
    }

    pub fn run(&mut self) {
        let mut task_input: Vec<u32> = Vec::new();

        self.canvas.clear();
        self.canvas.present();

        let mut event_pump = self.context.event_pump().unwrap();

        let mut now = Instant::now();
        let mut last = now;
        let mut delta;

        let mut acc: u128 = 0;
        let mut ticks = 0;

        let mut print_acc: u128 = 0;

        let mut running: bool = true;
        while running {
            now = Instant::now();
            delta = now-last;

            last = now;
            acc += delta.as_nanos();
            print_acc += delta.as_nanos();

            if acc > 1_000_000_000 / Self::TARGET_FPS {
                self.update(&task_input);

                task_input.clear();

                ticks += 1;
                acc -= 1_000_000_000 / Self::TARGET_FPS;
            }

            if print_acc >= 1_000_000_000 {
                println!("ticks: {ticks}");
                ticks = 0;
                print_acc = 0;
            }

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit {..} |
                    Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
                        running = false;
                    },
                    Event::KeyDown {keycode, ..} => {
                        match self.input_map.get(&keycode.unwrap()) {
                            Some(id) => { task_input.push(*id); },
                            None => {}
                        }
                    }
                    _ => {}
                }
            }

            self.render();
            self.canvas.present();
        }
    }

    fn update(&mut self, input: &Vec<u32>) {

        if !self.finished {
            let state = self.task.tick(TaskInput {data: input.clone() });

            if let Some(_) = state.result {
                self.finished = true;
            }
        }
    }

    fn render(&mut self) {
        self.canvas.set_draw_color(Color::RGB(255, 255, 255));
        self.canvas.clear();

        self.task.render(&mut self.canvas);
    }
}

//use visual::window::Window;
//use visual::window::WindowComponent;
//
//use visual::fg::ForceGraphComponent;
//
//fn graph_window() {
//    let components: Vec<Box<dyn WindowComponent>> = vec![
//        Box::new(ForceGraphComponent::new(10, vec![(0,1), (2,2), (1,2)], None))
//    ];
//
//    let mut w = Window::new(WINDOW_SIZE, components);
//
//    println!("Running..");
//    w.run();
//
//}
