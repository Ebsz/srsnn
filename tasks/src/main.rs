//! tasks/src/main.rs

use tasks::{Task, TaskName, TaskInput, TaskRenderer};
use tasks::catching_task::{CatchingTask, CatchingTaskConfig, ARENA_SIZE};
use tasks::movement_task::{MovementTask, MovementTaskConfig};

use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};

use sdl2::Sdl;
use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::render::WindowCanvas;
use sdl2::keyboard::Keycode;

use std::collections::HashMap;
use std::time::Instant;


fn main () {
    const taskname: TaskName = TaskName::CatchingTask;
    let mut input_map: HashMap<Keycode, i32> = HashMap::new();

    match taskname {
        TaskName::MovementTask => {
            input_map.insert(Keycode::D, 0);
            input_map.insert(Keycode::A, 1);
            input_map.insert(Keycode::W, 2);
            input_map.insert(Keycode::S, 3);

            let task = MovementTask::new(MovementTaskConfig {});

            let mut t = TaskTester::new(task, input_map);
            t.run();

        },
        TaskName::CatchingTask => {
            input_map.insert(Keycode::D, 0);
            input_map.insert(Keycode::A, 1);

            let mut rng = StdRng::seed_from_u64(0);
            let x: i32 = rng.gen_range(0..ARENA_SIZE.0);

            let task = CatchingTask::new(CatchingTaskConfig {
                target_pos: x
            });

            let mut t = TaskTester::new(task, input_map);
            t.run();
        }
    }

}

/// Enables running a task with human input, which is
/// very useful for testing tasks.
pub struct TaskTester<T: Task + TaskRenderer> {
    canvas: WindowCanvas,
    context: Sdl,
    task: T,
    input_map: HashMap<Keycode, i32>,
    finished: bool
}

impl<T: Task + TaskRenderer> TaskTester<T> {
    const TARGET_FPS: u128 = 60;

    pub fn new(task: T, input_map: HashMap<Keycode, i32>) -> TaskTester<T> {
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
        let mut task_input: Vec<TaskInput> = Vec::new();

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
                            Some(id) => { task_input.push(TaskInput {input_id: *id}); },
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

    fn update(&mut self, input: &Vec<TaskInput>) {
        if !self.finished {
            let state = self.task.tick(input);

            if let Some(r) = state.result {
                println!("{:?}", r);
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
