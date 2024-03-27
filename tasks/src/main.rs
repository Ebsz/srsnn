use tasks::cognitive_task::{CognitiveTask, TaskInput, TaskAgent, TaskState};
use tasks::catching_task::{CatchingTask, CatchingTaskConfig, SENSOR_LEN, ARENA_SIZE};

use macroquad::prelude::*;

use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};

use sdl2::gfx::primitives::DrawRenderer;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;

use tasks::window::{TaskWindow, TaskRenderer};


//struct CatchingRenderer {}
//
//impl TaskRenderer for CatchingRenderer {
//
//    fn render(&self, canvas: &mut WindowCanvas) {
//        canvas.set_draw_color(Color::RGB(200,155,255));
//        canvas.filled_circle(100, 100, 30, Color::RGB(0,0,0));
//    }
//}


struct HumanAgent {}

impl TaskAgent for HumanAgent {
    fn step(&mut self, task_state: &TaskState) -> Vec<TaskInput> {
        vec![TaskInput{input_id: 1}]
    }
}


fn main () {
    let mut rng = StdRng::seed_from_u64(0);
    let x: i32 = rng.gen_range(0..ARENA_SIZE.0);

    let mut task = CatchingTask::new(CatchingTaskConfig {
        target_pos: x});

    let mut window = TaskWindow::new(task, (500,600));

    window.run();
}



//#[macroquad::main("Apple")]
//async fn main() {
//
//    let mut rng = StdRng::seed_from_u64(0);
//    let x: i32 = rng.gen_range(0..ARENA_SIZE.0);
//
//    let mut task = CatchingTask::new(CatchingTaskConfig {
//        target_pos: x});
//
//    println!("{:?}", (task.agent.x, task.agent.y));
//
//    request_new_screen_size(ARENA_SIZE.0 as f32,ARENA_SIZE.1 as f32);
//    set_fullscreen(false);
//
//    let mut finished: bool = false;
//
//    let mut task_input: Vec<TaskInput> = Vec::new();
//
//    loop {
//        if !finished {
//            let res = task.tick(&task_input);
//
//            if let Some(r) = res.result {
//                println!("{:?}", r);
//                finished = true;
//            }
//        }
//
//        task_input.clear();
//
//        if is_key_down(KeyCode::A) {
//            task_input.push(TaskInput{input_id: 1});
//        }
//        if is_key_down(KeyCode::D) {
//            task_input.push(TaskInput{input_id: 0});
//        }
//
//        clear_background(LIGHTGRAY);
//
//        draw_rectangle(0.0,0.0, ARENA_SIZE.0 as f32, ARENA_SIZE.1 as f32, WHITE);
//
//        // Draw apple
//        draw_circle(task.apple.x as f32,
//            task.apple.y as f32,
//            task.apple.r as f32,
//            RED);
//
//        // Draw Sensors
//        let agent_pos = (task.agent.x as f32, task.agent.y as f32);
//
//        for s in &task.sensors {
//            //let x = pos.0 + s.angle.cos() * SENSOR_LEN;
//            //let y = pos.1 - s.angle.sin() * SENSOR_LEN;
//
//            let sensor_endpoint = s.endpoint(agent_pos);
//
//            draw_line(agent_pos.0, agent_pos.1, sensor_endpoint.0, sensor_endpoint.1, 2.0, BLACK);
//
//        }
//
//        // Draw agent
//        draw_circle(task.agent.x as f32,
//            task.agent.y as f32,
//            task.agent.r as f32,
//            BLUE);
//
//
//
//        next_frame().await
//    }
//}
