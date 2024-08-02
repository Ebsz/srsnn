use luna::models::srsnn::typed::{TypedModel, TypedConfig};
use luna::models::rsnn::{RSNNModel, RSNNConfig};

use evolution::algorithm::Algorithm;
use evolution::algorithm::snes::{SeparableNES, SNESConfig};
use evolution::algorithm::nes::{NES, NESConfig};

use utils::math;
use utils::random;
use utils::environment::Environment;

use ndarray::{array, Array1};

const MIN_EVAL: f32 = -0.01;


#[test]
fn test_snes() {
    let best = test_algorithm::<SeparableNES>(100, snes_conf());

    assert!(best > MIN_EVAL, "expected best eval > {}, got {}", MIN_EVAL, best);
}

#[test]
fn test_nes() {
    let best = test_algorithm::<NES>(100, nes_conf());

    assert!(best > MIN_EVAL, "expected best eval > {}, got {}", MIN_EVAL, best);
}


fn test_algorithm<A: Algorithm>(epochs: usize, conf: A::Config) -> f32 {
    let m_conf = typed_conf();

    let env = Environment {
        inputs: 1,
        outputs: 1,
    };

    let mut s = A::new::<RSNNModel<TypedModel>>(conf, &m_conf, &env);

    let epochs = 300;

    let mut best: Option<f32> = None;

    for t in 0..epochs {
        let ps = s.parameter_sets();
        let evals: Vec<f32> = ps.iter().map(|p| f(&p.linearize())).collect();
        s.step(evals.clone());

        let best_eval = math::maxf(&evals);

        best = match best {
            Some(e) => {
                println!("{e}");
                Some(math::maxf(&[e, best_eval])) }
            None    => { Some(best_eval) }
        };
    }

    best.unwrap()
}

// Fitness function
fn f(w: &Array1<f32>) -> f32 {
    let solution: Array1<f32> = array![0.5, 0.1, -0.3, 0.8, -1.1, 0.4,0.5, -1.0, 0.1, 1.55];

    -(solution - w).iter().map(|x| x*x).sum::<f32>()
}

fn snes_conf() -> SNESConfig {
    SNESConfig {
        pop_size: 50,
        lr_mu: 0.1, // PS: 1.0
        lr_sigma: 0.01, // PS: 0.01
    }
}

fn nes_conf() -> NESConfig {
    NESConfig {
        population_size: 50,
        alpha: 0.01,
        sigma: 0.1,
        init_mean: 0.0,
        init_stddev: 1.0
    }
}

fn typed_conf() -> RSNNConfig<TypedModel> {
    RSNNConfig {
        n: 64,
        model: TypedConfig {
            k: 1,
            max_w: 1.0,
        }
    }
}
