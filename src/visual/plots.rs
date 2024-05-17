use plotters::prelude::*;

use visual::plots::plot_single_variable;

use ndarray::{s, Array, Array1, Array2};
use model::record::{Record, RecordType, RecordDataType};

use evolution::stats::EvolutionStatistics;


pub fn generate_plots(record: &Record) {
    // Potentials
    let single_pot = record.get_potentials().iter().map(|x| x[0]).collect();

    let plot_ok = plot_single_neuron_potential(single_pot);
    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }

    // Spikes
    let mut spikedata: Vec<Array1<f32>> = vec![];

    for i in record.get(RecordType::Spikes) {
        if let RecordDataType::Spikes(s) = i {
            spikedata.push(s.clone());
        } else {
            panic!("Error parsing spike records");
        }
    }

    let plot_ok = plot_spikes(spikedata.clone());

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }

    let mut spike_array: Array2<f32> = Array::zeros((spikedata.len(), spikedata[0].shape()[0]));

    for (i, p) in spikedata.iter().enumerate() {
        //println!("{:?}, {:?}", i, p);
        spike_array.slice_mut(s!(i,..)).assign(&p);
    }

    //let psth = crate::analysis::to_firing_rate(spike_array);
    //plot_firing_rates(psth);
}

pub fn plot_evolution_stats(stats: &EvolutionStatistics) {
    log::info!("Plotting evolution stats");

    let _ = plot_single_variable(stats.generation_best_fitness.clone(), "Generation best", "Evolution", "evolution_best.png", &BLUE);
    let _ = plot_single_variable(stats.generation_mean_fitness.clone(), "Generation mean", "Evolution", "evolution_mean.png", &BLUE);
}

pub fn plot_network_energy(energy: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plot_single_variable(energy, "Energy", "Energy", "energy.png", &RED)
}

pub fn plot_single_neuron_potential(potentials: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    plot_single_variable(potentials, "Potential", "Potentials", "pots.png", &BLACK)
}


pub fn plot_firing_rates(data: Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let filename = "firing_rates.png";
    //let points = data.iter().map()

    let max_x = data.nrows();
    let max_y = data.ncols();

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Network firing rates", ("sans-serif", 30).into_font())
        .margin(15)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0..max_x, 0..max_y)?;


    chart.configure_mesh().draw()?;

    //chart.draw_series(
    //points.iter().map(|p|

    Ok(())
}

pub fn plot_spikes(spikedata: Vec<Array1<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    let filename = "spikeplot.png";

    let max_x = spikedata.len() as i32;
    let max_y = spikedata[0].shape()[0] as i32;

    let points: Vec<(i32, i32)> = to_spike_points(&spikedata);

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Spike plot", ("sans-serif", 30).into_font())
        .margin(15)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0..max_x, 0..max_y)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        points.iter().map(|p| Circle::new(*p, 3, BLACK.filled())),
    ).unwrap();

    root.present()?;

    log::info!("Plot saved to {}", filename);

    Ok(())
}

/// Convert the raw record data to points that can be plotted
fn to_spike_points(spikedata: &Vec<Array1<f32>>) -> Vec<(i32, i32)> {
    let mut points: Vec<(i32, i32)> = vec![];

    for (t, s) in spikedata.iter().enumerate() {
        // Get indices i of neurons that fired at time t
        // TODO: This is duplicated in FiringState; use
        let ixs: Vec<usize> = s.iter().enumerate().filter(|(_, n)| **n != 0.0).map(|(i,_)| i).collect();

        // Map from i to (t, i)
        let mut k: Vec<(i32,i32)> = ixs.iter().map(|i| (t as i32, *i as i32)).collect();

        points.append(&mut k);
    }

    points
}
