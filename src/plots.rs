use plotters::prelude::*;

use ndarray::{s, Array, Array1, Array2};
use model::record::{Record, RecordType};

use evolution::stats::OptimizationStatistics;


pub fn generate_plots(record: &Record) {
    let single_pot: Vec<f32> = record.get(RecordType::Potentials).iter().map(|x| x[0]).collect();

    let plot_ok = plot_single_neuron_potential(&single_pot);
    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }

    let mut spikedata: Vec<Array1<f32>> = record.get(RecordType::Spikes);

    let plot_ok = plot_spikes(spikedata, "spikeplot.png");

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}

pub fn plot_all_potentials(record: &Record) {
    let mut pots: Vec<Vec<f32>> = Vec::new();
    let potentials = record.get(RecordType::Potentials);

    for i in 0..potentials[0].shape()[0] {
        pots.push(potentials.iter().map(|x| x[i]).collect());
    }

    let plot_ok = plt::plot_multiple_series(pots, "Potential", "allpots.png");

    match plot_ok {
        Ok(_) => (),
        Err(e) => println!("Error creating plot: {:?}", e),
    }
}

pub fn plot_stats(stats: &OptimizationStatistics, name: &str) {
    let best: Vec<Vec<f32>> = stats.runs.iter().map(|x| x.best_series().clone()).collect();
    let mean: Vec<Vec<f32>> = stats.runs.iter().map(|x| x.mean_series().clone()).collect();
    let std: Vec<Vec<f32>> = stats.runs.iter().map(|x| x.stddev_series().clone()).collect();

    let _ = plt::plot_multiple_series(best,
        "Generation best", format!("{}_best.png", name).as_str());
    let _ = plt::plot_multiple_series(mean,
        "Generation mean", format!("{}_mean.png", name).as_str());
    let _ = plt::plot_multiple_series(std,
        "Generation std", format!("{}_std.png", name).as_str());
}

pub fn plot_single_neuron_potential(potentials: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    plt::plot_single_variable(&potentials, "Potential", "Potentials", "pots.png", &BLACK)
}

pub fn plot_run_spikes(r: &Record, filename: Option<&str>) {
    let mut spk_rec = r.get(RecordType::Spikes);

    let mut spk_in = r.get(RecordType::InputSpikes);
    let mut spk_out = r.get(RecordType::OutputSpikes);

    let file = match filename {
        Some(f) => {f},
        None => "dual_spikeplot.png"
    };

    let _ = plot_spikes_with_io(spk_rec, spk_in, spk_out, file);
}

/// Plot a split spike chart with recurrent spikes on top, input spikes below
fn plot_spikes_with_io(
    spk_rec: Vec<Array1<f32>>,
    spk_in: Vec<Array1<f32>>,
    spk_out: Vec<Array1<f32>>,
    filename: &str)
    -> Result<(), Box<dyn std::error::Error>>
{
    let max_x = spk_rec.len() as i32;

    let rec_points: Vec<(i32, i32)> = to_spike_points(&spk_rec);
    let in_points: Vec<(i32, i32)> = to_spike_points(&spk_in);
    let out_points: Vec<(i32, i32)> = to_spike_points(&spk_out);

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, btm) = root.split_vertically((60).percent());

    let (btm1, btm2) = btm.split_vertically((50).percent());

    let top_max_y = spk_rec[0].shape()[0] as i32;
    let in_max_y = spk_in[0].shape()[0] as i32;
    let out_max_y = spk_out[0].shape()[0] as i32;

    let mut top_chart = ChartBuilder::on(&top)
        //.caption("Recurrent", ("sans-serif", 30).into_font())
        .set_label_area_size(LabelAreaPosition::Left, 30)
        .set_label_area_size(LabelAreaPosition::Top, 30)
        .margin(5)
        //.x_label_area_size(20)
        //.y_label_area_size(20)
        .build_cartesian_2d(0..max_x, 0..top_max_y)?;

    top_chart.configure_mesh().draw()?;

    top_chart.draw_series(
        rec_points.iter().map(|p| Circle::new(*p, 2, BLACK.filled())),
    ).unwrap();

    let mut in_chart = ChartBuilder::on(&btm1)
        .margin(5)
        //.caption("input", ("sans-serif", 30).into_font())
        //.x_label_area_size(20)
        //.y_label_area_size(20)
        .set_label_area_size(LabelAreaPosition::Left, 30)
        .build_cartesian_2d(0..max_x, 0..in_max_y)?;

    in_chart.configure_mesh().draw()?;

    in_chart.draw_series(
        in_points.iter().map(|p| Circle::new(*p, 2, BLACK.filled())),
    ).unwrap();

    let mut out_chart = ChartBuilder::on(&btm2)
        .margin(5)
        //.x_label_area_size(20)
        //.y_label_area_size(20)
        .set_label_area_size(LabelAreaPosition::Left, 30)
        .build_cartesian_2d(0..max_x, 0..out_max_y)?;

    out_chart.configure_mesh().draw()?;
    out_chart.draw_series(
        out_points.iter().map(|p| Circle::new(*p, 2, BLACK.filled())),
    ).unwrap();

    log::info!("Plot saved to {}", filename);

    Ok(())
}

pub fn plot_spikes(spikedata: Vec<Array1<f32>>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
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
fn to_spike_points(spikedata: &[Array1<f32>]) -> Vec<(i32, i32)> {
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

pub fn plot_degree_distribution(data: &Array1<u32>) {
    let _ = plt::histogram(data, "dgdist.png", "Degree distribution");
}

pub fn single_neuron_dynamics(record: &Record)
    -> Result<(), Box<dyn std::error::Error>> {
    let filename = "dynamics.png";

    let pots: Vec<f32> = record.get(RecordType::Potentials).iter().map(|x| x[0]).collect();
    let spikes: Vec<Array1<f32>> = record.get(RecordType::Spikes);
    let synaptic_current: Vec<f32> = record.get(RecordType::SynapticCurrent).iter().map(|x| x[0]).collect();

    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    //let areas = root.split_evenly((3,1));
    let (top, btm) = root.split_vertically((85).percent());

    let (t1, t2) = top.split_vertically((50).percent());

    plt::subplot::series(&synaptic_current, &t1);
    plt::subplot::series(&pots, &t2);
    plt::subplot::spikes(spikes, &btm);

    root.present()?;

    log::info!("Plot saved to {}", filename);

    Ok(())
}

pub mod plt {
    use super::*;

    use utils::math;

    pub fn plot_single_variable(
        data: &[f32],
        description: &str,
        caption: &str,
        filename: &str,
        color: &RGBColor)
        -> Result<(), Box<dyn std::error::Error>> {

        let max_x: f32 = data.len() as f32;
        let min_x: f32 = 0.0;

        let max_y: f32 = data.iter().fold(0.0f32, |acc, &x| if x > acc {x} else {acc});
        let min_y: f32 = data.iter().fold(0.0f32, |acc, &x| if x < acc {x} else {acc});

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        let series = LineSeries::new(
            data.iter().enumerate().map(|(i, x)| (i as f32, *x)),
            color,
        );

        chart
            .draw_series(series)?
            .label(description)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    /// Plot multiple series
    pub fn plot_multiple_series(
        data: Vec<Vec<f32>>,
        caption: &str,
        filename: &str)
        -> Result<(), Box<dyn std::error::Error>> {
        let max_x: f32 = data[0].len() as f32;
        let min_x: f32 = 0.0;

        let max_y: f32 = data[0].iter().fold(0.0f32, |acc, &x| if x > acc {x} else {acc});
        let min_y: f32 = data[0].iter().fold(0.0f32, |acc, &x| if x < acc {x} else {acc});

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        //let colors: Vec<RGBColor> = (0..data.len()).map(|i| RGBColor((0 + i * 20) as u8, 0 , (255 - i * 20) as u8)).collect();

        for d in data.iter() {
            let series = LineSeries::new(
                d.iter().enumerate().map(|(i, x)| (i as f32, *x)),
                &RED,
            );
            chart
                .draw_series(series)?;
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    /// Plot two variables against each other
    pub fn xy_plot(
        points: Vec<(f32, f32)>,
        filename: &str,
        caption: &str)
        -> Result<(), Box<dyn std::error::Error>> {

        let xs: Vec<f32> = points.iter().map(|(x,_)| *x).collect();
        let ys: Vec<f32> = points.iter().map(|(_,y)| *y).collect();

        let max_x: f32 = math::maxf(&xs);
        let max_y: f32 = math::maxf(&ys);
        let min_x: f32 = math::minf(&xs);
        let min_y: f32 = math::minf(&ys);

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 30).into_font())
            .margin(15)
            .x_label_area_size(20)
            .y_label_area_size(20)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart.configure_mesh().draw()?;

        let series = LineSeries::new(points, &BLACK);
        chart.draw_series(series).unwrap();

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    pub fn histogram(
        data: &Array1<u32>,
        filename: &str,
        caption: &str)
        -> Result<(), Box<dyn std::error::Error>> {

        let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_x: u32 = data.shape()[0] as u32 / 2;

        let mut chart = ChartBuilder::on(&root)
            .caption(caption, ("sans-serif", 30).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(35)
            .build_cartesian_2d((0u32..max_x).into_segmented(), 0u32..10u32)?;

        chart.configure_mesh()
            .disable_x_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .y_desc("Count")
            .x_desc("Bucket")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(RED.mix(0.5).filled())
                .data(data.iter().map(|x: &u32| (*x, 1)))
        )?;

        root.present()?;

        log::info!("Plot saved to {}", filename);

        Ok(())
    }

    //use ndarray::Array3;
    //pub fn plot_3d(
    //    data: &Array3<u32>,
    //    filename: &str,
    //    caption: &str)
    //    -> Result<(), Box<dyn std::error::Error>> {

    //    let min_x = 0.0;
    //    let min_y = 0.0;
    //    let max_z = 0.0;
    //    let max_x = 1.0;
    //    let max_y = 1.0;
    //    let max_z = 1.0;

    //    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
    //    root.fill(&WHITE)?;

    //    let mut chart = ChartBuilder::on(&root)
    //        .caption(caption, ("sans-serif", 30).into_font())
    //        .margin(5)
    //        .x_label_area_size(30)
    //        .y_label_area_size(35)
    //        .build_cartesian_3d(min_x..max_x, min_y..max_y, min_x..max_x)?;

    //    chart.configure_axes().draw()?;

    //    Ok(())
    //}

    pub fn plot_matrix(data: &Array2<f32>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (dw, dh) = (data.shape()[0], data.shape()[1]);

        let max_x = dw as u32 * 4;
        let max_y = dh as u32 * 4;

        let root = BitMapBackend::new(filename, (max_x, max_y)).into_drawing_area();
        root.fill(&WHITE)?;
        let areas = root.split_evenly((dw, dh));

        let c_val = |x: f32| (255.0 * x) as u8; // higher val is darker
        //let c_val = |x: f32| (255.0 * x) as u8; // lower val is darker

        let color = |x: f32| RGBColor(c_val(x), 128 - c_val(x) * 2, 128 - c_val(x) * 2);

        for (v, a) in data.iter().zip(areas.iter()) {
            let c = color(*v);
            a.fill(&c);
        }

        root.present()?;

        log::info!("Plot saved to {}", filename);
        Ok(())
    }

    pub mod subplot {
        use super::*;

        use plotters::coord::Shift;


        pub fn series<DB: DrawingBackend + 'static> (
            data: &[f32],
            d: &DrawingArea<DB, Shift>)
        -> Result<(), Box<dyn std::error::Error>> {
            let max_x: f32 = data.len() as f32;

            let mut min_y: f32 = math::minf(data);
            let mut max_y: f32 = math::maxf(data);

            if min_y == max_y {
                min_y = min_y * 0.9;
                max_y = max_y * 1.1;
            }

            let mut chart = ChartBuilder::on(&d)
                .margin(30)
                .set_label_area_size(LabelAreaPosition::Left, 20)
                .set_label_area_size(LabelAreaPosition::Bottom, 20)
                .build_cartesian_2d(0.0..max_x, min_y..max_y)?;

            chart
                .configure_mesh()
                .disable_mesh()
                .draw()?;

            let series: LineSeries<DB, (f32, f32)> = LineSeries::new(
                data.iter().enumerate().map(|(i, x)| (i as f32, *x)),
                &BLACK,
            );

            chart.draw_series(series)?;

            Ok(())
        }

        pub fn spikes<DB: DrawingBackend + 'static> (
            spikedata: Vec<Array1<f32>>,
            d: &DrawingArea<DB, Shift>)
        -> Result<(), Box<dyn std::error::Error>> {
            let max_x = spikedata.len() as i32;

            let min_y = -1;
            let max_y = spikedata[0].shape()[0] as i32;

            let points = to_spike_points(&spikedata);

            let mut chart = ChartBuilder::on(&d)
                .margin(15)
                //.set_label_area_size(LabelAreaPosition::Left, 20)
                //.set_label_area_size(LabelAreaPosition::Bottom, 20)
                .x_label_area_size(20)
                .y_label_area_size(20)
                .build_cartesian_2d(0..max_x, min_y..max_y)?;

            chart.configure_mesh().draw()?;

            chart.draw_series(
                points.iter().map(|p| Circle::new(*p, 3, BLACK.filled())),
            ).unwrap();

            Ok(())
        }
    }
}
