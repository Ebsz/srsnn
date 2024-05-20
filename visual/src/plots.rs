use plotters::prelude::*;


pub fn plot_single_variable(data: Vec<f32>, description: &str, caption: &str, filename: &str, color: &RGBColor)
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

//pub fn plot_heatmap() -> Result<(), Box<dyn std::error::Error>> {
//    let filename = "heatmap.png";
//    let caption = "heatmap";
//
//    let max_x: f32 = 1000.0;
//    let min_x: f32 = 0.0;
//
//    let max_y: f32 = 200.0;
//    let min_y: f32 = 0.0;
//
//    let root = BitMapBackend::new(filename, (960, 720)).into_drawing_area();
//    root.fill(&WHITE)?;
//
//    let mut chart = ChartBuilder::on(&root)
//        .caption(caption, ("sans-serif", 50).into_font())
//        .margin(15)
//        .x_label_area_size(30)
//        .y_label_area_size(30)
//        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
//
//    chart.draw_series(
//        points.iter().map(|p| Circle::new(*p, 3, BLACK.filled())),
//    ).unwrap();
//
//    root.present()?;
//}