//! Spiking network analysis.

use srsnn::analysis::graph::{Graph, GraphAnalysis};
use srsnn::analysis;
use model::network::representation::DefaultRepresentation;
use utils::data;

use std::env;


fn load_network() -> DefaultRepresentation {
    let path = match parse_arg()  {
        Some(s) => { s },
        None => { println!("usage: analysis [path]"); std::process::exit(1); }
    };

    match data::load::<DefaultRepresentation>(path.as_str()) {
        Ok(r) => { r },
        Err(e) =>     { println!("error: {e}"); std::process::exit(1); }
    }
}

fn parse_arg() -> Option<String> {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        return Some(args[1].clone());
    }

    None
}


fn main() {
    let r = load_network();

    analysis::analyze_network(&r);


    //let rg = Graph::reduce(&graph);
    //let rga = GraphAnalysis::analyze(&rg);

    //println!("{rg}");
    //println!("{rga}");
    //println!("{:?}", rg.edges());
}
