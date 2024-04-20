use ndarray::Array1;
use std::collections::HashMap;


#[derive(Debug, Hash, Eq, PartialEq)]
pub enum RecordType {
    Spikes,
    Potentials
}

#[derive(Debug)]
pub enum RecordDataType {
    Spikes(Array1<f32>),
    Potentials(Array1<f32>)
}

pub struct Record {
    records: HashMap<RecordType, Vec<RecordDataType>>
}

pub trait Recording {
    type Data;

    fn get(&self) -> Self::Data;
}


impl Record {
    pub fn new() -> Record {
        let mut records = HashMap::new();

        // TODO: use strum to iterate over RecordTypes instead
        records.insert(RecordType::Spikes, Vec::new());
        records.insert(RecordType::Potentials, Vec::new());

        Record {
            records
        }
    }

    pub fn log(&mut self, record_type: RecordType, data: RecordDataType) {
        if let Some(d) = self.records.get_mut(&record_type) {
            d.push(data);
        }
    }

    pub fn get(&self, record_type: RecordType) -> &Vec<RecordDataType>{
        if let Some(d) = self.records.get(&record_type) {
            d
        } else {
            panic!("Could not get record of type {:?}", record_type);
        }
    }

    //pub fn get_value<T: Recording>(&self, record_type: RecordType) -> T {
    //    self.records.get(&record_type).clone()
    //}

    pub fn get_potentials(&self) -> Vec<Array1<f32>>{
        // TODO: won't want to implement this for each RecordType, so
        // this unpacking should probably be performed where it's used
        // ...macros?
        let rec = self.get(RecordType::Potentials);

        let mut potentials = vec![];
        for a in rec {
            if let RecordDataType::Potentials(p) = a {
                potentials.push(p.clone())
            } else {
                panic!("Could not get record of potentials")
            }
        }

        potentials
    }
}

//impl SpikeRecord {
//    pub fn to_firing_rate(&self) -> {
//        const TIME_WINDOW: u32 = 10;
//    }
//}
