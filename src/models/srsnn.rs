//pub mod er_model;
//pub mod typed;
//pub mod nd_typed;

pub mod gt_model;
pub mod minimal;
pub mod test_model;
pub mod base_model;

use utils::parameters::ParameterSet;


trait SRSNN {
    type Parameters;

    fn parameters(p: ParameterSet) -> Self::Parameters;

    fn ps() -> ParameterSet;
}
