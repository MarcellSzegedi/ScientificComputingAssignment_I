use _core::cylinder::{Cylinder, CylinderZOrder};

fn main() {
    let mut cylinder = Cylinder::new(1024, vec![]);
    for _ in 1..=1000 {
        cylinder.update(1.0 / 128.0, 0.001, 1.0);
    }
}
