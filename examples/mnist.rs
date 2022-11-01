#[cfg(not(feature = "mnist"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features mnist --example mnist");
}

#[cfg(feature = "mnist")]
fn main() {
    use hog_detector::{dataset::MnistDataSet, DataSet, HogDetector, Trainable};

    let mut model = HogDetector::default();

    let mut dataset = MnistDataSet::default();
    dataset.load(false);

    model.train_class(&dataset, 1);
    assert!(model.svc.is_some());
    println!("class 1: {:?} %", model.evaluate(&dataset, 1) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);
    model.train_class(&dataset, 2);
    model.train_class(&dataset, 3);
    model.train_class(&dataset, 4);
    assert!(model.svc.is_some());
    println!("class 2: {:?} %", model.evaluate(&dataset, 2) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);
    println!("class 3: {:?} %", model.evaluate(&dataset, 3) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);
    println!("class 4: {:?} %", model.evaluate(&dataset, 4) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);
}
