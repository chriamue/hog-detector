#[cfg(not(feature = "mnist"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features mnist --example mnist");
}

#[cfg(feature = "mnist")]
fn main() {
    use hog_detector::classifier::SVMClassifier;
    use hog_detector::{dataset::MnistDataSet, DataSet, HogDetector, Trainable};

    let mut model = HogDetector::<SVMClassifier>::default();

    let mut dataset = MnistDataSet::default();
    dataset.load();

    model.train_class(&dataset, 1);
    assert!(model.classifier.is_some());
    println!("class 1: {:?} %", model.evaluate(&dataset, 1) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);

    model.train_class(&dataset, 2);
    println!("class 2: {:?} %", model.evaluate(&dataset, 2) * 100.0);
    assert!(model.evaluate(&dataset, 2) > 0.0);

    model.train_class(&dataset, 3);
    println!("class 3: {:?} %", model.evaluate(&dataset, 3) * 100.0);
    assert!(model.evaluate(&dataset, 3) > 0.0);
    model.train_class(&dataset, 4);
    println!("class 4: {:?} %", model.evaluate(&dataset, 4) * 100.0);
    assert!(model.evaluate(&dataset, 4) > 0.0);
    model.train_class(&dataset, 5);
    println!("class 5: {:?} %", model.evaluate(&dataset, 5) * 100.0);
    model.train_class(&dataset, 6);
    println!("class 6: {:?} %", model.evaluate(&dataset, 6) * 100.0);
    model.train_class(&dataset, 7);
    println!("class 7: {:?} %", model.evaluate(&dataset, 7) * 100.0);
    model.train_class(&dataset, 8);
    println!("class 8: {:?} %", model.evaluate(&dataset, 8) * 100.0);
    model.train_class(&dataset, 9);

    println!("class 9: {:?} %", model.evaluate(&dataset, 9) * 100.0);
}
