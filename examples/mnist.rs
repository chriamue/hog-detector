#[cfg(not(feature = "mnist"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features mnist --example mnist");
}

#[cfg(feature = "mnist")]
fn main() {
    use hog_detector::dataset::MnistDataSet;
    use hog_detector::hogdetector::HogDetectorTrait;
    use hog_detector::DataSet;
    use hog_detector::HogDetector;
    use object_detector_rust::prelude::BayesClassifier;

    let mut model: HogDetector<f32, usize, BayesClassifier<_, _>, _> = HogDetector::default();

    let mut dataset = MnistDataSet::default();
    dataset.load().unwrap();

    let (x, y) = dataset.get_data();
    let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
    model.fit_class(&x, &y, 1).unwrap();

    assert!(model.classifier.is_some());
    println!("class 1: {:?} %", model.evaluate(&dataset, 1) * 100.0);
    assert!(model.evaluate(&dataset, 1) > 0.0);

    model.fit_class(&x, &y, 2).unwrap();
    println!("class 2: {:?} %", model.evaluate(&dataset, 2) * 100.0);
    assert!(model.evaluate(&dataset, 2) > 0.0);

    model.fit_class(&x, &y, 3).unwrap();
    println!("class 3: {:?} %", model.evaluate(&dataset, 3) * 100.0);
    assert!(model.evaluate(&dataset, 3) > 0.0);
    model.fit_class(&x, &y, 4).unwrap();
    println!("class 4: {:?} %", model.evaluate(&dataset, 4) * 100.0);
    assert!(model.evaluate(&dataset, 4) > 0.0);
    model.fit_class(&x, &y, 5).unwrap();
    println!("class 5: {:?} %", model.evaluate(&dataset, 5) * 100.0);
    model.fit_class(&x, &y, 6).unwrap();
    println!("class 6: {:?} %", model.evaluate(&dataset, 6) * 100.0);
    model.fit_class(&x, &y, 7).unwrap();
    println!("class 7: {:?} %", model.evaluate(&dataset, 7) * 100.0);
    model.fit_class(&x, &y, 8).unwrap();
    println!("class 8: {:?} %", model.evaluate(&dataset, 8) * 100.0);
    model.fit_class(&x, &y, 9).unwrap();

    println!("class 9: {:?} %", model.evaluate(&dataset, 9) * 100.0);
}
