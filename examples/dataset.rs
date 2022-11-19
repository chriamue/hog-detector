#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("dataset example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::{
        classifier::SVMClassifier, data_augmentation::DataAugmentation, dataset::FolderDataSet,
        DataSet, HogDetector, Trainable,
    };
    let mut model: HogDetector<SVMClassifier> = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load();
    dataset.augment();

    model.train_class(&dataset, 5);
    println!("evaluated: {:?} %", model.evaluate(&dataset, 5) * 100.0);

    dataset.generate_hard_negative_samples(&model, 5, None);
    model.train_class(&dataset, 5);
    println!(
        "evaluated after training with hard negative samples: {:?} %",
        model.evaluate(&dataset, 5) * 100.0
    );
}
