/// data augmentation trait
pub trait DataAugmentation {
    /// augment dataset images
    fn augment(&mut self);
}
