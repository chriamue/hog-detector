// source: https://github.com/12101111/yolo-rs/blob/master/src/yolo.rs
// x, y is the upper left corner
/// A bounding box struct with mergre functions
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// left side
    pub x: f32,
    /// upper side
    pub y: f32,
    /// width
    pub w: f32,
    /// height
    pub h: f32,
}

impl BBox {
    /// left side of bounding box
    pub fn left(&self) -> f32 {
        self.x
    }
    /// right side of bounding box
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    /// top of bounding box
    pub fn top(&self) -> f32 {
        self.y
    }
    /// bottom of bounding box
    pub fn bot(&self) -> f32 {
        self.y + self.h
    }

    /// calculate the amount of overlap with another bounding box
    pub fn overlay(&self, rhs: &BBox) -> f32 {
        let left = self.left().max(rhs.left());
        let right = self.right().min(rhs.right());
        let w = (right - left).max(0.0);
        let top = self.top().max(rhs.top());
        let bot = self.bot().min(rhs.bot());
        let h = (bot - top).max(0.0);
        w * h
    }

    /// calculate the union with another bounding box
    pub fn union(&self, rhs: &BBox) -> f32 {
        self.w * self.h + rhs.w * rhs.h - self.overlay(rhs)
    }

    /// calculate the intersection over union with another bounding box
    pub fn iou(&self, rhs: &BBox) -> f32 {
        self.overlay(rhs) / self.union(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_iou() {
        let b1 = BBox {
            x: 0.5,
            y: 0.5,
            w: 1.0,
            h: 1.0,
        };
        assert_eq!(b1.left(), 0.5);
        assert_eq!(b1.right(), 1.5);
        assert_eq!(b1.top(), 0.5);
        assert_eq!(b1.bot(), 1.5);
        assert_eq!(b1.overlay(&b1), 1.0);
        assert_eq!(b1.union(&b1), 1.0);
        assert_eq!(b1.iou(&b1), 1.0);
    }

    #[test]
    pub fn test_overlay() {
        let b1 = BBox {
            x: 4.0,
            y: 4.0,
            w: 4.4,
            h: 4.4,
        };
        let b2 = BBox {
            x: 8.0,
            y: 4.0,
            w: 4.4,
            h: 4.4,
        };
        assert!(b1.overlay(&b2) > 1.0);
    }
}
