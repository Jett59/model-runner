use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{ConcreteTensor, Tensor};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Incompatible shapes {:?} and {:?}",
            self.shape(),
            rhs.shape(),
        );
        if matches!(self, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            rhs
        } else if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            self
        } else {
            match (self, rhs) {
                (Tensor::Concrete(left), Tensor::Concrete(right)) => {
                    let shape = left.shape().to_vec();
                    let data = left
                        .data()
                        .into_iter()
                        .zip(right.data())
                        .map(|(a, b)| a + b)
                        .collect();
                    Tensor::Concrete(ConcreteTensor::new(data, shape))
                }
                (left, right) => Tensor::AddTensor(Box::new(left), Box::new(right)),
            }
        }
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            self
        } else {
            match self {
                Tensor::Concrete(tensor) => {
                    let data = tensor.data().into_iter().map(|&x| x + rhs).collect();
                    Tensor::Concrete(ConcreteTensor::new(data, tensor.shape().to_vec()))
                }
                left => Tensor::AddScalar(Box::new(left), rhs),
            }
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Incompatible shapes {:?} and {:?}",
            self.shape(),
            rhs.shape(),
        );
        if matches!(self, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            -rhs
        } else if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            self
        } else {
            match (self, rhs) {
                (Tensor::Concrete(left), Tensor::Concrete(right)) => {
                    let shape = left.shape().to_vec();
                    let data = left
                        .data()
                        .into_iter()
                        .zip(right.data())
                        .map(|(a, b)| a - b)
                        .collect();
                    Tensor::Concrete(ConcreteTensor::new(data, shape))
                }
                (left, right) => Tensor::SubtractTensor(Box::new(left), Box::new(right)),
            }
        }
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            self
        } else {
            match self {
                Tensor::Concrete(tensor) => {
                    let data = tensor.data().into_iter().map(|&x| x - rhs).collect();
                    Tensor::Concrete(ConcreteTensor::new(data, tensor.shape().to_vec()))
                }
                left => Tensor::SubtractScalar(Box::new(left), rhs),
            }
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Incompatible shapes {:?} and {:?}",
            self.shape(),
            rhs.shape(),
        );
        if matches!(self, Tensor::Concrete(ref tensor) if tensor.all_equal_to(1.0)) {
            rhs
        } else if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(1.0)) {
            self
        } else if matches!(self, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0))
            || matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0))
        {
            Tensor::Concrete(ConcreteTensor::constant(0.0, self.shape().to_vec()))
        } else {
            match (self, rhs) {
                (Tensor::Concrete(left), Tensor::Concrete(right)) => {
                    let shape = left.shape().to_vec();
                    let data = left
                        .data()
                        .into_iter()
                        .zip(right.data())
                        .map(|(a, b)| a * b)
                        .collect();
                    Tensor::Concrete(ConcreteTensor::new(data, shape))
                }
                (left, right) => Tensor::MultiplyTensor(Box::new(left), Box::new(right)),
            }
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        if rhs == 1.0 {
            self
        } else if rhs == 0.0 {
            Tensor::Concrete(ConcreteTensor::constant(0.0, self.shape().to_vec()))
        } else {
            match self {
                Tensor::Concrete(tensor) => {
                    let data = tensor.data().into_iter().map(|&x| x * rhs).collect();
                    Tensor::Concrete(ConcreteTensor::new(data, tensor.shape().to_vec()))
                }
                left => Tensor::MultiplyScalar(Box::new(left), rhs),
            }
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Incompatible shapes {:?} and {:?}",
            self.shape(),
            rhs.shape(),
        );
        if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            panic!("Division by zero");
        } else if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(1.0)) {
            self
        } else if matches!(rhs, Tensor::Concrete(ref tensor) if tensor.all_equal_to(-1.0)) {
            -self
        } else if matches!(self, Tensor::Concrete(ref tensor) if tensor.all_equal_to(0.0)) {
            self
        } else {
            match (self, rhs) {
                (Tensor::Concrete(left), Tensor::Concrete(right)) => {
                    let shape = left.shape().to_vec();
                    let data = left
                        .data()
                        .into_iter()
                        .zip(right.data())
                        .map(|(a, b)| a / b)
                        .collect();
                    Tensor::Concrete(ConcreteTensor::new(data, shape))
                }
                (left, right) => Tensor::DivideTensor(Box::new(left), Box::new(right)),
            }
        }
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        if rhs == 1.0 {
            self
        } else if rhs == -1.0 {
            -self
        } else if rhs == 0.0 {
            panic!("Division by zero");
        } else {
            match self {
                Tensor::Concrete(tensor) => {
                    let data = tensor.data().into_iter().map(|&x| x / rhs).collect();
                    Tensor::Concrete(ConcreteTensor::new(data, tensor.shape().to_vec()))
                }
                left => Tensor::DivideScalar(Box::new(left), rhs),
            }
        }
    }
}
