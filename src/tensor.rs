use std::{
    cell::RefCell,
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
};

mod operators;

#[derive(Clone, Debug)]
pub struct ConcreteTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl ConcreteTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "Data size {} does not match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape }
    }

    pub fn constant(value: f32, shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![value; size];
        Self { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn all_equal_to(&self, value: f32) -> bool {
        self.data.iter().all(|&x| x == value)
    }
}

#[derive(Clone, Debug)]
pub enum Tensor {
    Concrete(ConcreteTensor),
    Variable(u32, Rc<RefCell<ConcreteTensor>>, Vec<usize>),

    Reshape(Box<Tensor>, Vec<usize>),

    AddTensor(Box<Tensor>, Box<Tensor>),
    SubtractTensor(Box<Tensor>, Box<Tensor>),
    MultiplyTensor(Box<Tensor>, Box<Tensor>),
    DivideTensor(Box<Tensor>, Box<Tensor>),

    AddScalar(Box<Tensor>, f32),
    SubtractScalar(Box<Tensor>, f32),
    MultiplyScalar(Box<Tensor>, f32),
    DivideScalar(Box<Tensor>, f32),
}

impl From<ConcreteTensor> for Tensor {
    fn from(tensor: ConcreteTensor) -> Self {
        Tensor::Concrete(tensor)
    }
}

impl Tensor {
    pub fn new_variable(initial_value: ConcreteTensor) -> (Self, u32, Rc<RefCell<ConcreteTensor>>) {
        static NEXT_ID: AtomicU32 = AtomicU32::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let shape = initial_value.shape().to_vec();
        let tensor_handle1 = Rc::new(RefCell::new(initial_value));
        let tensor_handle2 = tensor_handle1.clone();
        (
            Tensor::Variable(id, tensor_handle1, shape),
            id,
            tensor_handle2,
        )
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Concrete(tensor) => tensor.shape(),
            Tensor::Variable(_, _, shape) => shape,
            Tensor::Reshape(_, shape) => shape,
            Tensor::AddTensor(left, _) => left.shape(),
            Tensor::SubtractTensor(left, _) => left.shape(),
            Tensor::MultiplyTensor(left, _) => left.shape(),
            Tensor::DivideTensor(left, _) => left.shape(),
            Tensor::AddScalar(left, _) => left.shape(),
            Tensor::SubtractScalar(left, _) => left.shape(),
            Tensor::MultiplyScalar(left, _) => left.shape(),
            Tensor::DivideScalar(left, _) => left.shape(),
        }
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        let new_size: usize = shape.iter().product();
        let old_size = self.shape().iter().product();
        assert_eq!(
            new_size,
            old_size,
            "Incompatible shapes {:?} and {:?}",
            self.shape(),
            shape
        );
        match self {
            Tensor::Concrete(tensor) => Tensor::Concrete(ConcreteTensor::new(tensor.data, shape)),
            tensor => Tensor::Reshape(Box::new(tensor), shape),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tensor_arithmetic() {
        let tensor_a = Tensor::from(ConcreteTensor::constant(2.0, vec![2, 2]));
        let tensor_b = Tensor::from(ConcreteTensor::constant(3.0, vec![2, 2]));

        let result = tensor_a.clone() + tensor_b.clone();
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(5.0)));

        let result = tensor_a.clone() + 3.0;
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(5.0)));

        let result = tensor_a.clone() - tensor_b.clone();
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(-1.0)));

        let result = tensor_a.clone() - 3.0;
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(-1.0)));

        let result = tensor_a.clone() * tensor_b.clone();
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(6.0)));

        let result = tensor_a.clone() * 3.0;
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(6.0)));

        let result = tensor_a.clone() / tensor_b.clone();
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(2.0/3.0)));

        let result = tensor_a.clone() / 3.0;
        assert_eq!(result.shape(), &[2, 2]);
        assert!(matches!(result, Tensor::Concrete(tensor) if tensor.all_equal_to(2.0/3.0)));
    }
}
