use model_runner::tensor::{ConcreteTensor, Tensor};

fn main() {
    let tensor_a = Tensor::from(ConcreteTensor::new(
        vec![1., 2., 3., 4., 5., 6.],
        vec![2, 3],
    ));
    let tensor_b = Tensor::from(ConcreteTensor::new(
        vec![7., 8., 9., 10., 11., 12.],
        vec![2, 3],
    ));

    println!("{:?}", tensor_a.clone() + tensor_b.clone());

    let (variable, _, _) = Tensor::new_variable(ConcreteTensor::constant(0.0, vec![2, 3]));

    println!("{:?}", variable.clone() + tensor_a.clone());
    println!("{:?}", variable.clone() * 0.0);
    println!("{:?}", variable.clone() / -1.0);
}
