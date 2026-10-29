#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

struct ValueInner {
    data: f64,
    grad: f64,
    op: Op,
}

#[derive(Clone)]
pub enum Op {
    Leaf,
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    Neg(Value),
    Exp(Value),
    Ln(Value),
    Tanh(Value),
    Sin(Value),
    Cos(Value),
    Powf(Value, f64),
    Sigmoid(Value),
    Relu(Value),
}

impl Value {
    pub fn new(data: f64, op: Op) -> Self {
        Self(Rc::new(RefCell::new(ValueInner { data, grad: 0.0, op })))
    }

    pub fn leaf(data: f64) -> Self {
        Self::new(data, Op::Leaf)
    }

    // --- acceso (borrows cortos: evita RefCell panics) ---
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }

    pub fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    pub fn op(&self) -> Op {
        self.0.borrow().op.clone()
    }

    // --- forward de cada op: calcula data y REGISTRA la arista ---
    pub fn add(&self, other: &Value) -> Value {
        Value::new(
            self.data() + other.data(),
            Op::Add(self.clone(), other.clone()),
        )
    }

    pub fn sub(&self, other: &Value) -> Value {
        Value::new(
            self.data() - other.data(),
            Op::Sub(self.clone(), other.clone()),
        )
    }

    pub fn mul(&self, other: &Value) -> Value {
        Value::new(
            self.data() * other.data(),
            Op::Mul(self.clone(), other.clone()),
        )
    }

    pub fn div(&self, other: &Value) -> Value {
        Value::new(
            self.data() / other.data(),
            Op::Div(self.clone(), other.clone()),
        )
    }

    pub fn neg(&self) -> Value {
        Value::new(-self.data(), Op::Neg(self.clone()))
    }

    pub fn exp(&self) -> Value {
        Value::new(self.data().exp(), Op::Exp(self.clone()))
    }

    pub fn ln(&self) -> Value {
        Value::new(self.data().ln(), Op::Ln(self.clone()))
    }

    pub fn tanh(&self) -> Value {
        Value::new(self.data().tanh(), Op::Tanh(self.clone()))
    }

    pub fn sin(&self) -> Value {
        Value::new(self.data().sin(), Op::Sin(self.clone()))
    }

    pub fn cos(&self) -> Value {
        Value::new(self.data().cos(), Op::Cos(self.clone()))
    }

    pub fn powf(&self, p: f64) -> Value {
        Value::new(self.data().powf(p), Op::Powf(self.clone(), p))
    }

    pub fn sigmoid(&self) -> Value {
        let s = 1.0 / (1.0 + (-self.data()).exp());
        Value::new(s, Op::Sigmoid(self.clone()))
    }

    pub fn relu(&self) -> Value {
        let value = self.data().max(0.0);
        Value::new(value, Op::Relu(self.clone()))
    }

    // --- reverse ---
    pub fn backward(&self) {
        let orden = topo_sort(self);
        for n in &orden {
            n.zero_grad();
        }
        self.set_grad(1.0);
        for n in orden.iter().rev() {
            n.apply_vjp();
        }
    }

    fn apply_vjp(&self) {
        match self.op() {
            Op::Leaf => {}
            Op::Add(a, b) => {
                let g = self.grad();
                a.add_grad(g);
                b.add_grad(g);
            }
            Op::Mul(a, b) => {
                let g = self.grad();
                a.add_grad(g * b.data());
                b.add_grad(g * a.data());
            }
            Op::Sub(a, b) => {
                let g = self.grad();
                a.add_grad(g);
                b.add_grad(-g);
            }
            Op::Div(a, b) => {
                let g = self.grad();
                let bd = b.data();
                a.add_grad(g / bd);
                b.add_grad(-g * a.data() / (bd * bd));
            }
            Op::Neg(a) => {
                a.add_grad(-self.grad());
            }
            Op::Exp(a) => {
                a.add_grad(self.grad() * self.data());
            }
            Op::Ln(a) => {
                a.add_grad(self.grad() / a.data());
            }
            Op::Tanh(a) => {
                let t = self.data();
                a.add_grad(self.grad() * (1.0 - t * t));
            }
            Op::Sin(a) => {
                a.add_grad(self.grad() * a.data().cos());
            }
            Op::Cos(a) => {
                a.add_grad(self.grad() * -a.data().sin());
            }
            Op::Powf(a, p) => {
                a.add_grad(self.grad() * p * a.data().powf(p - 1.0));
            }
            Op::Sigmoid(a) => {
                let s = self.data();
                a.add_grad(self.grad() * s * (1.0 - s));
            }
            Op::Relu(a) => {
                if a.data() > 0.0 {
                    a.add_grad(self.grad());
                }
            }
        }
    }

    fn operands(&self) -> Vec<Value> {
        match self.op() {
            Op::Leaf => vec![],
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => vec![a, b],
            Op::Neg(a)
            | Op::Exp(a)
            | Op::Ln(a)
            | Op::Tanh(a)
            | Op::Sin(a)
            | Op::Cos(a)
            | Op::Sigmoid(a)
            | Op::Relu(a) => vec![a],
            Op::Powf(a, _) => vec![a],
        }
    }
}

fn topo_sort(root: &Value) -> Vec<Value> {
    fn visit(
        value: &Value,
        visited: &mut HashSet<*const RefCell<ValueInner>>,
        order: &mut Vec<Value>,
    ) {
        let ptr = Rc::as_ptr(&value.0);
        if !visited.insert(ptr) {
            return;
        }
        for child in value.operands() {
            visit(&child, visited, order);
        }
        order.push(value.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    visit(root, &mut visited, &mut order);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual;

    #[test]
    fn multivariate_chain_matches_forward() {
        // z = (x*y + sin(x)).exp()  con x=0.5, y=1.5
        let x = Value::leaf(0.5);
        let y = Value::leaf(1.5);
        let z = x.mul(&y).add(&x.sin()).exp();
        z.backward();

        let dx = Dual::<2>::var(0.5, 0);
        let dy = Dual::<2>::var(1.5, 1);
        let dz = (dx * dy + dx.sin()).exp();

        assert!((x.grad() - dz.grad[0]).abs() < 1e-9);
        assert!((y.grad() - dz.grad[1]).abs() < 1e-9);
    }
}
