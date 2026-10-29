#![allow(dead_code)]

use std::cell::RefCell;
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
    // TODO: Sin, Cos, Powf, Sigmoid, Relu, ...
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

    pub fn sub(&self, _other: &Value) -> Value {
        todo!()
    }

    pub fn mul(&self, _other: &Value) -> Value {
        todo!()
    }

    pub fn div(&self, _other: &Value) -> Value {
        todo!()
    }

    pub fn neg(&self) -> Value {
        todo!()
    }

    pub fn exp(&self) -> Value {
        todo!()
    }

    pub fn ln(&self) -> Value {
        todo!()
    }

    pub fn tanh(&self) -> Value {
        todo!()
    }

    // --- reverse ---
    pub fn backward(&self) {
        // 1. let orden = topo_sort(self);
        // 2. for n in &orden { n.zero_grad() }
        // 3. self.set_grad(1.0);
        // 4. for n in orden.iter().rev() { n.apply_vjp() }
        todo!()
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
            Op::Sub(_, _)
            | Op::Div(_, _)
            | Op::Neg(_)
            | Op::Exp(_)
            | Op::Ln(_)
            | Op::Tanh(_) => todo!(),
        }
    }

    fn operands(&self) -> Vec<Value> {
        match self.op() {
            Op::Leaf => vec![],
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => vec![a, b],
            Op::Neg(a) | Op::Exp(a) | Op::Ln(a) | Op::Tanh(a) => vec![a],
        }
    }
}

fn topo_sort(_root: &Value) -> Vec<Value> {
    // DFS postorden con un HashSet<*const ValueInner> para no repetir nodos.
    // Postorden => hijos antes que padres; la raíz queda al final.
    // Pista: Rc::as_ptr(&value.0) te da un puntero estable para la identidad.
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual;

    #[test]
    #[ignore = "TODO: implementar el modo reverse antes de activar este test"]
    fn multivariate_chain_matches_forward() {
        // z = (x*y + sin(x)).exp()  con x=0.5, y=1.5
        // Comparar x.grad() / y.grad() con Dual::<2> (aprox. 8.128 y 1.709).
        let _ = Dual::<2>::var(0.5, 0);
        todo!()
    }
}
