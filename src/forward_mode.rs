use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
}

impl<const N: usize> Dual<N> {
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; N],
        }
    }

    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Self { value, grad }
    }

    fn unary(self, value: f64, derivative: f64) -> Self {
        Self {
            value,
            grad: self.grad.map(|g| g * derivative),
        }
    }

    pub fn sin(self) -> Self {
        self.unary(self.value.sin(), self.value.cos())
    }

    pub fn cos(self) -> Self {
        self.unary(self.value.cos(), -self.value.sin())
    }

    pub fn tan(self) -> Self {
        self.unary(self.value.tan(), 1.0 / self.value.cos().powi(2))
    }

    pub fn exp(self) -> Self {
        let e = self.value.exp();
        self.unary(e, e)
    }

    pub fn ln(self) -> Self {
        self.unary(self.value.ln(), 1.0 / self.value)
    }

    pub fn log(self, base: f64) -> Self {
        self.unary(self.value.log(base), 1.0 / (self.value * base.ln()))
    }

    pub fn log2(self) -> Self {
        self.unary(self.value.log2(), 1.0 / (self.value * std::f64::consts::LN_2))
    }

    pub fn log10(self) -> Self {
        self.unary(
            self.value.log10(),
            1.0 / (self.value * std::f64::consts::LN_10),
        )
    }

    pub fn sqrt(self) -> Self {
        let r = self.value.sqrt();
        self.unary(r, 0.5 / r)
    }

    pub fn recip(self) -> Self {
        self.unary(self.value.recip(), -1.0 / (self.value * self.value))
    }

    pub fn powf(self, p: f64) -> Self {
        self.unary(self.value.powf(p), p * self.value.powf(p - 1.0))
    }

    pub fn powi(self, n: i32) -> Self {
        self.unary(self.value.powi(n), (n as f64) * self.value.powi(n - 1))
    }

    pub fn tanh(self) -> Self {
        let t = self.value.tanh();
        self.unary(t, 1.0 - t * t)
    }

    pub fn sigmoid(self) -> Self {
        let s = 1.0 / (1.0 + (-self.value).exp());
        self.unary(s, s * (1.0 - s))
    }

    pub fn logistic(self) -> Self {
        self.sigmoid()
    }

    pub fn relu(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Self {
                value: self.value.max(0.0),
                grad: [0.0; N],
            }
        }
    }

    pub fn abs(self) -> Self {
        let derivative = if self.value >= 0.0 { 1.0 } else { -1.0 };
        self.unary(self.value.abs(), derivative)
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Dual<N>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] + rhs.grad[i]),
        }
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Dual<N>;

    fn add(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Add<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn add(self, rhs: Dual<N>) -> Self::Output {
        Dual {
            value: self + rhs.value,
            grad: rhs.grad,
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Dual<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] - rhs.grad[i]),
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Dual<N>;

    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Sub<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn sub(self, rhs: Dual<N>) -> Self::Output {
        Dual {
            value: self - rhs.value,
            grad: rhs.grad.map(|g| -g),
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Dual<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            grad: std::array::from_fn(|i| self.grad[i] * rhs.value + self.value * rhs.grad[i]),
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Dual<N>;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value * rhs,
            grad: self.grad.map(|g| g * rhs),
        }
    }
}

impl<const N: usize> Mul<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn mul(self, rhs: Dual<N>) -> Self::Output {
        Dual {
            value: self * rhs.value,
            grad: rhs.grad.map(|g| g * self),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Dual<N>;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value,
            grad: std::array::from_fn(|i| {
                (self.grad[i] * rhs.value - self.value * rhs.grad[i]) / (rhs.value * rhs.value)
            }),
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Dual<N>;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value / rhs,
            grad: self.grad.map(|g| g / rhs),
        }
    }
}

impl<const N: usize> Div<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn div(self, rhs: Dual<N>) -> Self::Output {
        Dual {
            value: self / rhs.value,
            grad: rhs.grad.map(|g| -self * g / (rhs.value * rhs.value)),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Dual<N>;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            grad: self.grad.map(|g| -g),
        }
    }
}

impl<const N: usize> AddAssign for Dual<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign for Dual<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> MulAssign for Dual<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> DivAssign for Dual<N> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    #[test]
    fn arithmetic() {
        let x = Dual::<1>::var(3.0, 0);
        let y = Dual::<1>::var(2.0, 0);

        assert_close((x + y).value, 5.0);
        assert_close((x - y).value, 1.0);
        assert_close((x * y).value, 6.0);
        assert_close((x / y).value, 1.5);
        assert_close((x + y).grad[0], 2.0);
        assert_close((x - y).grad[0], 0.0);
        assert_close((x * y).grad[0], 5.0);
        assert_close((x / y).grad[0], -0.25);
    }

    #[test]
    fn scalar_ops() {
        let x = Dual::<1>::var(3.0, 0);
        assert_close((x + 1.0).value, 4.0);
        assert_close((1.0 + x).value, 4.0);
        assert_close((2.0 * x).grad[0], 2.0);
        assert_close((6.0 / x).grad[0], -6.0 / 9.0);
    }

    #[test]
    fn elementary_functions() {
        let x = 0.7;
        let d = Dual::<1>::var(x, 0);

        assert_close(d.sin().grad[0], x.cos());
        assert_close(d.cos().grad[0], -x.sin());
        assert_close(d.tan().grad[0], 1.0 / x.cos().powi(2));
        assert_close(d.exp().grad[0], x.exp());
        assert_close(d.ln().grad[0], 1.0 / x);
        assert_close(d.log2().grad[0], 1.0 / (x * std::f64::consts::LN_2));
        assert_close(d.log10().grad[0], 1.0 / (x * std::f64::consts::LN_10));
        assert_close(d.sqrt().grad[0], 0.5 / x.sqrt());
        assert_close(d.recip().grad[0], -1.0 / (x * x));
        assert_close(d.powf(3.0).grad[0], 3.0 * x * x);
        assert_close(d.powi(3).grad[0], 3.0 * x * x);
    }

    #[test]
    fn activations() {
        let x = 0.3;
        let d = Dual::<1>::var(x, 0);

        let t = x.tanh();
        assert_close(d.tanh().grad[0], 1.0 - t * t);

        let s = 1.0 / (1.0 + (-x).exp());
        assert_close(d.sigmoid().grad[0], s * (1.0 - s));
        assert_close(d.logistic().grad[0], s * (1.0 - s));

        assert_close(Dual::<1>::var(2.0, 0).relu().grad[0], 1.0);
        assert_close(Dual::<1>::var(-2.0, 0).relu().grad[0], 0.0);
        assert_close(Dual::<1>::var(-2.0, 0).abs().grad[0], -1.0);
    }

    #[test]
    fn multivariate_chain() {
        let x = Dual::<2>::var(0.5, 0);
        let y = Dual::<2>::var(1.5, 1);
        let z = (x * y + x.sin()).exp();
        assert_close(z.grad[0], z.value * (1.5 + 0.5f64.cos()));
        assert_close(z.grad[1], z.value * 0.5);
    }
}
