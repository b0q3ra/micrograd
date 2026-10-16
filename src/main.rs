use micrograd::Dual;

fn main() {
    let x = Dual::<1>::var(2.0, 0);
    let y = x.sin() * x.exp();
    println!("f(x) = sin(x) * exp(x)");
    println!("x    = {}", x.value);
    println!("f(x) = {}", y.value);
    println!("f'(x)= {}", y.grad[0]);

    let analytical = 2.0f64.sin() * 2.0f64.exp() + 2.0f64.cos() * 2.0f64.exp();
    println!("analítica = {analytical}");

    let a = Dual::<2>::var(0.5, 0);
    let b = Dual::<2>::var(1.5, 1);
    let z = (a * b + a.sin()).sigmoid();
    println!("\ng(a, b) = sigmoid(a * b + sin(a))");
    println!("∂g/∂a = {}", z.grad[0]);
    println!("∂g/∂b = {}", z.grad[1]);
}
