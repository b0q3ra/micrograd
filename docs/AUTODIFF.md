# Diferenciación automática (autodiff)

Guía de cómo funciona este motor (`micrograd`) y, sobre todo, de la teoría detrás
de la diferenciación automática en modo **forward** y **backward**, con recursos
para profundizar.

---

## 1. ¿Qué es la diferenciación automática?

La **diferenciación automática** (AD, *automatic/algorithmic differentiation*) es
un conjunto de técnicas para calcular derivadas **exactas** de funciones
expresadas como programas, evaluando el programa y aplicando mecánicamente la
regla de la cadena sobre las operaciones elementales. No es lo mismo que:

| Técnica | Cómo funciona | Problema |
|---|---|---|
| **Diferenciación numérica** | diferencias finitas: `f'(x) ≈ (f(x+h) − f(x−h)) / 2h` | error de truncamiento y de redondeo; catastrófico con muchas variables (`2n` evaluaciones) |
| **Diferenciación simbólica** | manipula expresiones algebraicas (`d/dx`) | explosión de la expresión (duplicación de subexpresiones); necesita la fórmula cerrada |
| **Diferenciación automática** | aplica la regla de la cadena op por op sobre los valores | exacta hasta precisión de máquina y coste comparable a evaluar `f` |

La clave es: **cualquier programa es una composición de operaciones elementales**
(suma, producto, `sin`, `exp`, ...) de las que conocemos su derivada. Si
registramos esas operaciones y aplicamos la regla de la cadena, obtenemos la
derivada sin aproximar.

### Regla de la cadena y grafo computacional

Para una composición `y = f(g(x))`:

```
dy/dx = (dy/dg) · (dg/dx)
```

En general, si `y` depende de variables intermedias `v_i`, cada una derivable,
entonces:

```
dy/dx = Σ_i (∂y/∂v_i) · (∂v_i/∂x)
```

El programa define implícitamente un **grafo computacional dirigido acíclico**
(DAG): las hojas son entradas/constantes, los nodos intermedios son operaciones y
la raíz es la salida. AD propaga derivadas por ese grafo, y la dirección en que lo
hace define los dos modos.

---

## 2. Modo forward (tangente / JVP)

Se recorre el grafo **desde las entradas hacia la salida**, propagando a la vez el
valor y su derivada. La derivada que se propaga es un *tangente* (dirección).

**Números duales.** La forma más elegante de hacer forward mode con una sola
variable es el **número dual**:

```
x = a + b·ε,   con ε² = 0  (ε es un infinitésimo nilpotente)
```

Al evaluar cualquier función suave `f` sobre un dual con `b = 1`, la parte `ε`
del resultado es exactamente `f'(a)`:

```
f(a + ε) = f(a) + f'(a)·ε
```

Ejemplo con `f(x) = sin(x)`:

```
sin(a + ε) = sin(a)·cos(ε) + cos(a)·sin(ε)
           = sin(a)·1 + cos(a)·ε        (cos(ε)=1, sin(ε)=ε para ε²=0)
           = sin(a) + cos(a)·ε
```

Por tanto la parte `ε` es `cos(a) = f'(a)`. Para `N` variables de entrada, el
tangente es un vector de `N` componentes (un **Jacobiano-vector producto**, JVP):
cada *semilla* unitaria da una columna del Jacobiano.

**Coste.** Para `f : ℝⁿ → ℝᵐ`, un barrido forward calcula un JVP, así que
necesitas `n` barridos para el Jacobiano completo: `O(n · ops(f))`.

---

## 3. Modo backward (adjunto / VJP / backprop)

Se hace primero un **barrido forward** que evalúa y **registra** el grafo, y luego
un **barrido reverse** desde la salida hacia las entradas propagando **adjuntos**.

El adjunto de una variable `v_i` respecto a la salida `y` es:

```
v̄_i = ∂y / ∂v_i        (sensibilidad de y a cambios en v_i)
```

Regla de propagación (regla de la cadena "hacia atrás"): si `v_i` alimenta a `v_j`,
cada consumidor aporta su contribución al adjunto del productor:

```
v̄_i = Σ_{j : v_i → v_j}  v̄_j · (∂v_j / ∂v_i)
```

Se arranca con `ȳ = ∂y/∂y = 1` y se recorre el grafo en orden topológico inverso.

**Coste.** Para `f : ℝⁿ → ℝᵐ`, un único barrido reverse obtiene una fila del
Jacobiano (un **Jacobiano-vector-producto traspuesto**, VJP). El Jacobiano
completo cuesta `O(m · ops(f))`.

**Caso estrella `f : ℝⁿ → ℝ`** (la pérdida escalar de una red neuronal): **un solo
barrido** da el gradiente completo `∇f` con respecto a los `n` parámetros. Esto es
exactamente **backpropagation**.

---

## 4. Forward vs backward: cuándo usar cada uno

| | Forward mode | Backward mode |
|---|---|---|
| Dirección | entradas → salidas | salidas → entradas |
| Unidad que propaga | tangente (JVP) | adjunto (VJP) |
| Coste Jacobiano `f: ℝⁿ→ℝᵐ` | `O(n · ops(f))` | `O(m · ops(f))` |
| Barridos para el gradiente completo (pérdida escalar, muchos params) | `n` | `1` |
| Memoria | baja (se puede hacer *on the fly*) | alta (guardar el grafo/activaciones) |
| Implementación | sobrecarga de operadores / números duales | *tapes* / grafo diferencial |
| Uso típico | pocas entradas, muchas salidas; Hessianos; física/CFD | redes neuronales (muchos params, salida escalar) |

Regla mnemotécnica: **si `n ≫ m`, usa backward; si `n ≪ m`, usa forward.**
Este repo implementa **forward** con `N` tangentes.

---

## 5. Este repositorio: `Dual<const N: usize>`

El motor vive en `src/autodiff.rs`. La estructura es un número dual con `N`
tangentes simultáneos (un vector fila del Jacobiano):

```rust
pub struct Dual<const N: usize> {
    pub value: f64,      // valor (la parte "real")
    pub grad: [f64; N],  // la parte ε: N derivadas direccionales
}
```

### Semillas y constantes

```rust
let x = Dual::<2>::var(0.5, 0);   // ∂/∂x₀ = [1, 0], value = 0.5
let y = Dual::<2>::var(1.5, 1);   // ∂/∂x₁ = [0, 1], value = 1.5
let c = Dual::<2>::constant(2.0); // grad = [0, 0]
```

`var(v, i)` enciende la componente `i` a `1.0`: es una semilla (producto
Jacobiano-vector con el vector base `eᵢ`). Con `N` igual al número de variables,
un único cálculo forward da el gradiente completo de una función de `N` variables.
(Por eso en este repo el "coste" es que cada variable necesita su propio índice,
a diferencia del backward mode que se usa en las redes reales.)

### Operadores

Implementados en `src/autodiff.rs:114` en adelante como `Add`, `Sub`, `Mul`,
`Div`, `Neg` en tres variantes (`Dual op Dual`, `Dual op f64`, `f64 op Dual`),
más los `*Assign`. Las reglas son las habituales; nótese cómo la multiplicación
encadena ambos tangentes (regla del producto):

```
(a*b).value = a.value * b.value
(a*b).grad  = a.grad * b.value + a.value * b.grad
```

### Funciones elementales y sus derivadas

Definidas en `src/autodiff.rs:30`–`111`. La ayuda `unary(value, derivative)`
aplica `grad[i] *= derivative`:

| Método | `value` | derivada usada |
|---|---|---|
| `sin` | `sin v` | `cos v` |
| `cos` | `cos v` | `−sin v` |
| `tan` | `tan v` | `1/cos²v` |
| `exp` | `eᵛ` | `eᵛ` |
| `ln` | `ln v` | `1/v` |
| `log(b)` | `log_b v` | `1/(v·ln b)` |
| `log2`, `log10` | — | `1/(v·ln 2)`, `1/(v·ln 10)` |
| `sqrt` | `√v` | `1/(2√v)` |
| `recip` | `1/v` | `−1/v²` |
| `powf(p)`, `powi(n)` | `vᵖ` | `p·vᵖ⁻¹` |
| `tanh` | `tanh v` | `1 − tanh²v` |
| `sigmoid` / `logistic` | `σ(v)` | `σ(v)(1−σ(v))` |
| `relu` | `max(0,v)` | `1` si `v>0`, si no `0` |
| `abs` | `|v|` | `sign(v)` |

### Demo

`src/main.rs` calcula `f(x) = sin(x)·exp(x)` y compara `f'(2)` con la derivada
analítica, y un ejemplo multivariable `sigmoid(a·b + sin(a))` con sus dos
derivadas parciales.

### Limitaciones y extensiones

- **Memoria/rendimiento**: forward mode escala linealmente con el nº de parámetros;
  para redes grandes lo correcto es reverse mode (ver §3).
- `relu`/`abs` no son derivables en `0`; aquí se elige una subderivada.
- Dominios: `ln`, `sqrt`, `log` con `v ≤ 0` dan `NaN`/`−inf` (no se validan).
- Extensiones naturales: un `Value` escalar con grafo + `.backward()` (reverse),
  `softmax`, operaciones entre arrays/matrices, y un *tape* para no fijar `N`.

---

## 6. Aplicación a redes neuronales

Una red es una función escalar composición de operaciones elementales
(`f: ℝⁿ → ℝ`, `n` = nº de parámetros). Con forward mode necesitarías `n` barridos;
con **backward mode un único barrido** da `∇_θ L`, que es lo que necesita el
descenso de gradiente:

```
θ ← θ − η · ∇_θ L(θ)
```

### Neurona y capa

```
z   = Σ_j w_j x_j + b          (preactivación)
a   = σ(z)                     (activación)
```

El grafo de una neurona es: productos → suma → sesgo → `sigmoid`. Cada nodo
guarda su `value`; el barrido reverse multiplica adjuntos locales (§3).

### Backprop en una capa

Con `δ = ∂L/∂z` (adjunto de la preactivación), los gradientes son:

```
∂L/∂w_j = δ · x_j
∂L/∂b   = δ
∂L/∂x_j = δ · w_j          (para propagar hacia la capa anterior)
```

y `δ` se obtiene del error de la capa siguiente: `δ = (Wᵀ δ_next) ⊙ σ'(z)`.

### Pérdidas y activaciones típicas

- Clasificación: `softmax` + *cross-entropy*. `∂L/∂logits = ŷ − y`.
- Regresión: error cuadrático `L = ½‖ŷ − y‖²`, `∂L/∂ŷ = ŷ − y`.
- Activaciones: `relu` (barata, evita saturación), `sigmoid`/`tanh` (se saturan),
  `softmax` (salida probabilística).

Un `autograd` (como JAX/PyTorch) no programa estas reglas a mano por red: aplica
reverse mode sobre el grafo que el propio programa de la red construye. Este
`micrograd` hace lo mismo en modo forward.

### Ejemplo mínimo de entrenamiento (pseudocódigo)

```text
para cada epoch:
    ŷ = forward(X, θ)          # registra el grafo (o N tangentes en forward)
    L = loss(ŷ, y)
    ∇θ = backward(L)           # reverse en un barrido
    θ ← θ − η · ∇θ
```

---

## 7. Recursos

### Papers descargados en `docs/resources/`

- **`baydin-2018-autodiff-survey.pdf`** — Baydin, Pearlmutter, Radul & Siskind,
  *Automatic Differentiation in Machine Learning: a Survey*, JMLR 18(153), 2018.
  La referencia introductoria por excelencia: forward vs reverse, implementación y
  aplicaciones en ML. (arXiv:1502.05767)
- **`pearlmutter-siskind-reverse-mode-framework.pdf`** — Pearlmutter & Siskind,
  *Reverse-Mode AD in a Functional Framework: Lambda the Ultimate Backpropagator*,
  TOPLAS 2008. Base teórica/PL del reverse mode. (arXiv:0704.0886)
- **`carpenter-2015-stan-math-reverse-mode.pdf`** — Carpenter et al.,
  *The Stan Math Library: Reverse-Mode Automatic Differentiation in C++*, 2015.
  Implementación real de un *tape* de reverse mode. (arXiv:1509.07164)
- **`cs231n-backprop-notes.html`** — Stanford CS231n, *Backpropagation, Intuitions*.
  Derivación práctica de backprop en redes, con el ejemplo `sigmoid`.
- **`jax-autodiff-cookbook.html`** — JAX, *The Autodiff Cookbook*: `grad`, `jvp`,
  `vjp`, forward vs reverse con código.
- **`grosse-csc321-backprop-lecture.pdf`** — Roger Grosse (U. Toronto), *CSC321
  Lecture 6: Backpropagation*. Notas de clase muy claras, con el grafo de cómputo.

### Enlaces online (no descargados)

- Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation*
  (video "micrograd"): <https://www.youtube.com/watch?v=VMj-3S1tku0>
- Karpathy, *Neural Networks: Zero to Hero*: <https://karpathy.ai/zero-to-hero.html>
- Autograd (HIPS): <https://github.com/HIPS/autograd>
- Griewank & Walther, *Evaluating Derivatives* (libro, SIAM 2008) — la referencia
  canónica en AD. No libre; enlaces en <https://www.autodiff.org/>
- `autodiff.org` (Community Portal for AD): <https://www.autodiff.org/>
- PyTorch autograd: <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>
- TensorFlow *Introduction to gradients*:
  <https://www.tensorflow.org/guide/autodiff>

---

## 8. Cómo ejecutar

```bash
cargo test     # 5 tests: aritmética, escalares, elementales, activaciones, cadena multivariable
cargo run      # demo con comparación de derivada analítica
```
