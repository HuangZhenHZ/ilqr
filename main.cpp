#include <bits/stdc++.h>

template <int N, int M>
struct Matrix {
  double a[N][M];
  
  Matrix() {
    memset(a, 0, sizeof(a));
  }
  double* operator[] (int x) {
    return a[x];
  }
  const double* operator[] (int x) const {
    return a[x];
  }
  void print() const {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        printf("%lf", a[i][j]);
        putchar(j < M-1 ? ' ' : '\n');
      }
    }
  }
  
  [[nodiscard]] Matrix<M,N> transpose() const {
    Matrix<M,N> b;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        b[j][i] = a[i][j];
      }
    }
    return b;
  }
  
  Matrix operator+ (const Matrix &b) const {
    Matrix c;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        c[i][j] = a[i][j] + b[i][j];
      }
    }
    return c;
  }
  
  Matrix operator- (const Matrix &b) const {
    Matrix c;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        c[i][j] = a[i][j] - b[i][j];
      }
    }
    return c;
  }
  
  Matrix& operator+= (const Matrix &b) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        a[i][j] += b[i][j];
      }
    }
    return *this;
  }
  
  Matrix& operator-= (const Matrix &b) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        a[i][j] -= b[i][j];
      }
    }
    return *this;
  }
  
  Matrix operator* (const double t) const {
    Matrix b;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        b[i][j] = a[i][j] * t;
      }
    }
    return b;
  }
  
  std::optional<Matrix> ComputeLLT() const {
    static_assert(N == M);
    Matrix l;
    for (int k = 0; k < N; ++k) {
      double tmp = a[k][k];
      for (int i = 0; i < k; ++i) {
        tmp -= l[k][i] * l[k][i];
      }
      if (tmp < 1e-6) {
        return std::nullopt;
      }
      l[k][k] = sqrt(tmp);
      
      for (int i = k + 1; i < N; ++i) {
        double tmp = a[i][k];
        for (int j = 0; j < k; ++j) {
          tmp -= l[i][j] * l[k][j];
        }
        l[i][k] = tmp / l[k][k];
      }
    }
    return l;
  }
};

template <int N, int K, int M>
Matrix<N,M> operator* (const Matrix<N,K> &a, const Matrix<K,M> &b) {
  Matrix<N,M> c;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < K; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

// Solve l * l^t * x = b
template <int N, int M>
Matrix<N,M> SolveUsingLLT(const Matrix<N,N> &l, Matrix<N,M> b) {
  for (int k = 0; k < M; ++k) {
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        b[j][k] -= b[i][k] / l[i][i] * l[j][i];
      }
      b[i][k] /= l[i][i];
    }
    
    for (int i = N - 1; i >= 0; --i) {
      for (int j = 0; j < i; ++j) {
        b[j][k] -= b[i][k] / l[i][i] * l[i][j];
      }
      b[i][k] /= l[i][i];
    }
  }
  return b;
}

template <int XDim, int UDim>
class DynamicModel {
public:
  using VectorX = Matrix<XDim, 1>;
  using VectorU = Matrix<UDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  using MatrixXU = Matrix<XDim, UDim>;
  virtual ~DynamicModel() = default;
  virtual VectorX ComputeNextState(const VectorX &x, const VectorU &u) const;
  virtual std::pair<MatrixXX, MatrixXU> ComputeDerivatives(const VectorX &x, const VectorU &u) const;
};

class MyDynamicModel : public DynamicModel<2,2> {
public:
  ~MyDynamicModel() override = default;
  VectorX ComputeNextState(const VectorX &x, const VectorU &u) const override {
    return x + u;
  }
  std::pair<MatrixXX, MatrixXU> ComputeDerivatives(const VectorX &, const VectorU &) const override {
    MatrixXX fx;
    fx[0][0] = fx[1][1] = 1;
    return std::make_pair(fx, fx);
  }
};

template <int XDim, int UDim>
class TransitionCost {
public:
  using VectorX = Matrix<XDim, 1>;
  using VectorU = Matrix<UDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  using MatrixUX = Matrix<UDim, XDim>;
  using MatrixUU = Matrix<UDim, UDim>;
  virtual ~TransitionCost() = default;
  struct Derivatives {
    VectorX df_dx;
    VectorU df_du;
    MatrixXX d2f_dx_dx;
    MatrixUX d2f_du_dx;
    MatrixUU d2f_du_du;
  };
  virtual double ComputeCostAndDerivatives(const VectorX &x, const VectorU &u, Derivatives *derivatives) const;
};

class MyTransitionCost : public TransitionCost<2,2> {
public:
  ~MyTransitionCost() override = default;
  double ComputeCostAndDerivatives(const VectorX &, const VectorU &u, Derivatives *derivatives) const override {
    double cost = (u.transpose() * u)[0][0];
    if (derivatives) {
      *derivatives = Derivatives();
      derivatives->df_du[0][0] = u[0][0] * 2.0;
      derivatives->df_du[1][0] = u[1][0] * 2.0;
      derivatives->d2f_du_du[0][0] = 2.0;
      derivatives->d2f_du_du[1][1] = 2.0;
    }
    return cost;
  }
};

template <int XDim>
class StateCost {
public:
  using VectorX = Matrix<XDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  virtual ~StateCost() = default;
  struct Derivatives {
    VectorX df_dx;
    MatrixXX d2f_dx_dx;
  };
  virtual double ComputeCostAndDerivatives(const VectorX &x, Derivatives *derivatives) const;
};

class MyFinalCost : public StateCost<2> {
public:
  ~MyFinalCost() override = default;
  double ComputeCostAndDerivatives(const VectorX &x, Derivatives *derivatives) const override {
    double cost = (x[0][0] - 100) * (x[0][0] - 100) + (x[1][0] - 100) * (x[1][0] - 100);
    if (derivatives) {
      *derivatives = Derivatives();
      derivatives->df_dx[0][0] = 2.0 * (x[0][0] - 100);
      derivatives->df_dx[1][0] = 2.0 * (x[1][0] - 100);
      derivatives->d2f_dx_dx[0][0] = 2.0;
      derivatives->d2f_dx_dx[1][1] = 2.0;
    }
    return cost;
  }
};

template <int XDim, int UDim>
class Solver {
public:
  using VectorX = Matrix<XDim, 1>;
  using VectorU = Matrix<UDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  using MatrixUX = Matrix<UDim, XDim>;
  using MatrixUU = Matrix<UDim, UDim>;
  using MatrixXU = Matrix<XDim, UDim>;
  using DynamicModel = DynamicModel<XDim, UDim>;
  using TransitionCost = TransitionCost<XDim, UDim>;
  using StateCost = StateCost<XDim>;
  using Q = TransitionCost::Derivatives;
  using V = StateCost::Derivatives;

  static bool ComputeKV(const Q &q, VectorU *ku, MatrixUX *kux, V *v) {
    assert(ku != nullptr);
    assert(kux != nullptr);
    assert(v != nullptr);
    std::optional<MatrixUU> l = q.d2f_du_du.ComputeLLT();
    if (l == std::nullopt) {
      return false;
    }
    *ku = SolveUsingLLT(*l, q.df_du) * -1.0;
    *kux = SolveUsingLLT(*l, q.d2f_du_dx) * -1.0;
    v->df_dx = q.df_dx +
               kux->transpose() * q.d2f_du_du * *ku +
               kux->transpose() * q.df_du +
               q.d2f_du_dx.transpose() * *ku;
    v->d2f_dx_dx = q.d2f_dx_dx +
                   kux->transpose() * q.d2f_du_du * *kux +
                   kux->transpose() * q.d2f_du_dx +
                   q.d2f_du_dx.transpose() * *kux;
    return true;
  }
  
  static Q ComputeQ(const V &v, const MatrixXX &fx, const MatrixXU &fu) {
    return Q {
      .df_dx = fx.transpose() * v.df_dx,
      .df_du = fu.transpose() * v.df_dx,
      .d2f_dx_dx = fx.transpose() * v.d2f_dx_dx * fx,
      .d2f_du_dx = fu.transpose() * v.d2f_dx_dx * fx,
      .d2f_du_du = fu.transpose() * v.d2f_dx_dx * fu,
    };
  }
  
  void Solve() {
    assert(x_.size() == 1);
    assert(u_.size() >= 1);
    assert(dynamic_model_ != nullptr);
    assert(transition_cost_ != nullptr);
    assert(final_cost_ != nullptr);
    for (const auto& u : u_) {
      x_.push_back(dynamic_model_->ComputeNextState(x_.back(), u));
    }
    int n = u_.size();
    q_.resize(n);
    v_.resize(n + 1);
    ku_.resize(n);
    kux_.resize(n);
    for (int iter = 0; iter < 5; ++iter) {
      final_cost_->ComputeCostAndDerivatives(x_[n], &v_[n]);
      for (int i = n - 1; i >= 0; --i) {
        std::pair<MatrixXX, MatrixXU> fx_and_fu = dynamic_model_->ComputeDerivatives(x_[i], u_[i]);
        q_[i] = ComputeQ(v_[i + 1], fx_and_fu.first, fx_and_fu.second);
        Q derivatives;
        transition_cost_->ComputeCostAndDerivatives(x_[i], u_[i], &derivatives);
        q_[i].df_dx += derivatives.df_dx;
        q_[i].df_du += derivatives.df_du;
        q_[i].d2f_dx_dx += derivatives.d2f_dx_dx;
        q_[i].d2f_du_dx += derivatives.d2f_du_dx;
        q_[i].d2f_du_du += derivatives.d2f_du_du;
        
        if (!ComputeKV(q_[i], &ku_[i], &kux_[i], &v_[i])) {
          printf("end at iter = %d\n", iter);
          return;
        }
      }
      
      std::vector<VectorX> last_x = x_;
      
      for (int i = 0; i < n; ++i) {
        u_[i] += ku_[i] + kux_[i] * (x_[i] - last_x[i]);
        x_[i + 1] = dynamic_model_->ComputeNextState(x_[i], u_[i]);
      }
      
      printf("iter = %d\n", iter);
      for (int i = 0; i < (int)x_.size(); ++i) {
        printf("x_[%d]\n", i);
        x_[i].print();
      }
      for (int i = 0; i < (int)u_.size(); ++i) {
        printf("u_[%d]\n", i);
        u_[i].print();
      }
      for (int i = 0; i < (int)v_.size(); ++i) {
        printf("vx %d\n", i);
        v_[i].df_dx.print();
        printf("vxx %d\n", i);
        v_[i].d2f_dx_dx.print();
      }
    }
  }

// private:
  std::vector<VectorX> x_;
  std::vector<VectorU> u_;
  std::vector<Q> q_;
  std::vector<V> v_;
  std::vector<VectorU> ku_;
  std::vector<MatrixUX> kux_;
  std::unique_ptr<DynamicModel> dynamic_model_;
  std::unique_ptr<TransitionCost> transition_cost_;
  std::unique_ptr<StateCost> final_cost_;
};


int main() {
  /*
  Matrix<2,2> a;
  Matrix<2,2> b;
  a[0][0] = 1;
  a[1][1] = 1;
  b[0][1] = 1;
  Matrix<2,2> c = a * b;
  c.print();
   */
  
  Matrix<3,3> a;
  a[0][0] = 4;
  a[0][1] = 12;
  a[0][2] = -16;
  a[1][0] = 12;
  a[1][1] = 37;
  a[1][2] = -43;
  a[2][0] = -16;
  a[2][1] = -43;
  a[2][2] = 98;
  
  a.print();
  
  std::optional<Matrix<3,3>> l = a.ComputeLLT();
  if (l) {
    l->print();
  } else {
    printf("failed\n");
  }
  
  Matrix<3,2> b;
  b[0][0] = 1;
  b[1][0] = 1;
  b[2][0] = 1;
  b[0][1] = 1;
  b[1][1] = 2;
  b[2][1] = 3;
  
  Matrix<3,2> x = SolveUsingLLT(*l, b);
  x.print();
  
  (a * x).print();
  
  using Solver = Solver<2, 2>;
  
  {
    Solver::Q q;
    q.df_dx[0][0] = -200;
    q.df_dx[1][0] = -200;
    q.df_du = q.df_dx;
    q.d2f_dx_dx[0][0] = 2;
    q.d2f_dx_dx[1][1] = 2;
    q.d2f_du_dx = q.d2f_dx_dx;
    q.d2f_du_du[0][0] = 4;
    q.d2f_du_du[1][1] = 4;
    Solver::V v;
    Solver::VectorU ku;
    Solver::MatrixUX kux;    
    if (Solver::ComputeKV(q, &ku, &kux, &v)) {
      v.df_dx.print();
      v.d2f_dx_dx.print();
    }
  }
  
  {
    Solver::V v;
    v.df_dx[0][0] = -200;
    v.df_dx[1][0] = -200;
    v.d2f_dx_dx[0][0] = 2;
    v.d2f_dx_dx[1][1] = 2;
    Solver::MatrixXX fx;
    fx[0][0] = 1;
    fx[1][1] = 1;
    Solver::MatrixXU fu;
    fu[0][0] = 1;
    fu[1][1] = 1;
    Solver::Q q = Solver::ComputeQ(v, fx, fu);
    q.df_dx.print();
    q.df_du.print();
    q.d2f_dx_dx.print();
    q.d2f_du_dx.print();
    q.d2f_du_du.print();
  }
  
  Solver solver;
  solver.dynamic_model_ = std::make_unique<MyDynamicModel>();
  solver.transition_cost_ = std::make_unique<MyTransitionCost>();
  solver.final_cost_ = std::make_unique<MyFinalCost>();
  solver.x_.push_back(Solver::VectorX());
  solver.u_.push_back(Solver::VectorU());
  solver.u_.push_back(Solver::VectorU());
  solver.Solve();
  for (const auto& x : solver.x_) {
    x.print();
  }
  for (const auto& u : solver.u_) {
    u.print();
  }
  return 0;
}

