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
  virtual ~DynamicModel();
  virtual VectorX ComputeNextState(const VectorX &x, const VectorU &u);
  virtual std::pair<MatrixXX, MatrixXU> ComputeDerivatives(const VectorX &x, const VectorU &u);
};

template <int XDim, int UDim>
class TransitionCost {
public:
  using VectorX = Matrix<XDim, 1>;
  using VectorU = Matrix<UDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  using MatrixUX = Matrix<UDim, XDim>;
  using MatrixUU = Matrix<UDim, UDim>;
  virtual ~TransitionCost();
  struct Derivatives {
    VectorX df_dx;
    VectorU df_du;
    MatrixXX d2f_dx_dx;
    MatrixUX d2f_du_dx;
    MatrixUU d2f_du_du;
  };
  virtual double ComputeCostAndDerivatives(const VectorX &x, const VectorU &u, Derivatives *derivatives);
};

template <int XDim>
class StateCost {
public:
  using VectorX = Matrix<XDim, 1>;
  using MatrixXX = Matrix<XDim, XDim>;
  virtual ~StateCost();
  struct Derivatives {
    VectorX df_dx;
    MatrixXX d2f_dx_dx;
  };
  virtual double ComputeCostAndDerivatives(const VectorX &x, Derivatives *derivatives);
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
  using TransitionCost = TransitionCost<XDim, UDim>;
  using StateCost = StateCost<XDim>;
  using Q = TransitionCost::Derivatives;
  using V = StateCost::Derivatives;

  static std::optional<V> ComputeV(const Q &q) {
    std::optional<MatrixUU> l = q.d2f_du_du.ComputeLLT();
    if (l == std::nullopt) {
      return std::nullopt;
    }
    VectorU ku = SolveUsingLLT(*l, q.df_du) * -1.0;
    MatrixUX kux = SolveUsingLLT(*l, q.d2f_du_dx) * -1.0;
    return V {
      .df_dx = q.df_dx + kux.transpose() * q.d2f_du_du * ku + kux.transpose() * q.df_du + q.d2f_du_dx * ku,
      .d2f_dx_dx = q.d2f_dx_dx + kux.transpose() * q.d2f_du_du * kux +
                   kux.transpose() * q.d2f_du_dx + q.d2f_du_dx.transpose() * kux,
    };
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

private:
  int n_;
  std::vector<VectorX> x_;
  std::vector<VectorU> u_;
  std::vector<Q> q_;
  std::vector<V> v_;
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
    std::optional<Solver::V> v = Solver::ComputeV(q);
    if (v) {
      v->df_dx.print();
      v->d2f_dx_dx.print();
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
  
  return 0;
}

