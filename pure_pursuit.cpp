#include <bits/stdc++.h>
#include <raylib.h>

constexpr double Sqr(double x) {
  return x * x;
}

struct Vec {
  double x = 0, y = 0;
  Vec() = default;
  Vec(double x, double y) : x(x), y(y) {}
  static Vec FromUnit(double a) {
    return Vec(std::cos(a), std::sin(a));
  }
  double Len() const {
    return sqrt(x * x + y * y);
  }
  double DistanceToPoint(const Vec &a) const {
    return sqrt(Sqr(x - a.x) + Sqr(y - a.y));
  }
  Vec operator+ (const Vec &a) const {
    return Vec(x + a.x, y + a.y);
  }
  Vec operator- (const Vec &a) const {
    return Vec(x - a.x, y - a.y);
  }
  void operator+= (const Vec &a) {
    x += a.x;
    y += a.y;
  }
  void operator-= (const Vec &a) {
    x -= a.x;
    y -= a.y;
  }
  Vec operator* (double t) const {
    return Vec(x * t, y * t);
  }
  Vec operator/ (double t) const {
    return Vec(x / t, y / t);
  }
  void operator*= (double t) {
    x *= t;
    y *= t;
  }
  void operator/= (double t) {
    x /= t;
    y /= t;
  }
  double Inner(const Vec &a) const {
    return x * a.x + y * a.y;
  }
  double Cross(const Vec &a) const {
    return x * a.y - y * a.x;
  }
  Vec ComplexMul(const Vec &a) const {
    return Vec(x * a.x - y * a.y, x * a.y + y * a.x);
  }
  Vec ComplexMul(double ax, double ay) const {
    return Vec(x * ax - y * ay, x * ay + y * ax);
  }
  Vec Rotate90() const {
    return Vec(-y, x);
  }
};

struct Pose {
  Vec position;
  double heading = 0;
};

Pose ComputeNextPose(const Pose &pose, double signed_curvature, double signed_distance) {
  Pose next_pose;
  double heading_change = signed_curvature * signed_distance;
  next_pose.heading = pose.heading + heading_change;
  if (std::abs(heading_change) < 1e-6) {
    next_pose.position = pose.position + Vec::FromUnit(pose.heading) * signed_distance;
    return next_pose;
  }
  double signed_radius = 1.0 / signed_curvature;
  next_pose.position = pose.position + Vec::FromUnit(pose.heading) * signed_radius * sin(heading_change) +
      Vec::FromUnit(pose.heading).Rotate90() * signed_radius * (1 - cos(heading_change));
  return next_pose;
}

std::vector<Pose> PurePursuit(const Pose& start_pose, const std::vector<Vec>& points) {
  std::vector<Pose> result;
  Pose curr_pose = start_pose;
  result.push_back(curr_pose);
  for (const Vec& point : points) {
    constexpr double kLookAHead = 5.0;
    while (true) {
      if (result.size() > 1000) {
        return result;
      }
      double l = curr_pose.position.DistanceToPoint(point);
      if (l < kLookAHead) {
        break;
      }
      Vec curr_dir = Vec::FromUnit(curr_pose.heading);
      Vec vec_to_point = point - curr_pose.position;
      // double sin_alpha = curr_dir.Cross(vec_to_point) / l;
      // double r = l / (2 * sin_alpha);  // l * l / (2 * curr_dir.Cross(vec_to_point))
      // double curvature = 1 / r;        // 2 * curr_dir.Cross(vec_to_point) / (l * l)
      double curvature = 2 * curr_dir.Cross(vec_to_point) / (l * l);
      curvature = std::clamp(curvature, -0.2, 0.2);
      Pose next_pose = ComputeNextPose(curr_pose, curvature, 0.2);
      result.push_back(next_pose);
      curr_pose = next_pose;
    }
  }
  return result;
}

struct Painter {
  double width = 50;
  double height = 50;
  double scale = 20;
  struct Segment {
    Vec a, b;
    Color color;
  };
  std::vector<Segment> segments;
  void AddSegment(const Vec &a, const Vec &b, const Color &color) {
    segments.emplace_back(a, b, color);
  }
  void Draw() {
    InitWindow(width * scale, height * scale, "window");
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
      BeginDrawing();
      ClearBackground(RAYWHITE);
      for (const auto& segment : segments) {
        DrawLine(segment.a.x * scale, (width - segment.a.y) * scale,
          segment.b.x * scale, (width - segment.b.y) * scale, segment.color);
      }
      EndDrawing();
    }
    CloseWindow();
  }
};

int main() {
  std::vector<Vec> points;
  for (int i = 20; i < 50; ++i) {
    points.emplace_back(i, 20);
  }
  Painter painter;
  for (int i = 0; i + 1 < (int)points.size(); ++i) {
    painter.AddSegment(points[i], points[i + 1], BLACK);
  }
  
  Pose start_pose;
  start_pose.position = Vec(20, 10);
  start_pose.heading = 0.0;
  
  std::vector<Pose> result = PurePursuit(start_pose, points);
  for (int i = 0; i + 1 < (int)result.size(); ++i) {
    painter.AddSegment(result[i].position, result[i + 1].position, RED);
  }
  
  painter.Draw();
  return 0;
}




