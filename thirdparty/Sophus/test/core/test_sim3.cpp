#include <iostream>

#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/sim3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Sim3<double>>;
template class Map<Sophus::Sim3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class Sim3<double, Eigen::AutoAlign>;
template class Sim3<float, Eigen::DontAlign>;

template <class Scalar>
class Tests {
 public:
  using Sim3Type = Sim3<Scalar>;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename Sim3<Scalar>::Point;
  using Vector4Type = Vector4<Scalar>;
  using Tangent = typename Sim3<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    sim3_vec_.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 1.)),
                                 Point(0, 0, 0)));
    sim3_vec_.push_back(Sim3Type(
        RxSO3Type::exp(Vector4Type(0.2, 0.5, -1.0, 1.1)), Point(10, 0, 0)));
    sim3_vec_.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 0.)), Point(0, 10, 5)));
    sim3_vec_.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 1.1)),
                                 Point(0, 10, 5)));
    sim3_vec_.push_back(Sim3Type(
        RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.)), Point(0, 0, 0)));
    sim3_vec_.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.0000001)),
                 Point(1, -1.00000001, 2.0000000001)));
    sim3_vec_.push_back(Sim3Type(
        RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0)), Point(0.01, 0, 0)));
    sim3_vec_.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(kPi, 0, 0, 0.9)), Point(4, -5, 0)));
    sim3_vec_.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 0)),
                 Point(0, 0, 0)) *
        Sim3Type(RxSO3Type::exp(Vector4Type(kPi, 0, 0, 0)), Point(0, 0, 0)) *
        Sim3Type(RxSO3Type::exp(Vector4Type(-0.2, -0.5, -0.0, 0)),
                 Point(0, 0, 0)));
    sim3_vec_.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(0.3, 0.5, 0.1, 0)),
                 Point(2, 0, -7)) *
        Sim3Type(RxSO3Type::exp(Vector4Type(kPi, 0, 0, 0)), Point(0, 0, 0)) *
        Sim3Type(RxSO3Type::exp(Vector4Type(-0.3, -0.5, -0.1, 0)),
                 Point(0, 6, 0)));
    Tangent tmp;
    tmp << 0, 0, 0, 0, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 1, 0, 0, 0, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 0, 1, 0, 1, 0, 0, 0.1;
    tangent_vec_.push_back(tmp);
    tmp << 0, 0, 1, 0, 1, 0, 0.1;
    tangent_vec_.push_back(tmp);
    tmp << -1, 1, 0, 0, 0, 1, -0.1;
    tangent_vec_.push_back(tmp);
    tmp << 20, -1, 0, -1, 1, 0, -0.1;
    tangent_vec_.push_back(tmp);
    tmp << 30, 5, -1, 20, -1, 0, 1.5;
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(1, 2, 4));
    point_vec_.push_back(Point(1, -3, 0.5));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<Sim3Type> tests(sim3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 7, 1> raw;
    raw << 0, 1, 0, 0, 1, 3, 2;
    Eigen::Map<Sim3Type const> map_of_const_sim3(raw.data());
    SOPHUS_TEST_APPROX(passed, map_of_const_sim3.quaternion().coeffs().eval(),
                       raw.template head<4>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, map_of_const_sim3.translation().eval(),
                       raw.template tail<3>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_const_sim3.quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_EQUAL(passed, map_of_const_sim3.translation().data(),
                      raw.data() + 4);
    Eigen::Map<Sim3Type const> const_shallow_copy = map_of_const_sim3;
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.quaternion().coeffs().eval(),
                      map_of_const_sim3.quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.translation().eval(),
                      map_of_const_sim3.translation().eval());

    Eigen::Matrix<Scalar, 7, 1> raw2;
    raw2 << 1, 0, 0, 0, 3, 2, 1;
    Eigen::Map<Sim3Type> map_of_sim3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2.template head<4>();
    map_of_sim3.setQuaternion(quat);
    map_of_sim3.translation() = raw2.template tail<3>();
    SOPHUS_TEST_APPROX(passed, map_of_sim3.quaternion().coeffs().eval(),
                       raw2.template head<4>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, map_of_sim3.translation().eval(),
                       raw2.template tail<3>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_sim3.quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_EQUAL(passed, map_of_sim3.translation().data(), raw.data() + 4);
    SOPHUS_TEST_NEQ(passed, map_of_sim3.quaternion().coeffs().data(),
                    quat.coeffs().data());
    Eigen::Map<Sim3Type> shallow_copy = map_of_sim3;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.quaternion().coeffs().eval(),
                      map_of_sim3.quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, shallow_copy.translation().eval(),
                      map_of_sim3.translation().eval());
    Eigen::Map<Sim3Type> const const_map_of_sim3 = map_of_sim3;
    SOPHUS_TEST_EQUAL(passed, const_map_of_sim3.quaternion().coeffs().eval(),
                      map_of_sim3.quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, const_map_of_sim3.translation().eval(),
                      map_of_sim3.translation().eval());

    Sim3Type const const_sim3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_sim3.data()[i], raw2.data()[i]);
    }

    Sim3Type se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw2.data()[i]);
    }

    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 4> I = Eigen::Matrix<Scalar, 4, 4>::Identity();
    SOPHUS_TEST_EQUAL(passed, Sim3Type().matrix(), I);

    Sim3Type sim3 = sim3_vec_.front();
    Point translation = sim3.translation();
    RxSO3Type rxso3 = sim3.rxso3();

    SOPHUS_TEST_APPROX(passed, Sim3Type(rxso3, translation).matrix(),
                       sim3.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed,
                       Sim3Type(rxso3.quaternion(), translation).matrix(),
                       sim3.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, Sim3Type(sim3.matrix()).matrix(), sim3.matrix(),
                       Constants<Scalar>::epsilon());

    Scalar scale(1.2);
    sim3.setScale(scale);
    SOPHUS_TEST_APPROX(passed, scale, sim3.scale(),
                       Constants<Scalar>::epsilon(), "setScale");

    sim3.setQuaternion(sim3_vec_[0].rxso3().quaternion());
    SOPHUS_TEST_APPROX(passed, sim3_vec_[0].rxso3().quaternion().coeffs(),
                       sim3_vec_[0].rxso3().quaternion().coeffs(),
                       Constants<Scalar>::epsilon(), "setQuaternion");
    return passed;
  }

  std::vector<Sim3Type, Eigen::aligned_allocator<Sim3Type>> sim3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_sim3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Sim3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_sim3(); }
