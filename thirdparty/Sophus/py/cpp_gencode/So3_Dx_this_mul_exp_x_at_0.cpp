Scalar const c0 = 0.5 * q.w();
Scalar const c1 = 0.5 * q.z();
Scalar const c2 = 0.5 * q.y();
Scalar const c3 = -c2;
Scalar const c4 = 0.5 * q.x();
Scalar const c5 = -c4;
Scalar const c6 = -c1;
result[0] = c0;
result[1] = c1;
result[2] = c3;
result[3] = c5;
result[4] = c6;
result[5] = c0;
result[6] = c4;
result[7] = c3;
result[8] = c2;
result[9] = c5;
result[10] = c0;
result[11] = c6;
