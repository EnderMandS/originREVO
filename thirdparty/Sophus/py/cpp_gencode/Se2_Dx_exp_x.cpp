Scalar const c0 = sin(theta);
Scalar const c1 = 1.0 / theta;
Scalar const c2 = c0 * c1;
Scalar const c3 = cos(theta);
Scalar const c4 = -c3 + 1;
Scalar const c5 = c1 * c4;
Scalar const c6 = c1 * c3;
Scalar const c7 = pow(theta, -2);
Scalar const c8 = c0 * c7;
Scalar const c9 = c4 * c7;
result[0] = 0;
result[1] = 0;
result[2] = c2;
result[3] = c5;
result[4] = 0;
result[5] = 0;
result[6] = -c5;
result[7] = c2;
result[8] = -c0;
result[9] = c3;
result[10] =
    -c2 * upsilon[1] + c6 * upsilon[0] - c8 * upsilon[0] + c9 * upsilon[1];
result[11] =
    c2 * upsilon[0] + c6 * upsilon[1] - c8 * upsilon[1] - c9 * upsilon[0];
