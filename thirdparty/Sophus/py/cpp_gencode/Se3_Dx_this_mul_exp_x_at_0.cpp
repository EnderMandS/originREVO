Scalar const c0 = pow(q.w(), 2);
Scalar const c1 = pow(q.x(), 2);
Scalar const c2 = pow(q.y(), 2);
Scalar const c3 = -c2;
Scalar const c4 = pow(q.z(), 2);
Scalar const c5 = -c4;
Scalar const c6 = 2 * q.w();
Scalar const c7 = c6 * q.z();
Scalar const c8 = 2 * q.x();
Scalar const c9 = c8 * q.y();
Scalar const c10 = c6 * q.y();
Scalar const c11 = c8 * q.z();
Scalar const c12 = c0 - c1;
Scalar const c13 = c6 * q.x();
Scalar const c14 = 2 * q.y() * q.z();
Scalar const c15 = 0.5 * q.w();
Scalar const c16 = 0.5 * q.z();
Scalar const c17 = 0.5 * q.y();
Scalar const c18 = -c17;
Scalar const c19 = 0.5 * q.x();
Scalar const c20 = -c19;
Scalar const c21 = -c16;
result[0] = 0;
result[1] = 0;
result[2] = 0;
result[3] = 0;
result[4] = c0 + c1 + c3 + c5;
result[5] = c7 + c9;
result[6] = -c10 + c11;
result[7] = 0;
result[8] = 0;
result[9] = 0;
result[10] = 0;
result[11] = -c7 + c9;
result[12] = c12 + c2 + c5;
result[13] = c13 + c14;
result[14] = 0;
result[15] = 0;
result[16] = 0;
result[17] = 0;
result[18] = c10 + c11;
result[19] = -c13 + c14;
result[20] = c12 + c3 + c4;
result[21] = c15;
result[22] = c16;
result[23] = c18;
result[24] = c20;
result[25] = 0;
result[26] = 0;
result[27] = 0;
result[28] = c21;
result[29] = c15;
result[30] = c19;
result[31] = c18;
result[32] = 0;
result[33] = 0;
result[34] = 0;
result[35] = c17;
result[36] = c20;
result[37] = c15;
result[38] = c21;
result[39] = 0;
result[40] = 0;
result[41] = 0;
