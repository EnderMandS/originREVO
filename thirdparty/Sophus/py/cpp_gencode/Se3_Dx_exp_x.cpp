Scalar const c0 = pow(omega[1], 2);
Scalar const c1 = -c0;
Scalar const c2 = pow(omega[2], 2);
Scalar const c3 = -c2;
Scalar const c4 = c1 + c3;
Scalar const c5 = pow(omega[0], 2);
Scalar const c6 = c0 + c2 + c5;
Scalar const c7 = pow(c6, -3.0L / 2.0L);
Scalar const c8 = sqrt(c6);
Scalar const c9 = sin(c8);
Scalar const c10 = c8 - c9;
Scalar const c11 = c10 * c7;
Scalar const c12 = 1.0 / c6;
Scalar const c13 = cos(c8);
Scalar const c14 = -c13 + 1;
Scalar const c15 = c12 * c14;
Scalar const c16 = c15 * omega[2];
Scalar const c17 = c11 * omega[0];
Scalar const c18 = c17 * omega[1];
Scalar const c19 = c15 * omega[1];
Scalar const c20 = c17 * omega[2];
Scalar const c21 = -c5;
Scalar const c22 = c21 + c3;
Scalar const c23 = c15 * omega[0];
Scalar const c24 = omega[1] * omega[2];
Scalar const c25 = c11 * c24;
Scalar const c26 = c1 + c21;
Scalar const c27 = 1.0 / c8;
Scalar const c28 = 0.5 * c8;
Scalar const c29 = sin(c28);
Scalar const c30 = c27 * c29;
Scalar const c31 = c29 * c7;
Scalar const c32 = cos(c28);
Scalar const c33 = 0.5 * c12 * c32;
Scalar const c34 = c29 * c7 * omega[0];
Scalar const c35 = 0.5 * c12 * c32 * omega[0];
Scalar const c36 = -c34 * omega[1] + c35 * omega[1];
Scalar const c37 = -c34 * omega[2] + c35 * omega[2];
Scalar const c38 = c27 * omega[0];
Scalar const c39 = 0.5 * c29;
Scalar const c40 = pow(c6, -5.0L / 2.0L);
Scalar const c41 = 3 * c10 * c40 * omega[0];
Scalar const c42 = c4 * c7;
Scalar const c43 = -c13 * c38 + c38;
Scalar const c44 = c7 * c9 * omega[0];
Scalar const c45 = c44 * omega[1];
Scalar const c46 = pow(c6, -2);
Scalar const c47 = 2 * c14 * c46 * omega[0];
Scalar const c48 = c47 * omega[1];
Scalar const c49 = c11 * omega[2];
Scalar const c50 = c45 - c48 + c49;
Scalar const c51 = 3 * c10 * c40 * c5;
Scalar const c52 = c7 * omega[0] * omega[2];
Scalar const c53 = c43 * c52 - c51 * omega[2];
Scalar const c54 = c7 * omega[0] * omega[1];
Scalar const c55 = c43 * c54 - c51 * omega[1];
Scalar const c56 = c44 * omega[2];
Scalar const c57 = c47 * omega[2];
Scalar const c58 = c11 * omega[1];
Scalar const c59 = -c56 + c57 + c58;
Scalar const c60 = -2 * c17;
Scalar const c61 = c22 * c7;
Scalar const c62 = -c24 * c41;
Scalar const c63 = -c15 + c62;
Scalar const c64 = c7 * c9;
Scalar const c65 = c5 * c64;
Scalar const c66 = 2 * c14 * c46;
Scalar const c67 = c5 * c66;
Scalar const c68 = c7 * omega[1] * omega[2];
Scalar const c69 = c43 * c68;
Scalar const c70 = c56 - c57 + c58;
Scalar const c71 = c26 * c7;
Scalar const c72 = c15 + c62;
Scalar const c73 = -c45 + c48 + c49;
Scalar const c74 = -c24 * c31 + c24 * c33;
Scalar const c75 = c27 * omega[1];
Scalar const c76 = -2 * c58;
Scalar const c77 = 3 * c10 * c40 * omega[1];
Scalar const c78 = -c13 * c75 + c75;
Scalar const c79 = c0 * c64;
Scalar const c80 = c0 * c66;
Scalar const c81 = c52 * c78;
Scalar const c82 = -c0 * c41 + c54 * c78;
Scalar const c83 = c24 * c64;
Scalar const c84 = c24 * c66;
Scalar const c85 = c17 - c83 + c84;
Scalar const c86 = c17 + c83 - c84;
Scalar const c87 = 3 * c10 * c40 * omega[2];
Scalar const c88 = -c0 * c87 + c68 * c78;
Scalar const c89 = c27 * omega[2];
Scalar const c90 = -2 * c49;
Scalar const c91 = -c13 * c89 + c89;
Scalar const c92 = c2 * c64;
Scalar const c93 = c2 * c66;
Scalar const c94 = c54 * c91;
Scalar const c95 = -c2 * c41 + c52 * c91;
Scalar const c96 = -c2 * c77 + c68 * c91;
result[0] = 0;
result[1] = 0;
result[2] = 0;
result[3] = 0;
result[4] = c11 * c4 + 1;
result[5] = c16 + c18;
result[6] = -c19 + c20;
result[7] = 0;
result[8] = 0;
result[9] = 0;
result[10] = 0;
result[11] = -c16 + c18;
result[12] = c11 * c22 + 1;
result[13] = c23 + c25;
result[14] = 0;
result[15] = 0;
result[16] = 0;
result[17] = 0;
result[18] = c19 + c20;
result[19] = -c23 + c25;
result[20] = c11 * c26 + 1;
result[21] = c30 - c31 * c5 + c33 * c5;
result[22] = c36;
result[23] = c37;
result[24] = -c38 * c39;
result[25] = upsilon[0] * (-c4 * c41 + c42 * c43) + upsilon[1] * (c55 + c59) +
             upsilon[2] * (c50 + c53);
result[26] = upsilon[0] * (c55 + c70) +
             upsilon[1] * (-c22 * c41 + c43 * c61 + c60) +
             upsilon[2] * (c63 - c65 + c67 + c69);
result[27] = upsilon[0] * (c53 + c73) + upsilon[1] * (c65 - c67 + c69 + c72) +
             upsilon[2] * (-c26 * c41 + c43 * c71 + c60);
result[28] = c36;
result[29] = -c0 * c31 + c0 * c33 + c30;
result[30] = c74;
result[31] = -c39 * c75;
result[32] = upsilon[0] * (-c4 * c77 + c42 * c78 + c76) +
             upsilon[1] * (c82 + c85) + upsilon[2] * (c72 + c79 - c80 + c81);
result[33] = upsilon[0] * (c82 + c86) + upsilon[1] * (-c22 * c77 + c61 * c78) +
             upsilon[2] * (c73 + c88);
result[34] = upsilon[0] * (c63 - c79 + c80 + c81) + upsilon[1] * (c50 + c88) +
             upsilon[2] * (-c26 * c77 + c71 * c78 + c76);
result[35] = c37;
result[36] = c74;
result[37] = -c2 * c31 + c2 * c33 + c30;
result[38] = -c39 * c89;
result[39] = upsilon[0] * (-c4 * c87 + c42 * c91 + c90) +
             upsilon[1] * (c63 - c92 + c93 + c94) + upsilon[2] * (c86 + c95);
result[40] = upsilon[0] * (c72 + c92 - c93 + c94) +
             upsilon[1] * (-c22 * c87 + c61 * c91 + c90) +
             upsilon[2] * (c59 + c96);
result[41] = upsilon[0] * (c85 + c95) + upsilon[1] * (c70 + c96) +
             upsilon[2] * (-c26 * c87 + c71 * c91);
