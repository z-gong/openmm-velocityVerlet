/**
 * Perform the first step of Langevin integration.
 */

extern "C" __global__ void updateImagePositions(real4 *__restrict__ posq,
                                                real4 *__restrict__ posqCorrection,
                                                const int2 *__restrict__ imagePairs,
                                                mixed mirror) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_IMAGES; i += blockDim.x * gridDim.x) {
        int2 pair = imagePairs[i];
        int index_img = pair.x;
        int index_par = pair.y;
//        if (i==0)
//            printf("Mirror = %f; Pair = %d %d, Index= %d %d, z = %f %f\n",
//                    mirror, pair.x, pair.y, index_img, index_par, posq[index_img].z, posq[index_par].z);
        posq[index_img].x = posq[index_par].x;
        posq[index_img].y = posq[index_par].y;
        posqCorrection[index_img].x = posqCorrection[index_par].x;
        posqCorrection[index_img].y = posqCorrection[index_par].y;

#ifdef USE_MIXED_PRECISION
        mixed z = posq[index_par].z + (mixed) posqCorrection[index_par].z;
        z = mirror * 2 - z;
        posq[index_img].z = (real) z;
        posqCorrection[index_img].z = (real) (z - (real) z);
#else
        posq[index_img].z = 2 * mirror - pos.z;
#endif
    }
}
